"""Dialogue and reference-summary generation from specs, with code checks that throw bad dialogues away.

The writer model only turns a spec into words. Every value the call must contain is in the spec, and
`check_dialogue` rejects a dialogue that drops one of them or invents another number, date or time.
"""

# [파일 역할] 데이터 만들기의 두 번째·세 번째 단계다. 명세를 교사 모델(Ollama qwen2.5 14B)에게 주어 상담
# 대화와 참고 요약을 쓰게 하고, 코드 검사(check_dialogue, check_summary)를 통과한 것만
# data/raw/<분할>.jsonl에, 통과하지 못한 명세는 .rejects.jsonl에 남긴다.
# [왜 필요한가] 재배포할 수 있는 한국어 상담 요약 데이터를 찾지 못해 데이터를 직접 만든다(docs/plan.md).
# 정답은 명세에서 나오므로 교사는 "명세를 말로 옮기는 일"만 한다. 대신 교사가 값을 빠뜨리거나 명세에 없는
# 번호·금액을 지어내면 정답과 대화가 어긋나므로, 그런 대화를 코드로 걸러 버린다(README "데이터를 만드는 방식"
# 3).
# [설계 이유] 교사는 파일럿에서 14B가 7B보다 통과율이 높고 대화가 자연스러워 골랐다(docs/experiments.md 1단계
# 파일럿). 대화는 temperature 0.8로 다양하게, 요약은 0.3으로 차분하게 쓰고, 명세마다 고정 seed로 다시 돌려도
# 같은 결과가 나오게 한다. 이어 하기가 되어 수십 시간 걸린 생성을 끊었다 다시 시작할 수 있다.
# [연결] 입력은 build_specs의 명세 JSONL, 출력은 finalize.py가 읽는다. finalize는 이 파일의 check_dialogue를
# 다시 불러 모든 분할에 같은 검사를 적용한다.

from __future__ import annotations

# zlib: 명세 id에서 crc32로 seed를 만든다. 파이썬 내장 hash()는 프로세스마다 값이 바뀌어 다시 돌릴 때 같은
# seed를 보장하지 못하므로 쓰지 않는다.
import argparse
import json
import re
import sys
import time
import zlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

# values의 normalize·occurs_in_transcript·values_in_transcript는 채점기와 같은 값 정규화다. 생성 검사와 채점이
# 같은 규칙으로 금액·날짜·번호를 읽어야, 검사를 통과한 대화의 값이 채점에서도 같은 값으로 잡힌다.
from .dataset import Item, read_jsonl
from .domains import DOMAINS
from .providers import OllamaProvider, Provider
from .specs import (
    EV_COMPLAINT,
    EV_HOLD,
    EV_REPEAT,
    EV_SMALLTALK,
    EV_VERIFY,
    Spec,
)
from .values import normalize, occurs_in_transcript, values_in_transcript

# WRITER_VERSION: 대화 지시문의 판. w2는 명세에 본인 확인이 없을 때 "본인 확인 절차는 하지 않는다"를 명시한
# 판이다(2026-09-27, 커밋 f4d0efe). 생성 시점에 따라 w1과 w2가 섞이지만 finalize가 모든 분할에 같은 검사를
# 다시 적용한다(docs/experiments.md 1단계).
# SPEAKERS: 허용하는 화자 두 가지. 서비스 입력 검사도 같은 두 값만 받는다.
WRITER_VERSION = "w2"  # w1: same, minus the explicit "no verification" line
SPEAKERS = ("상담원", "고객")

# 명세의 대화 사건 → 교사에게 줄 지시 문장. 사건은 specs.py가 확률로 뽑는다(번호 말실수, 불만, 대기, 되묻기,
# 잡담 등). 말실수(correction)와 추가 질문은 값이 들어가 writer_prompt에서 따로 만든다.
_EVENT_TEXT = {
    EV_VERIFY: "상담원이 처리 전에 본인 확인을 한다. 고객 이름과 생년월일 여섯 자리를 묻고, 고객이 답한다.",
    EV_COMPLAINT: "고객이 짜증이나 불만을 드러내고, 상담원이 사과하며 달랜다.",
    EV_HOLD: "상담원이 확인하는 동안 잠시 기다려 달라고 하고, 조금 뒤 돌아와 말을 잇는다.",
    EV_REPEAT: "고객이 이미 들은 내용을 한 번 더 되묻고, 상담원이 다시 설명한다.",
    EV_SMALLTALK: "고객이 본론 전에 날씨나 연결이 오래 걸렸다는 등 짧은 잡담을 한다.",
}

# 대화 작성 지시문. 명세의 사실·끝맺음·반드시 나올 값·말실수·사건을 채운다. 값은 "글자 그대로" 쓰라고 해 표기
# 변형("3만 2천 원" ↔ "32,000원")으로 검사에서 떨어지지 않게 한다. 위 값 말고 어떤 금액·날짜·번호도 말하지
# 말라는 규칙은 정답에 없는 값이 대화에 섞이는 것을 막는다(그래도 섞이면 check_dialogue가 버린다). 12~24턴을
# 지시하고 검사는 8~40턴을 허용한다(1단계 규칙).
_WRITER = """다음 조건으로 {company}({label}) 고객센터의 전화 상담 대화를 한국어로 써 주세요.

[상담 내용]
{facts}
{note}
[대화에 반드시 그대로 나와야 하는 값] (글자 그대로 쓰세요. 다른 표기로 바꾸지 마세요)
{values}
{distractor}
[대화 중 일어나는 일]
{events}

[규칙]
- 상담원이 먼저 인사하며 시작하고, 상담원과 고객이 번갈아 말한다. 전체 12~24턴.
- 실제 전화 상담처럼 자연스러운 구어체로 쓴다. 한 턴은 1~3문장.
- 위에 적힌 값 말고는 어떤 금액, 날짜, 시각, 주문번호·운송장번호·접수번호 같은 번호도 말하지 않는다. (생년월일 여섯 자리는 예외)
- 상담원이 한 처리는 [상담 내용]에 적힌 것뿐이다. 다른 처리를 했다고 말하지 않는다.
- 상담 내용의 결과가 대화에서 분명히 드러나야 한다.
- 실제 회사 이름이나 실존 인물 이름을 쓰지 않는다.

JSON으로만 답하세요: {{"turns": [{{"speaker": "상담원" 또는 "고객", "text": "..."}}, ...]}}"""

# 참고 요약 작성 지시문. 요약은 학습 목표 JSON의 summary 칸에 들어간다(채점은 요약문이 아니라 사실 목록으로
# 한다). 핵심 값을 대화 그대로 쓰라고 하고, 본인 확인 세부 값(이름·생년월일)은 넣지 말라고 한다. 그래도 교사
# 요약의 60%가 고객 이름을 담았는데, 대화에 있는 가상 이름이고 채점과 무관해 그대로 두었다(docs/experiments.md
# 1단계).
_SUMMARIZER = """다음은 {company}({label}) 고객센터의 상담 대화입니다. 상담 기록에 들어갈 요약을 3~5문장의 한국어로 써 주세요.

- 고객의 요청, 상담원이 확인하거나 처리한 것, 결과와 후속 조치를 담는다.
- 대화에 나온 핵심 값(번호, 금액, 날짜, 시각, 상품·요금제 이름)은 대화에 나온 그대로 쓴다.
- 대화에 없는 내용은 쓰지 않는다. 인사말, 본인 확인 절차의 세부 값(이름, 생년월일)은 쓰지 않는다.
- "고객은 ~함", "상담원은 ~함"처럼 기록체로 쓴다.

[상담 대화]
{transcript}

JSON으로만 답하세요: {{"summary": "..."}}"""

# 교사 답을 Ollama 스키마 강제 디코딩으로 묶는 JSON 스키마(main에서 use_schema=True). 대화는 {turns:
# [{speaker, text}]}, 요약은 {summary}. 화자는 두 값 중 하나로 제한한다.
TURNS_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": list(SPEAKERS)},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
            },
        }
    },
    "required": ["turns"],
}
SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


# 명세 하나로 대화 지시문을 만든다. 말실수 사건이 있으면 "처음엔 틀린 번호를 말했다가 조회가 안 된다고 하자
# 고친다"는 장면을 넣는다(정답에는 고친 값만 들어간다). 추가 질문은 숫자 없이 답하게 해 새 값이 생기지 않게
# 한다.
def writer_prompt(spec: Spec) -> str:
    d = DOMAINS[spec.domain]
    values = "\n".join(f"- {sv.type}: {sv.value}" for sv in spec.slots)
    distractor = ""
    if spec.distractor is not None:
        right = next(sv.value for sv in spec.slots if sv.slot == spec.distractor.slot)
        distractor = (
            f"\n[말실수] 고객이 {spec.distractor.type}를 처음에 '{spec.distractor.value}'라고 잘못 말했다가, "
            f"상담원이 조회가 안 된다고 하자 '{right}'로 고쳐 말한다.\n"
        )
    events = [_EVENT_TEXT[e] for e in spec.events if e in _EVENT_TEXT]
    if spec.extra_question:
        events.append(
            f"상담이 거의 끝날 때 고객이 '{spec.extra_question}'에 대해 짧게 하나 더 묻고, 상담원이 숫자 없이 간단히 답한다."
        )
    # 본인 확인은 정답 처리 목록에 들어가는 항목이라, 명세에 없으면 대화에서도 하지 않도록 명시한다(w2에서
    # 추가). dev 표본에서 명세에 없는 본인 확인이 대화에 나온 라벨 잡음을 찾아 넣은 줄이다(docs/experiments.md
    # 1단계).
    if EV_VERIFY not in spec.events:
        events.append("본인 확인 절차(이름·생년월일·주민번호 묻기)는 하지 않는다.")
    if not events:
        events.append("특별한 일 없이 차분하게 진행된다.")
    return _WRITER.format(
        company=d.company,
        label=d.label,
        facts="\n".join(f"- {f}" for f in spec.facts),
        note=f"- 끝맺음: {spec.note}\n" if spec.note else "",
        values=values,
        distractor=distractor,
        events="\n".join(f"- {e}" for e in events),
    )


# 통과한 대화의 전사로 요약 지시문을 만든다. 요약은 대화가 확정된 뒤에만 쓴다.
def summarizer_prompt(spec: Spec, transcript: str) -> str:
    d = DOMAINS[spec.domain]
    return _SUMMARIZER.format(company=d.company, label=d.label, transcript=transcript)


# _VERIFY_WORDS: 대화에서 본인 확인을 했는지 판단하는 어휘(생년월일, 주민번호, 본인 확인, 뒷자리 등).
# _FOREIGN_SCRIPT: 히라가나·가타카나·한자 범위. 파일럿에서 14B가 상담원 발화 한 턴을 중국어로 바꾼 일이 있어
# 넣었다(커밋 e98336e).
# _ID_LIKE: 번호처럼 보이는 문자열(영문 대문자 + 5자리 이상 숫자, 또는 하이픈으로 이은 숫자 묶음). 명세에 없는
# 번호를 찾는 데 쓴다. 앞뒤가 숫자·영문·하이픈이면 더 긴 문자열의 일부이므로 잡지 않는다.
_VERIFY_WORDS = re.compile(r"생년월일|주민등록|주민번호|본인 ?확인|뒷자리|뒤 ?[네4] ?자리")
# The writer model sometimes switches to Chinese mid-dialogue.
_FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_ID_LIKE = re.compile(r"(?<![\dA-Za-z\-])(?:[A-Z]\d{5,}|\d{2,5}(?:-\d{2,5}){1,3})(?![\dA-Za-z\-])")


# 생성된 대화를 쓸 수 없게 만드는 문제 목록을 돌려준다(빈 목록이면 통과). 생성 중 거르기와 finalize의 재검사가
# 함께 쓴다.
# 검사 순서: 형식(턴 수 8~40, 화자, 상담원이 먼저, 두 화자 모두, 같은 화자 연속 3번 이상 금지) → 외국 문자 →
# 본인 확인 일치 → 명세 값이 글자 그대로 모두 있는지(말실수 값 포함) → 명세에 없는 금액·날짜·시각·번호가
# 없는지.
def check_dialogue(spec: Spec, turns: Sequence[dict]) -> list[str]:
    """Problems that make a generated dialogue unusable. Empty list means it passes."""
    problems: list[str] = []
    if not 8 <= len(turns) <= 40:
        problems.append(f"turn count {len(turns)}")
    if not turns:
        return problems
    # 화자 이름이 틀렸거나 빈 발화가 있으면 뒤 검사가 의미 없으므로 바로 돌려준다.
    if any(t.get("speaker") not in SPEAKERS or not str(t.get("text", "")).strip() for t in turns):
        problems.append("bad speaker or empty turn")
        return problems
    if turns[0]["speaker"] != "상담원":
        problems.append("does not start with the agent")
    if {t["speaker"] for t in turns} != set(SPEAKERS):
        problems.append("one speaker only")
    # 같은 화자가 연달아 말한 경우는 두 번까지 허용하고, 그보다 많으면 대화 꼴이 무너진 것으로 본다.
    same_in_row = sum(1 for a, b in zip(turns, turns[1:], strict=False) if a["speaker"] == b["speaker"])
    if same_in_row > 2:
        problems.append(f"{same_in_row} consecutive same-speaker turns")

    text = "\n".join(t["text"] for t in turns)
    if _FOREIGN_SCRIPT.search(text):
        problems.append("non-Korean script (Han/kana)")
    # Identity verification happens exactly when the spec says so (it is a gold action).
    verified = bool(_VERIFY_WORDS.search(text))
    if verified and EV_VERIFY not in spec.events:
        problems.append("verification not in spec")
    if not verified and EV_VERIFY in spec.events:
        problems.append("spec verification missing")
    d = DOMAINS[spec.domain]
    # 명세 값은 글자 그대로 들어 있어야 한다(정규화하지 않고 부분 문자열로 확인). 말실수 값도 대화에 나와야
    # 고친 값만 기록하는 능력을 시험할 수 있다.
    for sv in spec.slots:
        if sv.value not in text:
            problems.append(f"missing {sv.type} {sv.value!r}")
    if spec.distractor is not None and spec.distractor.value not in text:
        problems.append(f"missing distractor {spec.distractor.value!r}")

    # No numbers, dates or times beyond the spec's.
    # 허용 값 집합: 명세 값과 말실수 값을 종류별로 정규화해 둔다. 대화에서 읽어 낸 금액·날짜·시각 중 이 집합에
    # 없는 것은 교사가 지어낸 값이다.
    allowed: dict[str, set[str]] = {}
    for sv in list(spec.slots) + ([spec.distractor] if spec.distractor else []):
        kind = d.entity_type(sv.type).kind
        norm = normalize(kind, sv.value)
        if norm is not None:
            allowed.setdefault(kind, set()).add(norm)
    for kind in ("amount", "date", "time"):
        extra = values_in_transcript(kind, text) - allowed.get(kind, set())
        # Amount suffix readings of an allowed amount ("2천" inside "3만 2천 원") are not new values.
        if kind == "amount":
            extra = {v for v in extra if not any(a.endswith(v) for a in allowed.get("amount", set()))}
        if extra:
            problems.append(f"extra {kind} {sorted(extra)}")
    # 번호 검사: 번호처럼 보이는 문자열마다, 정규화한 값이 허용 번호 중 하나의 부분 문자열이면 통과다. 첫
    # 파일럿에서 교사가 긴 번호의 앞부분만 말해 거절이 많았기 때문에, 허용 번호를 줄여 말한 것은 새 번호로
    # 치지 않는다(docs/experiments.md 1단계 파일럿).
    allowed_ids = {normalize("id", v) for v in (sv.value for sv in spec.slots)} | (
        {normalize("id", spec.distractor.value)} if spec.distractor else set()
    )
    for m in _ID_LIKE.finditer(text):
        # A shortened mention of an allowed number ("5891-6890 끝나는 거요") is not a new number.
        norm = normalize("id", m.group(0)) or ""
        if not any(norm in a for a in allowed_ids if a):
            problems.append(f"extra id {m.group(0)!r}")
    return problems


# 참고 요약 검사: 외국 문자 없음, 문장 수 2~8(마침표·물음표나 "~다/함/음/됨 "으로 끝나는 곳을 센다)과 길이
# 40~600자, 명세의 모든 핵심 값이 (정규화해서) 들어 있음, 말실수로 처음 말한 틀린 번호는 들어 있지 않음.
# 요약은 학습 목표의 일부이므로, 핵심 값이 빠지거나 고치기 전 값이 들어간 요약은 학습에 쓰지 않는다.
def check_summary(spec: Spec, summary: str) -> list[str]:
    problems = []
    if _FOREIGN_SCRIPT.search(summary):
        problems.append("non-Korean script (Han/kana)")
    n = len(re.findall(r"[.!?。]|[다함음됨] ", summary + " "))
    if not 2 <= n <= 8 or not 40 <= len(summary) <= 600:
        problems.append(f"length ({len(summary)} chars)")
    d = DOMAINS[spec.domain]
    for sv in spec.slots:
        if not occurs_in_transcript(d.entity_type(sv.type).kind, sv.value, summary):
            problems.append(f"summary misses {sv.type} {sv.value!r}")
    if spec.distractor is not None and occurs_in_transcript("id", spec.distractor.value, summary):
        problems.append("summary repeats the corrected value")
    return problems


# 교사 답에서 key의 값을 꺼낸다. 답 전체가 JSON이 아니면 첫 '{'부터 마지막 '}'까지를 다시 읽는다. 실패하면
# None.
def _parse_json(text: str, key: str) -> object | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return data.get(key) if isinstance(data, dict) else None


# 시도마다 샘플링 seed를 바꾼다. 처음에는 한 명세 안의 재시도가 같은 seed로 같은 출력을 되풀이해 재시도가 의미
# 없었다(dev 생성 중 발견, 커밋 386632f). dev 거절률(12.8%)이 test-a(5.0%)보다 높은 이유로 기록돼
# 있다(docs/experiments.md 1단계).
def _reseed(provider: Provider, seed: int | None, attempt: int) -> None:
    """A different sampling seed per try, so a retry is not a replay of the rejected output."""
    if seed is not None and hasattr(provider, "seed"):
        provider.seed = seed + attempt  # type: ignore[attr-defined]


# 명세 하나로 (Item 또는 None, 마지막 시도의 문제 목록, 통계)를 만든다. 대화를 최대 max_tries번 시도해 검사를
# 통과하면, 그 대화로 요약을 최대 max_tries번 시도한다. 둘 중 하나라도 끝내 통과하지 못하면 명세 전체를
# 거절한다.
# 통계(시도 횟수, 걸린 초, 실패한 요청 수, 마지막으로 거절된 출력)는 Item.meta나 거절 기록에 남아 거절 이유를
# 살펴볼 수 있다.
def generate_item(
    spec: Spec, writer: Provider, summarizer: Provider, max_tries: int = 3, seed: int | None = None
) -> tuple[Item | None, list[str], dict]:
    """Returns (item or None, problems of the last try, stats)."""
    stats: dict = {"dialogue_tries": 0, "summary_tries": 0, "seconds": 0.0}
    problems: list[str] = []
    turns: list[dict] | None = None
    for attempt in range(max_tries):
        stats["dialogue_tries"] += 1
        _reseed(writer, seed, attempt)
        try:
            reply = writer.generate(
                [{"role": "user", "content": writer_prompt(spec)}], json_schema=TURNS_SCHEMA
            )
        # 요청이 멈추거나 실패하면(시간 초과 등) 생성 전체를 죽이지 않고 시도 한 번으로 센다(커밋 4eb3191).
        except Exception as exc:  # noqa: BLE001 - a hung or failed request is one failed try
            problems = [f"request failed: {type(exc).__name__}"]
            stats["failed_requests"] = stats.get("failed_requests", 0) + 1
            continue
        stats["seconds"] += reply.latency_s
        got = _parse_json(reply.text, "turns")
        if not isinstance(got, list):
            problems = ["unparseable dialogue"]
            continue
        # 답에서 dict인 발화만 골라 화자·내용의 앞뒤 공백을 다듬는다. 형식이 이상한 원소는 여기서 빠지고, 남은
        # 것으로 검사한다.
        cand = [
            {"speaker": str(t.get("speaker", "")).strip(), "text": str(t.get("text", "")).strip()}
            for t in got
            if isinstance(t, dict)
        ]
        problems = check_dialogue(spec, cand)
        if not problems:
            turns = cand
            break
        stats["last_rejected"] = cand
    if turns is None:
        return None, problems, stats

    item = Item(item_id=spec.spec_id, split=spec.split, domain=spec.domain, spec=spec, turns=turns)
    for attempt in range(max_tries):
        stats["summary_tries"] += 1
        # 요약 seed는 100을 더해 대화 시도의 seed와 겹치지 않게 한다.
        _reseed(summarizer, seed, 100 + attempt)
        try:
            reply = summarizer.generate(
                [{"role": "user", "content": summarizer_prompt(spec, item.transcript)}],
                json_schema=SUMMARY_SCHEMA,
            )
        except Exception as exc:  # noqa: BLE001
            problems = [f"request failed: {type(exc).__name__}"]
            stats["failed_requests"] = stats.get("failed_requests", 0) + 1
            continue
        stats["seconds"] += reply.latency_s
        summary = _parse_json(reply.text, "summary")
        if not isinstance(summary, str):
            problems = ["unparseable summary"]
            continue
        summary = summary.strip()
        problems = check_summary(spec, summary)
        if not problems:
            item.summary = summary
            stats.pop("last_rejected", None)
            return item, [], stats
        stats["last_rejected"] = summary
    return None, problems, stats


# 명령줄 진입점. 출력 파일에 이어 쓰며, 이미 통과했거나 거절된 명세는 건너뛴다(거절된 명세를 다시 시도하지
# 않는다).
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", required=True, help="spec JSONL (from build_specs)")
    ap.add_argument("--out", required=True, help="item JSONL; appended to, already-done spec ids are skipped")
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int)
    # 컨텍스트 4,096: 8,192에서는 14B의 층 일부가 CPU로 밀려 건당 약 100초가 걸렸고, 4,096이면 GPU에 다
    # 올라간다(docs/experiments.md 1단계 진행 기록, 커밋 3220664).
    # --num-gpu: 층을 모두 올리면 VRAM이 차서 Windows가 시스템 메모리로 밀어내 아주 느려지므로 44층으로 여유를
    # 남겼다(나중에 다른 GPU 앱을 닫고 47층).
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument(
        "--num-gpu", type=int, help="GPU layers; leave VRAM headroom (44 of 49 for the 14B teacher on 11 GB)"
    )
    ap.add_argument(
        "--timeout", type=float, default=300.0, help="seconds per request; a timeout counts as a failed try"
    )
    args = ap.parse_args(argv)

    specs = [Spec.from_dict(d) for d in read_jsonl(args.specs)]
    out = Path(args.out)
    rejects = out.with_suffix(".rejects.jsonl")
    done = {d["item_id"] for d in read_jsonl(out)} if out.exists() else set()
    done |= {d["spec_id"] for d in read_jsonl(rejects)} if rejects.exists() else set()
    todo = [s for s in specs if s.spec_id not in done][: args.limit]
    print(f"{len(todo)} to generate ({len(done)} already done)", file=sys.stderr)

    # 대화용과 요약용 실행기는 같은 모델·스키마 강제·컨텍스트를 쓰고, temperature(0.8 / 0.3)와 답 상한(2,048 /
    # 512토큰)만 다르다(1단계 규칙).
    common = {"model": args.model, "use_schema": True, "num_ctx": args.num_ctx, "num_gpu": args.num_gpu}
    writer = OllamaProvider(temperature=args.temperature, num_predict=2048, timeout_s=args.timeout, **common)
    summarizer = OllamaProvider(temperature=0.3, num_predict=512, timeout_s=args.timeout, **common)
    counts: Counter = Counter()
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 통과 파일과 거절 파일을 이어 쓰기로 열고, 한 건마다 flush해 중간에 멈춰도 끝난 건은 남는다.
    with (
        open(out, "a", encoding="utf-8", newline="\n") as f_ok,
        open(rejects, "a", encoding="utf-8", newline="\n") as f_bad,
    ):
        for n, spec in enumerate(todo, 1):
            # A different sampling seed per spec keeps runs reproducible yet varied.
            item, problems, stats = generate_item(
                spec, writer, summarizer, seed=zlib.crc32(spec.spec_id.encode())
            )
            if item is not None:
                # 누가 쓴 대화인지(교사 모델 이름)와 지시문 판, 생성 통계를 함께 남긴다. test-b는 Claude가 쓴
                # 대화라 source가 다르다.
                item.source = f"teacher:{args.model}"
                item.meta = {"writer_version": WRITER_VERSION, **stats}
                f_ok.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                f_ok.flush()
                counts["ok"] += 1
            else:
                f_bad.write(
                    json.dumps({"spec_id": spec.spec_id, "problems": problems, **stats}, ensure_ascii=False)
                    + "\n"
                )
                f_bad.flush()
                counts["rejected"] += 1
            # 10건마다 진행 상황과 건당 평균 초를 표준 오류로 찍는다(생성이 수 시간~하루 걸려 속도 저하를 바로
            # 알아차리려는 것).
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"[{n}/{len(todo)}] {dict(counts)} {rate:.1f}s/item", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
