"""판정 모델로 요약을 채점한다: 명세의 필수 사실마다 포함 / 누락 / 틀림을 매기고, 대화와 어긋나는 문장
수를 센다.

판정 모델 자체의 신뢰도는 그 수치를 쓰기 전에 dev에서 사람 채점과 비교해 잰다(docs/experiments.md 2단계).
"""

# [파일 역할] 요약 채점의 핵심 모듈이다. 판정 모델(LLM)에게 대화, 명세의 필수 사실 목록, 채점할 요약을 주고
# 사실마다 포함/누락/틀림을 받아 SummaryScore로 만든다. 사실 재현율·오답 비율·채점자 간 일치율·Cohen's kappa
# 같은 집계 함수도 여기 있다.
# [왜 필요한가] 구조 필드는 코드로 채점하지만 요약은 표현이 자유로워 문자열 비교로는 채점할 수 없다. 그래서
# 정답 요약문이 아니라 명세에서 나온 "요약에 들어가야 할 사실" 목록과 대조하게 했다(docs/plan.md "요약 정답").
# 판정 모델은 교사와 같은 qwen2.5 14B라 생기는 편향을 줄이려고 판정을 사실 목록 대조로 좁혔다(docs/plan.md).
# [신뢰도] 2단계에서 판정 j1은 사실 단위 일치율 96.3%였지만 사실 대부분이 '포함'이라 kappa가 약 0.32에 그쳐
# 요약 지표는 참고치로만 썼다. 2026-10-08 재보정에서는 이 프롬프트와 스키마(j1)를 그대로 두고 판정 모델만 바꿔
# 다시 보정했다(docs/experiments.md "요약 판정 재보정").
# [연결] judge_run.py(실행 하나를 판정)와 judge_calib.py(보정 세트 판정·일치 표)가 make_judge, judge_reply,
# reply_fields와 집계 함수를 쓴다.

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

# OllamaProvider: 판정 모델을 Ollama로 부르는 실행기. Provider/Reply: 실행기 공통 인터페이스와 답(본문, 토큰
# 수, 지연 등). Spec: 사실 목록(facts)을 가진 명세.
from .providers import OllamaProvider, Provider, Reply
from .specs import Spec

# JUDGE_VERSION: 판정 프롬프트·요청의 판 이름. 판정 결과 파일의 메타에 남는다. 재보정에서도 바꾸지 않았다.
# VERDICTS: 사실 하나에 줄 수 있는 판정 세 가지.
JUDGE_VERSION = "j1"
VERDICTS = ("포함", "누락", "틀림")

# 판정 프롬프트. 사실마다 판정을 매기게 하고(표현이 달라도 뜻이 같으면 포함), 사실 목록과 별개로 대화와
# 어긋나는 문장 수(wrong_statements)를 세게 한다. 배열 길이를 사실 수와 같게 하라고 명시하고, 아래
# judge_schema로 길이를 강제한다. JSON 예시의 중괄호는 str.format이 읽지 않도록 두 겹으로 적었다.
_JUDGE = """상담 요약을 채점합니다.

[상담 대화]
{transcript}

[요약에 들어가야 할 사실]
{facts}

[채점할 요약]
{summary}

각 사실마다 요약이 그 사실을 담고 있으면 "포함", 빠뜨렸으면 "누락", 다르게(틀린 값이나 반대 내용으로) 적었으면 "틀림"으로 매기세요. 표현이 달라도 뜻이 같으면 "포함"입니다.
그리고 위 사실 목록과 상관없이, 요약에 상담 대화와 어긋나는 내용이 몇 개 있는지 세어 wrong_statements에 적으세요.

JSON으로만 답하세요: {{"facts": ["포함" | "누락" | "틀림", ...], "wrong_statements": 정수}}
facts 배열의 길이는 사실 목록의 개수({n})와 같아야 합니다."""


# 판정 답의 JSON 스키마. Ollama의 스키마 강제 디코딩(format)에 넘겨 facts 배열의 원소를 세 판정 중 하나로,
# 길이를 사실 수 n과 정확히 같게 묶는다. 서빙 학생은 스키마 강제가 키 순서를 바꿔 기본으로 껐지만(README
# 5단계), 판정 요청은 2단계 j1부터 스키마 강제를 켠 꼴 그대로다(make_judge의 use_schema=True).
def judge_schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {"type": "string", "enum": list(VERDICTS)},
                "minItems": n,
                "maxItems": n,
            },
            "wrong_statements": {"type": "integer", "minimum": 0},
        },
        "required": ["facts", "wrong_statements"],
    }


# 요약 한 건의 판정 결과. verdicts는 사실마다 판정(판정 실패면 빈 튜플), wrong_statements는 대화와 어긋난 문장
# 수, ok는 판정 답을 쓸 수 있었는지다. frozen으로 만들어 집계 중에 값이 바뀌지 않게 한다.
@dataclass(frozen=True)
class SummaryScore:
    item_id: str
    verdicts: tuple[str, ...]  # 필수 사실마다 하나. 판정이 실패하면 비어 있다
    wrong_statements: int
    ok: bool  # 판정 답을 쓸 수 있었는지

    # 이 건의 사실 재현율(포함 비율). 판정이 없으면 NaN으로 두어 평균에 섞이지 않게 한다.
    @property
    def recall(self) -> float:
        return self.verdicts.count("포함") / len(self.verdicts) if self.verdicts else float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


# 명세의 사실을 번호 목록으로 만들어 프롬프트를 채운다. 요약이 공백뿐이면 "(빈 요약)"으로 적는다(실제로는
# judge_reply가 빈 요약을 판정 모델에 보내지 않는다).
def judge_prompt(spec: Spec, transcript: str, summary: str) -> str:
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(spec.facts))
    return _JUDGE.format(
        transcript=transcript, facts=facts, summary=summary.strip() or "(빈 요약)", n=len(spec.facts)
    )


# 판정 답 글에서 첫 '{'부터 마지막 '}'까지를 JSON으로 읽어 (판정 튜플, 어긋난 문장 수)를 돌려준다. 배열 길이가
# 사실 수와 다르거나, 세 판정 밖의 값이 있거나, wrong_statements가 음이 아닌 정수가 아니면 None(판정 실패)을
# 돌려준다. 스키마 강제를 켜도 답이 잘리거나 형식이 깨질 수 있어 한 번 더 검사한다.
def parse_judge(text: str, n: int) -> tuple[tuple[str, ...], int] | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    facts = data.get("facts")
    wrong = data.get("wrong_statements")
    if not isinstance(facts, list) or len(facts) != n or any(v not in VERDICTS for v in facts):
        return None
    if not isinstance(wrong, int) or wrong < 0:
        return None
    return tuple(facts), wrong


# 판정 요청 설정(2단계 j1의 판정 요청): 스키마 강제 JSON, 컨텍스트 4,096토큰, 답 상한 256토큰(판정 답은
# 짧은 배열 하나다). 재보정에서 바꾸는 것은 판정 모델뿐이고 이 요청 꼴은 그대로 두도록 정했으므로 상수로
# 고정해 둔다(docs/experiments.md "요약 판정 재보정").
JUDGE_NUM_CTX = 4096
JUDGE_NUM_PREDICT = 256


# 판정 모델용 OllamaProvider를 만든다. 모델 태그, GPU 층 수, 생각 모드, 더할 Ollama 옵션만 바꿀 수 있고
# 나머지(스키마 강제, 컨텍스트, 답 상한)는 j1과 같다.
# num_gpu: 14B(J0)는 VRAM이 넘치지 않게 44층만 GPU에 올렸다(j1과 같음, docs/experiments.md 재보정 규칙 3의
# 표). think: 생각 모드가 기본으로 켜진 2026년 모델(qwen3.5:9b 등)은 끈다. options: 태그 기본
# 벌점(presence_penalty 등)을 끄는 데 쓴다(재보정 규칙 3).
# 호출 위치: judge_run.py, judge_calib.py.
def make_judge(*, model: str, num_gpu: int | None, think: bool | None, options: dict) -> OllamaProvider:
    return OllamaProvider(
        model=model,
        use_schema=True,
        num_ctx=JUDGE_NUM_CTX,
        num_predict=JUDGE_NUM_PREDICT,
        num_gpu=num_gpu,
        think=think,
        extra_options=dict(options) or None,
    )


# 요약 한 건을 판정해 (SummaryScore, 판정 모델의 원 답)을 돌려준다. 원 답을 함께 돌려주는 것은 토큰
# 수·잘림·생각 글 같은 측정 조건 확인값(reply_fields)을 판정 옆에 남기기 위해서다.
# 요약이 비어 있으면 판정 모델을 부르지 않고 모든 사실을 누락으로 친다(규칙: "요약이 빈 건은 판정 모델을
# 부르지 않고 모두 누락"). 답을 해석하지 못하면 ok=False인 점수를 돌려준다.
def judge_reply(
    judge: Provider, item_id: str, spec: Spec, transcript: str, summary: str
) -> tuple[SummaryScore, Reply | None]:
    """점수와 판정 모델의 원 답(판정 모델을 부르지 않은 빈 요약이면 None)."""
    n = len(spec.facts)
    if not summary.strip():
        return SummaryScore(item_id, ("누락",) * n, 0, True), None
    reply = judge.generate(
        [{"role": "user", "content": judge_prompt(spec, transcript, summary)}], json_schema=judge_schema(n)
    )
    parsed = parse_judge(reply.text, n)
    if parsed is None:
        return SummaryScore(item_id, (), 0, False), reply
    return SummaryScore(item_id, parsed[0], parsed[1], True), reply


# 원 답이 필요 없는 호출용 줄임 함수(점수만 돌려준다). 테스트(tests/test_gen.py)가 쓴다.
def judge_summary(judge: Provider, item_id: str, spec: Spec, transcript: str, summary: str) -> SummaryScore:
    return judge_reply(judge, item_id, spec, transcript, summary)[0]


# 판정 한 건 옆에 남길 확인값: 프롬프트·답 토큰 수, 끝난 이유(done_reason이 "length"면 답 상한에서 잘린 것),
# 지연, 생각 글 글자 수. 재보정 채택 조건(d)이 "컨텍스트를 채운 판정 0건, 생각 글이 붙은 판정 0건"을 요구해 이
# 값으로 확인한다. 판정 모델을 부르지 않은 건(빈 요약)은 None과 0으로 채운다.
def reply_fields(reply: Reply | None) -> dict:
    """판정 한 건 옆에 남기는 건별 확인값(잘림·생각 글 확인용)."""
    if reply is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "done_reason": None,
            "latency_s": 0.0,
            "thinking_chars": 0,
        }
    return {
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "done_reason": reply.done_reason,
        "latency_s": round(reply.latency_s, 3),
        "thinking_chars": len(reply.thinking or ""),
    }


# 쓸 수 있는 판정 전체에 걸친 사실 재현율(마이크로 평균: 모든 사실을 한데 모아 포함 비율을 낸다). 판정 실패
# 건은 빼고, 셀 사실이 없으면 NaN.
def fact_recall(scores: Sequence[SummaryScore]) -> float:
    """쓸 수 있는 판정의 모든 필수 사실에 걸친 마이크로 재현율."""
    total = sum(len(s.verdicts) for s in scores if s.ok)
    return sum(s.verdicts.count("포함") for s in scores if s.ok) / total if total else float("nan")


# 틀린 사실이 하나라도 있거나 대화와 어긋난 문장이 하나라도 있는 요약의 비율(요약 단위). 대화와 어긋난 문장
# 수는 보정하지 않아 참고치로만 쓴다(재보정 규칙 5).
def wrong_rate(scores: Sequence[SummaryScore]) -> float:
    """틀린 사실이나 대화와 어긋난 문장이 하나라도 있는 요약의 비율."""
    usable = [s for s in scores if s.ok]
    if not usable:
        return float("nan")
    return sum(1 for s in usable if s.wrong_statements > 0 or "틀림" in s.verdicts) / len(usable)


# 두 채점자(예: 판정 모델과 사람 채점)가 같은 건에서 사실마다 같은 판정을 낸 비율. 한쪽에만 있는 건, 판정 실패
# 건, 사실 수가 다른 건은 빼고 센다. 2단계 신뢰도 규칙(일치율 0.8 이상)에 쓰였다(judge_run의 hand_agreement).
def agreement(a: Sequence[SummaryScore], b: Sequence[SummaryScore]) -> float:
    """같은 건들에 대한 두 채점자(예: 판정 모델과 사람 채점)의 사실 단위 일치율."""
    by_b = {s.item_id: s for s in b}
    same = total = 0
    for s in a:
        o = by_b.get(s.item_id)
        if o is None or not s.ok or not o.ok or len(s.verdicts) != len(o.verdicts):
            continue
        same += sum(x == y for x, y in zip(s.verdicts, o.verdicts, strict=True))
        total += len(s.verdicts)
    return same / total if total else float("nan")


# 두 이진 채점자의 Cohen's kappa = (관측 일치 po − 우연 일치 pe) / (1 − pe). pe는 두 채점자가 각자의 '참'
# 비율대로 우연히 같은 답을 낼 확률이다.
# 왜 일치율 대신 kappa인가: 2단계에서 사실의 95.4%가 '포함'이라 모두 '포함'이라고만 해도 일치율이 95.4%가
# 나왔다. kappa는 이런 우연 일치를 빼므로 재보정의 채택 기준(kappa ≥ 0.6)으로 썼다. pe가 1이면(두 채점자가 한
# 가지 답만 냄) 정의되지 않아 NaN을 돌려준다.
# 호출 위치: judge_calib.py.
def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """같은 단위들에 대한 두 이진 채점자의 Cohen's kappa. 우연 일치가 1이면 nan."""
    if len(a) != len(b):
        raise ValueError("both raters must rate the same units")
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
