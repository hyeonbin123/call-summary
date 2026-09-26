"""Dialogue and reference-summary generation from specs, with code checks that throw bad dialogues away.

The writer model only turns a spec into words. Every value the call must contain is in the spec, and
`check_dialogue` rejects a dialogue that drops one of them or invents another number, date or time.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

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

WRITER_VERSION = "w1"
SPEAKERS = ("상담원", "고객")

_EVENT_TEXT = {
    EV_VERIFY: "상담원이 처리 전에 본인 확인을 한다. 고객 이름과 생년월일 여섯 자리를 묻고, 고객이 답한다.",
    EV_COMPLAINT: "고객이 짜증이나 불만을 드러내고, 상담원이 사과하며 달랜다.",
    EV_HOLD: "상담원이 확인하는 동안 잠시 기다려 달라고 하고, 조금 뒤 돌아와 말을 잇는다.",
    EV_REPEAT: "고객이 이미 들은 내용을 한 번 더 되묻고, 상담원이 다시 설명한다.",
    EV_SMALLTALK: "고객이 본론 전에 날씨나 연결이 오래 걸렸다는 등 짧은 잡담을 한다.",
}

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

_SUMMARIZER = """다음은 {company}({label}) 고객센터의 상담 대화입니다. 상담 기록에 들어갈 요약을 3~5문장의 한국어로 써 주세요.

- 고객의 요청, 상담원이 확인하거나 처리한 것, 결과와 후속 조치를 담는다.
- 대화에 나온 핵심 값(번호, 금액, 날짜, 시각, 상품·요금제 이름)은 대화에 나온 그대로 쓴다.
- 대화에 없는 내용은 쓰지 않는다. 인사말, 본인 확인 절차의 세부 값(이름, 생년월일)은 쓰지 않는다.
- "고객은 ~함", "상담원은 ~함"처럼 기록체로 쓴다.

[상담 대화]
{transcript}

JSON으로만 답하세요: {{"summary": "..."}}"""

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


def summarizer_prompt(spec: Spec, transcript: str) -> str:
    d = DOMAINS[spec.domain]
    return _SUMMARIZER.format(company=d.company, label=d.label, transcript=transcript)


# The writer model sometimes switches to Chinese mid-dialogue.
_FOREIGN_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_ID_LIKE = re.compile(r"(?<![\dA-Za-z\-])(?:[A-Z]\d{5,}|\d{2,5}(?:-\d{2,5}){1,3})(?![\dA-Za-z\-])")


def check_dialogue(spec: Spec, turns: Sequence[dict]) -> list[str]:
    """Problems that make a generated dialogue unusable. Empty list means it passes."""
    problems: list[str] = []
    if not 8 <= len(turns) <= 40:
        problems.append(f"turn count {len(turns)}")
    if not turns:
        return problems
    if any(t.get("speaker") not in SPEAKERS or not str(t.get("text", "")).strip() for t in turns):
        problems.append("bad speaker or empty turn")
        return problems
    if turns[0]["speaker"] != "상담원":
        problems.append("does not start with the agent")
    if {t["speaker"] for t in turns} != set(SPEAKERS):
        problems.append("one speaker only")
    same_in_row = sum(1 for a, b in zip(turns, turns[1:], strict=False) if a["speaker"] == b["speaker"])
    if same_in_row > 2:
        problems.append(f"{same_in_row} consecutive same-speaker turns")

    text = "\n".join(t["text"] for t in turns)
    if _FOREIGN_SCRIPT.search(text):
        problems.append("non-Korean script (Han/kana)")
    d = DOMAINS[spec.domain]
    for sv in spec.slots:
        if sv.value not in text:
            problems.append(f"missing {sv.type} {sv.value!r}")
    if spec.distractor is not None and spec.distractor.value not in text:
        problems.append(f"missing distractor {spec.distractor.value!r}")

    # No numbers, dates or times beyond the spec's.
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
    allowed_ids = {normalize("id", v) for v in (sv.value for sv in spec.slots)} | (
        {normalize("id", spec.distractor.value)} if spec.distractor else set()
    )
    for m in _ID_LIKE.finditer(text):
        # A shortened mention of an allowed number ("5891-6890 끝나는 거요") is not a new number.
        norm = normalize("id", m.group(0)) or ""
        if not any(norm in a for a in allowed_ids if a):
            problems.append(f"extra id {m.group(0)!r}")
    return problems


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


def _reseed(provider: Provider, seed: int | None, attempt: int) -> None:
    """A different sampling seed per try, so a retry is not a replay of the rejected output."""
    if seed is not None and hasattr(provider, "seed"):
        provider.seed = seed + attempt  # type: ignore[attr-defined]


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
        except Exception as exc:  # noqa: BLE001 - a hung or failed request is one failed try
            problems = [f"request failed: {type(exc).__name__}"]
            stats["failed_requests"] = stats.get("failed_requests", 0) + 1
            continue
        stats["seconds"] += reply.latency_s
        got = _parse_json(reply.text, "turns")
        if not isinstance(got, list):
            problems = ["unparseable dialogue"]
            continue
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--specs", required=True, help="spec JSONL (from build_specs)")
    ap.add_argument("--out", required=True, help="item JSONL; appended to, already-done spec ids are skipped")
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--limit", type=int)
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

    common = {"model": args.model, "use_schema": True, "num_ctx": args.num_ctx, "num_gpu": args.num_gpu}
    writer = OllamaProvider(temperature=args.temperature, num_predict=2048, timeout_s=args.timeout, **common)
    summarizer = OllamaProvider(temperature=0.3, num_predict=512, timeout_s=args.timeout, **common)
    counts: Counter = Counter()
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
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
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"[{n}/{len(todo)}] {dict(counts)} {rate:.1f}s/item", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
