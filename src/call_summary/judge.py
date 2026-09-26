"""Summary scoring with a judge model: each required fact of the spec is marked 포함 / 누락 / 틀림, and
statements that contradict the call are counted.

The judge's own reliability is measured against hand scores on dev before its numbers are used
(docs/experiments.md, stage 2).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from .providers import Provider
from .specs import Spec

JUDGE_VERSION = "j1"
VERDICTS = ("포함", "누락", "틀림")

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


@dataclass(frozen=True)
class SummaryScore:
    item_id: str
    verdicts: tuple[str, ...]  # one per required fact; empty when the judge failed
    wrong_statements: int
    ok: bool  # the judge reply was usable

    @property
    def recall(self) -> float:
        return self.verdicts.count("포함") / len(self.verdicts) if self.verdicts else float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def judge_prompt(spec: Spec, transcript: str, summary: str) -> str:
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(spec.facts))
    return _JUDGE.format(
        transcript=transcript, facts=facts, summary=summary.strip() or "(빈 요약)", n=len(spec.facts)
    )


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


def judge_summary(judge: Provider, item_id: str, spec: Spec, transcript: str, summary: str) -> SummaryScore:
    n = len(spec.facts)
    if not summary.strip():
        return SummaryScore(item_id, ("누락",) * n, 0, True)
    reply = judge.generate(
        [{"role": "user", "content": judge_prompt(spec, transcript, summary)}], json_schema=judge_schema(n)
    )
    parsed = parse_judge(reply.text, n)
    if parsed is None:
        return SummaryScore(item_id, (), 0, False)
    return SummaryScore(item_id, parsed[0], parsed[1], True)


def fact_recall(scores: Sequence[SummaryScore]) -> float:
    """Micro recall over all required facts of the usable judgements."""
    total = sum(len(s.verdicts) for s in scores if s.ok)
    return sum(s.verdicts.count("포함") for s in scores if s.ok) / total if total else float("nan")


def wrong_rate(scores: Sequence[SummaryScore]) -> float:
    """Share of summaries with at least one wrong fact or contradicting statement."""
    usable = [s for s in scores if s.ok]
    if not usable:
        return float("nan")
    return sum(1 for s in usable if s.wrong_statements > 0 or "틀림" in s.verdicts) / len(usable)


def agreement(a: Sequence[SummaryScore], b: Sequence[SummaryScore]) -> float:
    """Fact-level agreement between two raters over the same items (e.g. judge vs hand scores)."""
    by_b = {s.item_id: s for s in b}
    same = total = 0
    for s in a:
        o = by_b.get(s.item_id)
        if o is None or not s.ok or not o.ok or len(s.verdicts) != len(o.verdicts):
            continue
        same += sum(x == y for x, y in zip(s.verdicts, o.verdicts, strict=True))
        total += len(s.verdicts)
    return same / total if total else float("nan")
