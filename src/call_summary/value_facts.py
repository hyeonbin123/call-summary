"""Deterministic value check of summaries (docs/experiments.md "요약 판정 재보정").

Each required fact of a spec is formatted from a scenario or outcome template over slot names. The numeric
values in a fact (번호, 금액, 날짜, 시각: kinds id, amount, date, time) are looked up in the summary text
with the scoring rules (`values.occurs_in_transcript`, written text). Names (kind text) are left to the
judge: a summary may shorten or paraphrase them.

  python -m call_summary.value_facts reports/<run>... [--judge-file judge-<name>.jsonl] [--json out.json]

With --judge-file, the run's judge verdicts are also combined with the check: a fact whose numeric value
is missing from the summary counts as 누락 unless the judge already said 틀림 (`hybrid_verdicts`).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

from .dataset import load_items, read_jsonl
from .domains import DOMAINS
from .judge import VERDICTS
from .specs import Spec
from .values import Kind, normalize, occurs_in_transcript

NUMERIC_KINDS: tuple[Kind, ...] = ("id", "amount", "date", "time")
INCLUDED = VERDICTS[0]  # 포함
MISSING = VERDICTS[1]  # 누락
WRONG = VERDICTS[2]  # 틀림


@dataclass(frozen=True)
class ValueMention:
    fact: int  # index into spec.facts
    slot: str
    type: str  # entity label
    kind: Kind
    value: str  # gold surface form


def fact_slots(spec: Spec) -> list[tuple[str, ...]]:
    """Slot names each fact was formatted from, in template order. Raises when the spec's facts are not
    the templates filled with its slots (an edited or foreign spec)."""
    domain = DOMAINS[spec.domain]
    scenario = next(s for s in domain.scenarios if s.category == spec.category)
    outcome = scenario.outcomes[spec.outcome_index]
    templates = (scenario.request_fact, *outcome.facts)
    by_slot = {sv.slot: sv.value for sv in spec.slots}
    if tuple(t.format(**by_slot) for t in templates) != spec.facts:
        raise ValueError(f"{spec.spec_id}: the facts are not the templates filled with the slots")
    return [tuple(dict.fromkeys(name for _, name, _, _ in Formatter().parse(t) if name)) for t in templates]


def value_mentions(spec: Spec) -> list[ValueMention]:
    """The numeric gold values written in the facts, one per (fact, slot)."""
    domain = DOMAINS[spec.domain]
    by_slot = {sv.slot: sv for sv in spec.slots}
    out = []
    for i, names in enumerate(fact_slots(spec)):
        for name in names:
            sv = by_slot[name]
            kind = domain.entity_type(sv.type).kind
            if kind in NUMERIC_KINDS:
                out.append(ValueMention(i, name, sv.type, kind, sv.value))
    return out


def mention_found(spec: Spec, summary: str) -> list[tuple[ValueMention, bool]]:
    return [(m, occurs_in_transcript(m.kind, m.value, summary)) for m in value_mentions(spec)]


def fact_value_ok(spec: Spec, summary: str) -> list[bool | None]:
    """Per fact: None without a numeric value, True when all its numeric values are in the summary."""
    out: list[bool | None] = [None] * len(spec.facts)
    for m, found in mention_found(spec, summary):
        out[m.fact] = found and out[m.fact] is not False
    return out


def hybrid_verdicts(verdicts: Sequence[str], value_ok: Sequence[bool | None]) -> tuple[str, ...]:
    """Judge verdicts with the value check applied: a value miss makes 포함 into 누락 (틀림 stays).
    Empty verdicts (a failed judgement) start from 포함, so only the value check speaks."""
    base = tuple(verdicts) if verdicts else (INCLUDED,) * len(value_ok)
    if len(base) != len(value_ok):
        raise ValueError("one verdict per fact")
    return tuple(
        MISSING if ok is False and v == INCLUDED else v for v, ok in zip(base, value_ok, strict=True)
    )


def distractor_written(spec: Spec, summary: str) -> bool:
    """The caller's first, wrong value (EV_CORRECTION) appears in the summary."""
    d = spec.distractor
    if d is None:
        return False
    kind = DOMAINS[spec.domain].entity_type(d.type).kind
    return normalize(kind, d.value) is not None and occurs_in_transcript(kind, d.value, summary)


def _boot(
    rows: Sequence, stat: Callable[[Sequence], float], n_boot: int, seed: int = 0
) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    rng = random.Random(seed)
    k = len(rows)
    vals = []
    for _ in range(n_boot):
        v = stat([rows[rng.randrange(k)] for _ in range(k)])
        if v == v:
            vals.append(v)
    if not vals:
        return math.nan, math.nan
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def _ratio(rows: Sequence, num: Callable, den: Callable) -> float:
    d = sum(den(r) for r in rows)
    return sum(num(r) for r in rows) / d if d else math.nan


def run_value_table(run: str | Path, judge_file: str | None = None, n_boot: int = 2000) -> dict:
    """Value-check numbers of one recorded run (and, with judge_file, its raw and hybrid fact recall)."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(manifest["data"].replace("\\", "/"))}
    rows = []
    for r in read_jsonl(run / "items.jsonl"):
        it = items[r["item_id"]]
        summary = (r.get("pred") or {}).get("summary", "") or ""
        found = mention_found(it.spec, summary)
        ok = fact_value_ok(it.spec, summary)
        rows.append(
            {
                "item_id": it.item_id,
                "empty": not summary.strip(),
                "found": found,
                "value_ok": ok,
                "distractor": it.spec.distractor is not None,
                "distractor_written": distractor_written(it.spec, summary),
            }
        )

    def n_found(r):
        return sum(f for _, f in r["found"])

    def n_ment(r):
        return len(r["found"])

    def vf_ok(r):
        return sum(1 for x in r["value_ok"] if x is True)

    def vf_all(r):
        return sum(1 for x in r["value_ok"] if x is not None)

    by_kind: dict[str, list[int]] = {}
    for r in rows:
        for m, f in r["found"]:
            by_kind.setdefault(m.kind, [0, 0])
            by_kind[m.kind][0] += int(f)
            by_kind[m.kind][1] += 1
    with_values = [r for r in rows if r["found"]]
    table: dict = {
        "run": str(run),
        "n_items": len(rows),
        "n_empty": sum(r["empty"] for r in rows),
        "n_mentions": sum(n_ment(r) for r in rows),
        "value_recall": _ratio(rows, n_found, n_ment),
        "value_recall_ci": _boot(rows, lambda xs: _ratio(xs, n_found, n_ment), n_boot),
        "value_fact_recall": _ratio(rows, vf_ok, vf_all),
        "value_fact_recall_ci": _boot(rows, lambda xs: _ratio(xs, vf_ok, vf_all), n_boot),
        "items_all_values": (
            sum(1 for r in with_values if all(f for _, f in r["found"])) / len(with_values)
            if with_values
            else math.nan
        ),
        "by_kind": {k: by_kind[k] for k in NUMERIC_KINDS if k in by_kind},
        "distractor_items": sum(r["distractor"] for r in rows),
        "distractor_in_summary": sum(r["distractor_written"] for r in rows),
    }
    if judge_file:
        table["judge"] = _judge_numbers(run / judge_file, rows, n_boot)
    return table


def _judge_numbers(path: Path, rows: list[dict], n_boot: int) -> dict:
    judged = {d["item_id"]: d for d in read_jsonl(path)}
    missing = [r["item_id"] for r in rows if r["item_id"] not in judged]
    if missing:
        raise ValueError(f"{path} has no judgement for {len(missing)} items, e.g. {missing[:3]}")
    usable = []
    for r in rows:
        d = judged[r["item_id"]]
        if not d.get("ok", True):
            continue
        raw = tuple(d["verdicts"])
        usable.append(
            {
                "raw": raw,
                "hybrid": hybrid_verdicts(raw, r["value_ok"]),
                "wrong": int(d.get("wrong_statements", 0)) > 0 or WRONG in raw,
            }
        )

    def inc(key):
        return lambda x: x[key].count(INCLUDED)

    def total(x):
        return len(x["raw"])

    return {
        "file": path.name,
        "n_judge_failed": len(rows) - len(usable),
        "raw_fact_recall": _ratio(usable, inc("raw"), total),
        "raw_fact_recall_ci": _boot(usable, lambda xs: _ratio(xs, inc("raw"), total), n_boot),
        "hybrid_fact_recall": _ratio(usable, inc("hybrid"), total),
        "hybrid_fact_recall_ci": _boot(usable, lambda xs: _ratio(xs, inc("hybrid"), total), n_boot),
        "wrong_rate": sum(x["wrong"] for x in usable) / len(usable) if usable else math.nan,
    }


def _pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.1f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="recorded run dirs (reports/<run_id>)")
    ap.add_argument("--judge-file", help="judge output inside each run dir, e.g. judge-<name>.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", help="write all numbers to this file")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    tables = [run_value_table(r, args.judge_file, args.n_boot) for r in args.runs]
    print("run | n | value_recall [95%] | value_fact_recall | items_all_values | by kind | distractor")
    for t in tables:
        lo, hi = t["value_recall_ci"]
        kinds = ", ".join(f"{k} {f}/{n}" for k, (f, n) in t["by_kind"].items())
        print(
            f"{Path(t['run']).name} | {t['n_items']} | {_pct(t['value_recall'])} [{_pct(lo)}, {_pct(hi)}] | "
            f"{_pct(t['value_fact_recall'])} | {_pct(t['items_all_values'])} | {kinds} | "
            f"{t['distractor_in_summary']}/{t['distractor_items']}"
        )
        if "judge" in t:
            j = t["judge"]
            print(
                f"  judge {j['file']}: raw fact recall {_pct(j['raw_fact_recall'])}, "
                f"with value check {_pct(j['hybrid_fact_recall'])}, wrong_rate {_pct(j['wrong_rate'])}, "
                f"failed {j['n_judge_failed']}"
            )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
