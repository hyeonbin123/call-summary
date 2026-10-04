"""Identifiers that need confirmation before anyone acts on them.

An identifier in a record (order, tracking or receipt number, card last digits) is flagged when its
normalized form (values.normalize) does not have the deployment format (domains.EntityType.id_format), or
when it cannot be found in the transcript (values.occurs_in_transcript with spoken=True, so digits the
recogniser split with commas or spaces still count). The record itself is never changed.

A flag is not a correction and not an accuracy gain: a wrong number that has the right format and is in the
transcript (the recogniser's error copied, a same-length digit confusion) passes both checks.

python -m call_summary.verify RUN...   applies the checks to recorded evaluation runs (reports/<run_id>)
                                       and counts flagged correct and wrong identifiers (no model needed)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .dataset import load_items, read_jsonl
from .domains import DOMAINS, EntityType
from .schema import AfterCallRecord
from .values import normalize, occurs_in_transcript

Reason = Literal["format", "not_in_transcript"]


class IdentifierFlag(BaseModel):
    type: str
    value: str
    reasons: list[Reason]


def check_identifier(et: EntityType, value: str, transcript: str) -> list[Reason]:
    """Why this identifier needs confirmation (empty: it passes both checks)."""
    reasons: list[Reason] = []
    norm = normalize("id", value)
    if et.id_format is not None and (norm is None or re.fullmatch(et.id_format, norm) is None):
        reasons.append("format")
    if not occurs_in_transcript("id", value, transcript, spoken=True):
        reasons.append("not_in_transcript")
    return reasons


def _id_types(domain_key: str) -> dict[str, EntityType]:
    return {et.label: et for et in DOMAINS[domain_key].entity_types if et.kind == "id"}


def verify_entities(domain_key: str, record: AfterCallRecord, transcript: str) -> list[IdentifierFlag]:
    """Flags for the record's identifiers, in record order. Other entity kinds are not checked."""
    types = _id_types(domain_key)
    flags = []
    for e in record.entities:
        et = types.get(e.type)
        if et is None:
            continue
        reasons = check_identifier(et, e.value, transcript)
        if reasons:
            flags.append(IdentifierFlag(type=e.type, value=e.value, reasons=reasons))
    return flags


# ---------------------------------------------------------------------------------------------------------
# Applying the checks to recorded runs (docs/experiments.md "확인이 필요한 번호 표시").

# The tracking format as the brief wrote it, on the value as written. Report only: it shows which correct
# numbers a format rule on the raw string would flag (written without hyphens, or with spaces).
WRITTEN_FORMATS = {"운송장번호": r"\d{4}-\d{4}-\d{4}"}


def _written_form_ok(et: EntityType, value: str) -> bool:
    pattern = WRITTEN_FORMATS.get(et.label, et.id_format)
    return pattern is None or re.fullmatch(pattern, value.strip()) is not None


def _shape(norm: str | None, golds: list[str]) -> tuple[str, str]:
    """How a wrong identifier differs from the item's gold values of its type, and for a one-character
    difference which character became which ("1>2": the gold 1 came out as 2)."""
    if not golds:
        return "no_gold_of_type", ""
    if norm is None:
        return "unreadable", ""
    same_len = [g for g in golds if len(g) == len(norm)]
    if not same_len:
        return "other_length", ""
    best = min(same_len, key=lambda g: sum(a != b for a, b in zip(norm, g, strict=True)))
    diffs = [(g, p) for p, g in zip(norm, best, strict=True) if p != g]
    return ("one_char", f"{diffs[0][0]}>{diffs[0][1]}") if len(diffs) == 1 else ("same_length", "")


def _empty_counts() -> dict:
    return {"predicted": 0, "correct": 0, "wrong": 0, "flagged_correct": 0, "flagged_wrong": 0}


def guard_report(run: str | Path, data: str | Path | None = None) -> dict:
    """Flag counts of one evaluation run against its dataset (manifest `data` unless given)."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(data or manifest["data"].replace("\\", "/"))}
    total = _empty_counts()
    by_type: dict[str, dict] = {}
    reasons_wrong: Counter = Counter()
    shapes_silent: Counter = Counter()
    changes_silent: Counter = Counter()
    written_rule_extra = {"correct": 0, "wrong": 0}
    missed = no_record = 0
    silent: list[list] = []
    false_flags: list[list] = []
    for row in read_jsonl(run / "items.jsonl"):
        if not row.get("pred"):
            no_record += 1  # the service answers 502: no identifier goes out
            continue
        it = items[row["item_id"]]
        types = _id_types(it.domain)
        pred = AfterCallRecord.model_validate(row["pred"])
        gold_ids = [(e.type, normalize("id", e.value)) for e in it.gold().entities if e.type in types]
        gold_left: Counter = Counter(gold_ids)
        transcript = it.transcript
        for e in pred.entities:
            et = types.get(e.type)
            if et is None:
                continue
            norm = normalize("id", e.value)
            ok = gold_left[(e.type, norm)] > 0
            if ok:
                gold_left[(e.type, norm)] -= 1
            reasons = check_identifier(et, e.value, transcript)
            flagged = bool(reasons)
            cell = by_type.setdefault(e.type, _empty_counts())
            for c in (total, cell):
                c["predicted"] += 1
                c["correct" if ok else "wrong"] += 1
                if flagged:
                    c["flagged_correct" if ok else "flagged_wrong"] += 1
            if not ok and flagged:
                reasons_wrong["+".join(reasons)] += 1
            if "format" not in reasons and not _written_form_ok(et, e.value):
                written_rule_extra["correct" if ok else "wrong"] += 1
            if ok and flagged:
                false_flags.append([it.item_id, e.type, e.value, reasons])
            if not ok and not flagged:
                golds = sorted({g for t, g in gold_ids if t == e.type and g is not None})
                shape, change = _shape(norm, golds)
                shapes_silent[shape] += 1
                if change:
                    changes_silent[change] += 1
                silent.append([it.item_id, e.type, e.value, golds, shape, change])
        missed += sum(gold_left.values())
    return {
        **total,
        "by_type": dict(sorted(by_type.items())),
        "flagged_wrong_reasons": dict(reasons_wrong.most_common()),
        "silent_wrong_shapes": dict(shapes_silent.most_common()),
        "silent_wrong_one_char_changes": dict(changes_silent.most_common()),
        "written_format_rule_would_also_flag": written_rule_extra,
        "gold_ids_not_predicted": missed,
        "rows_without_record": no_record,
        "silent_wrong": silent,
        "false_flags": false_flags,
    }


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}" if b else "-"


def guard_table(got: dict[str, dict]) -> str:
    lines = [
        "| 실행 | 예측한 번호 | 맞음 | 틀림 | 표시된 맞은 번호 (오검출률) | 표시된 틀린 번호 "
        "| 조용히 나가는 틀린 번호: 표시 전 → 후 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, g in got.items():
        n, ok, bad = g["predicted"], g["correct"], g["wrong"]
        fc, fw = g["flagged_correct"], g["flagged_wrong"]
        lines.append(
            f"| `{name}` | {n} | {ok} | {bad} | {fc} ({_pct(fc, ok)}%) | {fw}/{bad} ({_pct(fw, bad)}%) "
            f"| {bad}/{n} ({_pct(bad, n)}%) → {bad - fw}/{n} ({_pct(bad - fw, n)}%) |"
        )
    return "\n".join(lines)


def guard_detail(got: dict[str, dict]) -> str:
    lines = []
    for name, g in got.items():
        lines.append(f"\n{name}")
        for t, c in g["by_type"].items():
            lines.append(
                f"  {t}: predicted {c['predicted']}, correct {c['correct']} "
                f"(flagged {c['flagged_correct']}), wrong {c['wrong']} (flagged {c['flagged_wrong']})"
            )
        lines.append(f"  flagged wrong by reason: {g['flagged_wrong_reasons']}")
        lines.append(f"  silent wrong by shape: {g['silent_wrong_shapes']}")
        lines.append(f"  silent one-character changes (gold>predicted): {g['silent_wrong_one_char_changes']}")
        lines.append(f"  written-format rule would also flag: {g['written_format_rule_would_also_flag']}")
        lines.append(
            f"  gold ids not predicted: {g['gold_ids_not_predicted']}, rows without record: "
            f"{g['rows_without_record']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Apply the identifier checks to recorded evaluation runs.")
    ap.add_argument("runs", nargs="+", help="reports/<run_id> directories")
    ap.add_argument("--show", action="store_true", help="also list silent wrong ids and false flags")
    ap.add_argument("--json", help="also write the numbers here")
    args = ap.parse_args(argv)
    got = {str(r): guard_report(r) for r in args.runs}
    print(guard_table(got))
    print(guard_detail(got))
    if args.show:
        for name, g in got.items():
            print(f"\n{name}")
            for item_id, t, value, golds, shape, change in g["silent_wrong"]:
                print(f"  silent {item_id} {t} {value!r} gold {golds} ({shape} {change})")
            for item_id, t, value, reasons in g["false_flags"]:
                print(f"  false flag {item_id} {t} {value!r} {reasons}")
    if args.json:
        Path(args.json).write_text(json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
