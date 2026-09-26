"""Turn raw generated files into the fixed dataset splits.

Per domain, keep the first N accepted items in spec-index order (docs/experiments.md, stage 1), drop test
items whose numbers also occur in train, and report how many specs were rejected, per category.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .dataset import Item, load_items, read_jsonl, write_jsonl
from .domains import DOMAINS
from .gen import check_dialogue
from .specs import Spec
from .values import normalize

TARGETS = {"train": 500, "dev": 80, "test-a": 100, "test-c": 200, "test-b": 20}


def _index(item_id: str) -> int:
    return int(item_id.rsplit("-", 1)[1])


def take_first(items: list[Item], per_domain: int) -> list[Item]:
    by_dom: dict[str, list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda it: (it.domain, _index(it.item_id))):
        by_dom[it.domain].append(it)
    out: list[Item] = []
    for dom in sorted(by_dom):
        out.extend(by_dom[dom][:per_domain])
    return out


def id_values(item: Item) -> set[str]:
    """Normalized id-kind values of an item (order, tracking, receipt numbers, card digits)."""
    d = DOMAINS[item.domain]
    out = set()
    for sv in item.spec.slots:
        if d.entity_type(sv.type).kind == "id":
            n = normalize("id", sv.value)
            if n:
                out.add(n)
    return out


def rejection_report(raw_dir: Path, split: str) -> dict:
    rej_path = raw_dir / f"{split}.rejects.jsonl"
    specs = {d["spec_id"]: Spec.from_dict(d) for d in read_jsonl(Path("data/specs") / f"{split}.jsonl")}
    rejected = list(read_jsonl(rej_path)) if rej_path.exists() else []
    ok = load_items(raw_dir / f"{split}.jsonl")
    by_cat: Counter = Counter()
    reasons: Counter = Counter()
    for r in rejected:
        s = specs.get(r["spec_id"])
        if s:
            by_cat[f"{s.domain}/{s.category}"] += 1
        for p in r["problems"]:
            reasons[re.sub(r"[\[\(].*", "", p).strip().split(" '")[0]] += 1
    total = len(ok) + len(rejected)
    return {
        "attempted": total,
        "accepted": len(ok),
        "rejected": len(rejected),
        "reject_rate": round(len(rejected) / total, 4) if total else None,
        "rejected_by_category": dict(by_cat.most_common()),
        "reasons": dict(reasons.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data")
    ap.add_argument("--splits", nargs="+", default=list(TARGETS))
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    report: dict = {}
    train_ids: set[str] = set()
    train_path = raw / "train.jsonl"
    if train_path.exists():
        for it in load_items(train_path):
            train_ids |= id_values(it)
    for split in args.splits:
        items = load_items(raw / f"{split}.jsonl")
        # The checks may have grown since a split was generated; apply the current ones to every split.
        failed = {it.item_id: check_dialogue(it.spec, it.turns) for it in items}
        recheck = {k: v for k, v in failed.items() if v}
        items = [it for it in items if it.item_id not in recheck]
        overlap = []
        if split != "train":
            overlap = [it.item_id for it in items if id_values(it) & train_ids]
            items = [it for it in items if it.item_id not in set(overlap)]
        chosen = take_first(items, TARGETS[split])
        short = {
            dom: TARGETS[split] - n
            for dom, n in Counter(it.domain for it in chosen).items()
            if n < TARGETS[split]
        }
        n = write_jsonl(Path(args.out) / f"{split}.jsonl", (it.to_dict() for it in chosen))
        rep = rejection_report(raw, split) if split != "test-b" else {}
        rep.update(
            {
                "written": n,
                "dropped_recheck": recheck,
                "dropped_train_overlap": overlap,
                "short_by_domain": short,
            }
        )
        report[split] = rep
        print(f"{split}: {n} items" + (f", short {short}" if short else ""))
    Path(args.out, "finalize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
