"""Score the summaries of one evaluation run with the judge model.

Reads <run>/items.jsonl (predicted records) and the dataset file named in <run>/manifest.json, writes
<run>/judge.jsonl (one SummaryScore per item) and <run>/judge_summary.json. With --hand <file>, a JSONL
of hand scores ({"item_id", "verdicts": [...], "wrong_statements": n}) is compared fact by fact and the
agreement rate is added to the summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import load_items, read_jsonl, write_jsonl
from .judge import JUDGE_VERSION, SummaryScore, agreement, fact_recall, judge_summary, wrong_rate
from .providers import OllamaProvider


def load_scores(path: Path) -> list[SummaryScore]:
    return [
        SummaryScore(
            d["item_id"], tuple(d["verdicts"]), int(d.get("wrong_statements", 0)), bool(d.get("ok", True))
        )
        for d in read_jsonl(path)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="evaluation run dir")
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--data", help="dataset JSONL (default: the run manifest's)")
    ap.add_argument("--hand", help="hand-scored JSONL to measure agreement against")
    ap.add_argument("--num-gpu", type=int, default=44, help="GPU layers for the 14B judge (VRAM headroom)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(args.data or manifest["data"])}
    rows = list(read_jsonl(run / "items.jsonl"))[: args.limit]
    judge = OllamaProvider(
        model=args.model, use_schema=True, num_ctx=4096, num_predict=256, num_gpu=args.num_gpu
    )
    out_path = run / "judge.jsonl"
    done = {d["item_id"] for d in read_jsonl(out_path)} if out_path.exists() else set()
    scores: list[SummaryScore] = load_scores(out_path) if done else []
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for n, row in enumerate(rows, 1):
            if row["item_id"] in done:
                continue
            it = items[row["item_id"]]
            summary = (row.get("pred") or {}).get("summary", "") or ""
            s = judge_summary(judge, it.item_id, it.spec, it.transcript, summary)
            scores.append(s)
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(rows):
                print(f"[{n}/{len(rows)}] recall={fact_recall(scores):.3f}", file=sys.stderr, flush=True)

    table: dict = {
        "judge_model": judge.name,
        "judge_version": JUDGE_VERSION,
        "n": len(scores),
        "n_judge_failed": sum(1 for s in scores if not s.ok),
        "fact_recall": fact_recall(scores),
        "wrong_rate": wrong_rate(scores),
    }
    if args.hand:
        hand = load_scores(Path(args.hand))
        table["hand_n"] = len(hand)
        table["hand_agreement"] = agreement(scores, hand)
        table["hand_fact_recall"] = fact_recall(hand)
    (run / "judge_summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False))
    return 0


def hand_template_main(argv: list[str] | None = None) -> int:
    """Write a hand-scoring template: item, facts, predicted summary, empty verdicts."""
    ap = argparse.ArgumentParser(description=hand_template_main.__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import random

    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(manifest["data"])}
    rows = list(read_jsonl(run / "items.jsonl"))
    random.Random(args.seed).shuffle(rows)
    out = []
    for row in rows[: args.n]:
        it = items[row["item_id"]]
        out.append(
            {
                "item_id": it.item_id,
                "facts": list(it.spec.facts),
                "summary": (row.get("pred") or {}).get("summary", "") or "",
                "verdicts": [],
                "wrong_statements": 0,
            }
        )
    write_jsonl(args.out, out)
    print(f"{len(out)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
