"""Score the summaries of one evaluation run with the judge model.

Reads <run>/items.jsonl (predicted records) and the dataset file named in <run>/manifest.json, writes
<run>/judge.jsonl (one SummaryScore per item, plus the reply's token counts) and <run>/judge_summary.json.
--out names another output file (its summary goes next to it as <stem>_summary.json), so a second judge
model never overwrites the recorded j1 judgements. With --hand <file>, a JSONL of hand scores
({"item_id", "verdicts": [...], "wrong_statements": n}) is compared fact by fact and the agreement rate is
added to the summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dataset import load_items, read_jsonl, write_jsonl
from .evaluate import THINK, model_fingerprint, parse_option, run_checks
from .judge import (
    JUDGE_NUM_CTX,
    JUDGE_VERSION,
    SummaryScore,
    agreement,
    fact_recall,
    judge_reply,
    make_judge,
    reply_fields,
    wrong_rate,
)


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
    ap.add_argument(
        "--think", choices=sorted(THINK), default="off", help="thinking switch sent to the judge (j1: off)"
    )
    ap.add_argument(
        "--option",
        type=parse_option,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Ollama option, repeatable (j1 sent none)",
    )
    ap.add_argument("--out", help="output JSONL (default <run>/judge.jsonl)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(args.data or manifest["data"])}
    rows = list(read_jsonl(run / "items.jsonl"))[: args.limit]
    options = dict(args.option)
    judge = make_judge(model=args.model, num_gpu=args.num_gpu, think=THINK[args.think], options=options)
    out_path = Path(args.out) if args.out else run / "judge.jsonl"
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {d["item_id"] for d in read_jsonl(out_path)} if out_path.exists() else set()
    scores: list[SummaryScore] = load_scores(out_path) if done else []
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for n, row in enumerate(rows, 1):
            if row["item_id"] in done:
                continue
            it = items[row["item_id"]]
            summary = (row.get("pred") or {}).get("summary", "") or ""
            s, reply = judge_reply(judge, it.item_id, it.spec, it.transcript, summary)
            scores.append(s)
            f.write(json.dumps(s.to_dict() | reply_fields(reply), ensure_ascii=False) + "\n")
            f.flush()
            if n % 20 == 0 or n == len(rows):
                print(f"[{n}/{len(rows)}] recall={fact_recall(scores):.3f}", file=sys.stderr, flush=True)

    table: dict = {
        "judge_model": judge.name,
        "judge_version": JUDGE_VERSION,
        "num_gpu": args.num_gpu,
        "think": args.think,
        "options": options,
        "fingerprint": model_fingerprint(judge),
        "run_checks": run_checks(list(read_jsonl(out_path)), JUDGE_NUM_CTX),
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
    summary_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
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
