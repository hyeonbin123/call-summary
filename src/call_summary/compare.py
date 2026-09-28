"""Tables from evaluation runs.

compare RUN...            one row per run: headline metrics with 95% CIs, latency
compare --paired A B      metric differences B - A with paired-bootstrap CIs (same items on both sides)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .dataset import read_jsonl
from .evaluate import HEADLINE
from .scoring import METRICS, ItemScore, paired_bootstrap_diff


def load_run(run: str | Path) -> tuple[dict, list[ItemScore]]:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    scores = [ItemScore(**d["score"]) for d in read_jsonl(run / "items.jsonl")]
    return manifest, scores


def run_name(manifest: dict) -> str:
    shots = manifest.get("shots") or 0
    prompt = manifest.get("prompt_version", "p1")
    return (
        manifest["provider"] + (f" {shots}-shot" if shots else "") + (f" {prompt}" if prompt != "p1" else "")
    )


def fmt(x: float, pct: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


def run_table(runs: list[str], metrics: tuple[str, ...] = HEADLINE) -> str:
    head = "| run | n | " + " | ".join(metrics) + " | p50 s |"
    lines = [head, "|" + "---|" * (len(metrics) + 3)]
    for r in runs:
        manifest, _ = load_run(r)
        summary = json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))
        cells = []
        for m in metrics:
            p = summary["point"][m]
            lo, hi = summary["ci95"].get(m, [float("nan"), float("nan")])
            cells.append(f"{fmt(p)} [{fmt(lo)}, {fmt(hi)}]")
        lat = summary.get("latency_s", {}).get("p50")
        lines.append(
            f"| {run_name(manifest)} | {summary['n']} | " + " | ".join(cells) + f" | {fmt(lat, False)} |"
        )
    return "\n".join(lines)


def paired_table(a: str, b: str, metrics: tuple[str, ...] = HEADLINE, n_boot: int = 2000) -> str:
    ma, sa = load_run(a)
    mb, sb = load_run(b)
    lines = [
        f"B − A: `{run_name(mb)}` − `{run_name(ma)}` (n={len(sa)})",
        "",
        "| metric | A | B | diff [95% CI] |",
        "|---|---|---|---|",
    ]
    for m in metrics:
        d, lo, hi = paired_bootstrap_diff(sa, sb, METRICS[m], n_boot=n_boot)
        lines.append(
            f"| {m} | {fmt(METRICS[m](sa))} | {fmt(METRICS[m](sb))} | {fmt(d)} [{fmt(lo)}, {fmt(hi)}] |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"))
    ap.add_argument("--metrics", nargs="+", default=list(HEADLINE))
    args = ap.parse_args(argv)
    metrics = tuple(args.metrics)
    if args.paired:
        print(paired_table(args.paired[0], args.paired[1], metrics))
    if args.runs:
        print(run_table(args.runs, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
