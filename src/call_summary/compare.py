"""Tables from evaluation runs.

compare RUN...            one row per run: headline metrics with 95% CIs, latency
compare --paired A B      metric differences B - A with paired-bootstrap CIs (same items on both sides)
compare --pick RUN...     runs ranked by the pick rule: exact, then entity F1, then lower hallucination
compare --think-check RUN...  per run: thinking text, <think> tags, replies cut at num_predict, answer tokens
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


def _pick_key(point: dict) -> tuple[float, float, float]:
    def num(x: float | None, worst: float) -> float:
        return worst if x is None or (isinstance(x, float) and math.isnan(x)) else x

    # undefined values (no entities predicted) rank last
    return (
        -num(point.get("exact"), -math.inf),
        -num(point.get("entity_f1"), -math.inf),
        num(point.get("hallucination"), math.inf),
    )


def pick_order(runs: list[str]) -> list[str]:
    """Runs best first by dev exact, then entity F1, then lower hallucination; full ties keep the given
    order (list the runs in their registered order)."""
    points = {r: json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))["point"] for r in runs}
    return sorted(runs, key=lambda r: _pick_key(points[r]))


def pick_table(runs: list[str]) -> str:
    lines = ["| rank | run | exact | entity_f1 | hallucination |", "|---|---|---|---|---|"]
    for i, r in enumerate(pick_order(runs), 1):
        manifest = json.loads((Path(r) / "manifest.json").read_text(encoding="utf-8"))
        point = json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))["point"]
        cells = " | ".join(fmt(point.get(m)) for m in ("exact", "entity_f1", "hallucination"))
        lines.append(f"| {i} | {run_name(manifest)} (`{Path(r).name}`) | {cells} |")
    return "\n".join(lines)


def think_rows(runs: list[str]) -> list[dict]:
    """Per run, what shows whether the thinking switch held: replies that came with thinking text
    (Ollama's message.thinking), replies with <think> tags in the answer, replies cut at num_predict,
    and the answer length in tokens (p50 as in evaluate, max)."""
    out = []
    for r in runs:
        manifest = json.loads((Path(r) / "manifest.json").read_text(encoding="utf-8"))
        rows = list(read_jsonl(Path(r) / "items.jsonl"))
        toks = sorted(x["completion_tokens"] for x in rows if x.get("completion_tokens") is not None)
        out.append(
            {
                "run": Path(r).name,
                "think": manifest.get("think"),
                "ollama_options": manifest.get("ollama_options"),
                "n": len(rows),
                "thinking_items": sum(1 for x in rows if x.get("thinking_chars")),
                "think_tags": sum(1 for x in rows if "<think>" in x["reply"] or "</think>" in x["reply"]),
                "length_stops": sum(1 for x in rows if x.get("done_reason") == "length"),
                "schema_ok": sum(1 for x in rows if x["score"]["schema_ok"]),
                "completion_p50": toks[len(toks) // 2] if toks else None,
                "completion_max": toks[-1] if toks else None,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"))
    ap.add_argument("--metrics", nargs="+", default=list(HEADLINE))
    ap.add_argument("--pick", nargs="+", metavar="RUN", help="rank runs by the pick rule (first = pick)")
    ap.add_argument("--think-check", nargs="+", metavar="RUN", help="thinking text, tags and cut replies")
    args = ap.parse_args(argv)
    metrics = tuple(args.metrics)
    if args.pick:
        print(pick_table(args.pick))
    if args.think_check:
        for row in think_rows(args.think_check):
            print(json.dumps(row, ensure_ascii=False))
    if args.paired:
        print(paired_table(args.paired[0], args.paired[1], metrics))
    if args.runs:
        print(run_table(args.runs, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
