"""Measure quality through the running service: every item goes to POST /summarize and the returned record is
scored like any evaluation run (same reports/ layout, so `compare` reads it).

    uv run python -m call_summary.service_eval --data datasets/dev.jsonl --official --label s5-svc

A request the service rejects (502) or fails (other status) scores as a format failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .dataset import Item, load_items, write_jsonl
from .evaluate import _file_hash, _git, _git_strict, summary_table
from .schema import ParseResult, parse_reply
from .scoring import ItemScore, score_item


def run_service_items(
    client, items: Sequence[Item], progress: bool = False
) -> tuple[list[dict], list[ItemScore]]:
    rows: list[dict] = []
    scores: list[ItemScore] = []
    for n, it in enumerate(items, 1):
        t0 = time.perf_counter()
        try:
            r = client.post("/summarize", json={"domain": it.domain, "turns": it.turns})
            status, body = r.status_code, r.json()
        except Exception as exc:  # noqa: BLE001 - a transport failure is a failed item, not a crashed run
            status, body = f"error:{type(exc).__name__}", {}
        latency = time.perf_counter() - t0
        if status == 200:
            parsed = parse_reply(json.dumps(body["record"], ensure_ascii=False))
        else:
            parsed = ParseResult(json_ok=False, schema_ok=False, error=f"service {status}")
        score = score_item(it.item_id, it.domain, it.gold(), parsed, it.transcript, it.split.endswith("-asr"))
        scores.append(score)
        rows.append(
            {
                "item_id": it.item_id,
                "domain": it.domain,
                "status": status,
                "attempts": body.get("attempts") if status == 200 else None,
                "latency_s": round(latency, 3),
                "service_latency_s": body.get("latency_s") if status == 200 else None,
                "pred": parsed.record.model_dump() if parsed.record else None,
                "detail": None if status == 200 else body,
                "score": score.to_dict(),
            }
        )
        if progress and (n % 20 == 0 or n == len(items)):
            print(
                f"[{n}/{len(items)}] exact={sum(s.exact for s in scores) / n:.3f}",
                file=sys.stderr,
                flush=True,
            )
    return rows, scores


def service_summary(rows: Sequence[dict], scores: Sequence[ItemScore]) -> dict:
    table = summary_table(scores)
    lat = sorted(r["latency_s"] for r in rows if r["status"] == 200)
    table["latency_s"] = {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))]} if lat else {}
    table["status"] = {str(k): v for k, v in sorted(Counter(str(r["status"]) for r in rows).items())}
    table["attempts"] = {
        str(k): v for k, v in sorted(Counter(r["attempts"] for r in rows if r["attempts"]).items())
    }
    return table


def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8072")
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="service")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--official", action="store_true", help="clean tree required; writes to reports/")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    items = load_items(data_path)
    if any(it.split.startswith("test") for it in items) and not args.allow_test:
        ap.error("test items: pass --allow-test once the stage's pick is fixed")
    if args.official:
        if args.limit:
            ap.error("--official runs use the whole file")
        try:
            dirty = _git_strict("status", "--porcelain", "--untracked-files=no")
        except Exception as exc:  # noqa: BLE001
            ap.error(f"--official needs git to verify a clean tree: {exc}")
        if dirty:
            ap.error("--official needs a clean working tree")
    items = items[: args.limit]

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        health = client.get("/health").json()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-service-{args.label}"
        out_dir = Path("reports" if args.official else "outputs/runs") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": run_id,
            "provider": f"service:{health.get('model')}",
            "service_health": health,
            "url": args.url,
            "data": str(data_path),
            "data_sha": _file_hash(data_path),
            "n_items": len(items),
            "prompt_version": "p1",
            "prompt_hash": health.get("prompt_hash"),
            "git_commit": _git("rev-parse", "HEAD"),
            "official": args.official,
            "started": stamp,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows, scores = run_service_items(client, items, progress=True)
        stats = client.get("/stats").json()
    write_jsonl(out_dir / "items.jsonl", rows)
    table = service_summary(rows, scores)
    table["service_stats"] = stats
    (out_dir / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v, 4) for k, v in table["point"].items()}, ensure_ascii=False))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
