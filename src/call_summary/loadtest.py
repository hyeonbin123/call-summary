"""Send dataset calls to a running service with N concurrent clients; report throughput, latency, errors.

    uv run python -m call_summary.loadtest --data datasets/dev.jsonl --concurrency 1 2 4

Latency is measured by the client (the whole HTTP round trip). Results go to --out as JSON.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dataset import Item, load_items


def request_body(item: Item) -> dict:
    return {"domain": item.domain, "turns": item.turns}


def run_load(client, items: Sequence[Item], concurrency: int) -> dict:
    """`client` is anything with .post(path, json=...) returning an object with .status_code (httpx.Client,
    FastAPI TestClient)."""

    def one(item: Item) -> tuple[int, float]:
        t0 = time.perf_counter()
        r = client.post("/summarize", json=request_body(item))
        return r.status_code, time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, items))
    wall = time.perf_counter() - t0
    lat = sorted(sec for status, sec in results if status == 200)
    codes = Counter(status for status, _ in results)
    out = {
        "concurrency": concurrency,
        "n": len(results),
        "status": {str(k): v for k, v in sorted(codes.items())},
        "wall_s": round(wall, 2),
        "throughput_rps": round(len(results) / wall, 3) if wall else None,
        "latency_s": None,
    }
    if lat:
        out["latency_s"] = {
            "p50": round(statistics.median(lat), 3),
            "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "mean": round(statistics.fmean(lat), 3),
            "max": round(lat[-1], 3),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8072")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--out", help="JSON file for the results")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)
    items = load_items(args.data)[: args.limit]
    runs = []
    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        health = client.get("/health").json()
        for c in args.concurrency:
            res = run_load(client, items, c)
            print(json.dumps(res, ensure_ascii=False), flush=True)
            runs.append(res)
        stats = client.get("/stats").json()
    report = {"url": args.url, "data": args.data, "health": health, "runs": runs, "service_stats": stats}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
