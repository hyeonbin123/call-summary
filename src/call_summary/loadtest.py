"""Send dataset calls to a running service with N concurrent clients; report throughput, latency, errors.

    uv run python -m call_summary.loadtest --data datasets/dev.jsonl --concurrency 1 2 4

Latency is measured by the client (the whole HTTP round trip). A request that fails in transport (timeout,
dropped connection) counts as status "error:<exception>". Results go to --out as JSON after every stage.
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

    def one(item: Item) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            status = str(client.post("/summarize", json=request_body(item)).status_code)
        except Exception as exc:  # noqa: BLE001 - a timeout or dropped connection is a counted error, not a crash
            status = f"error:{type(exc).__name__}"
        return status, time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, items))
    wall = time.perf_counter() - t0
    lat = sorted(sec for status, sec in results if status == "200")
    codes = Counter(status for status, _ in results)
    out = {
        "concurrency": concurrency,
        "n": len(results),
        "status": dict(sorted(codes.items())),
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
    report: dict = {"url": args.url, "data": args.data, "health": None, "runs": [], "service_stats": None}

    def save() -> None:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        report["health"] = client.get("/health").json()
        for c in args.concurrency:
            res = run_load(client, items, c)
            print(json.dumps(res, ensure_ascii=False), flush=True)
            report["runs"].append(res)
            save()  # finished stages survive a later failure
        report["service_stats"] = client.get("/stats").json()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
