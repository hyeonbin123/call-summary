"""Send dataset calls to a running service with N concurrent clients; report throughput, latency, errors.

    uv run python -m call_summary.loadtest --data datasets/dev.jsonl --concurrency 1 2 4

Latency is measured by the client (the whole HTTP round trip). A request that fails in transport (timeout,
dropped connection) counts as status "error:<exception>". Results go to --out as JSON after every stage.
"""

# [파일 역할] 5단계(서빙)의 부하 시험 도구다. 떠 있는 FastAPI 서비스(/summarize)에 데이터셋 통화를 동시 요청
# N개로 보내 처리량(요청/초), 지연 분포, 상태 코드별 개수를 잰다.
# [왜 필요한가] 5단계 측정 규칙이 "dev 240건을 동시 요청 1, 2, 4로 보내 처리량, 지연 p50/p95, 오류 수를
# 잰다"고 정해 두었다(docs/experiments.md 5단계). 이 측정으로 Ollama가 요청을 하나씩 처리해 동시 요청을 늘려도
# 처리량은 약 0.4요청/초에서 멈추고 지연만 늘어난다는 것을 확인했다(README 5단계).
# [흐름] main이 /health로 서비스 상태를 기록 → 동시성 단계마다 run_load → 단계가 끝날 때마다 결과 JSON 저장 →
# 마지막에 서비스의 /stats를 함께 저장.

from __future__ import annotations

# statistics: 지연의 중앙값·평균. time.perf_counter: 단조 증가 고해상도 시계라 경과 시간 측정에 쓴다.
# ThreadPoolExecutor: 동시 요청을 스레드로 흉내 낸다(요청 대부분이 네트워크·모델 대기라 스레드로 충분하다).
import argparse
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dataset import Item, load_items


# 데이터셋 한 건을 /summarize 요청 본문({domain, turns})으로 바꾼다. 서비스 API가 받는 형식 그대로다.
def request_body(item: Item) -> dict:
    return {"domain": item.domain, "turns": item.turns}


# items 전체를 동시성 concurrency로 보내고 결과 요약 dict를 돌려준다.
# client를 인자로 받는 이유: 실제 측정에서는 httpx.Client를, 테스트(tests/test_loadtest.py)에서는 오프라인
# 서비스의 FastAPI TestClient나 일부러 예외를 내는 가짜 client를 넣어 같은 코드를 GPU 없이 확인한다.
# 출력: 동시성, 건수, 상태별 개수, 전체 경과 시간, 처리량, 지연(p50, p95, 평균, 최대).
def run_load(client, items: Sequence[Item], concurrency: int) -> dict:
    """`client` is anything with .post(path, json=...) returning an object with .status_code (httpx.Client,
    FastAPI TestClient)."""

    # 요청 하나를 보내고 (상태 문자열, 걸린 초)를 돌려준다. 지연은 클라이언트에서 잰 HTTP 왕복 전체다.
    def one(item: Item) -> tuple[str, float]:
        t0 = time.perf_counter()
        try:
            status = str(client.post("/summarize", json=request_body(item)).status_code)
        # 타임아웃이나 연결 끊김으로 측정 전체가 죽지 않게, 예외를 "error:<예외 이름>" 상태로 세어 결과에
        # 남긴다(커밋 6238213).
        except Exception as exc:  # noqa: BLE001 - a timeout or dropped connection is a counted error, not a crash
            status = f"error:{type(exc).__name__}"
        return status, time.perf_counter() - t0

    # pool.map은 입력 순서대로 결과를 모은다. max_workers가 곧 동시 요청 수다. 전체 경과 시간(wall)은 처리량
    # 계산에 쓴다.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, items))
    wall = time.perf_counter() - t0
    # 지연 분포는 200으로 끝난 요청만으로 낸다. 실패 요청의 짧은 지연이 섞여 분포가 좋아 보이지 않게 하려는
    # 것이다. 처리량은 실패까지 포함한 전체 요청 수로 낸다.
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
    # p95는 정렬한 목록에서 0.95 위치의 값을 고른다(보간 없이 가장 가까운 순위, 마지막 원소를 넘지 않게
    # 자른다).
    if lat:
        out["latency_s"] = {
            "p50": round(statistics.median(lat), 3),
            "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
            "mean": round(statistics.fmean(lat), 3),
            "max": round(lat[-1], 3),
        }
    return out


# 명령줄 진입점. 기본 주소 127.0.0.1:8072는 README의 서비스 실행 포트와 같다. --concurrency에 여러 값을 주면
# 같은 데이터로 단계를 차례로 돈다.
# httpx는 실제 HTTP 측정에만 필요해 main 안에서 불러온다.
def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8072")
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--out", help="JSON file for the results")
    # 요청 하나의 시간 한도(초, 기본 300). 넘으면 위의 error 상태로 센다.
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)
    items = load_items(args.data)[: args.limit]
    report: dict = {"url": args.url, "data": args.data, "health": None, "runs": [], "service_stats": None}

    # 보고서를 --out에 덮어쓴다. 단계마다 부르므로 나중 단계가 실패해도 끝난 단계의 결과가 파일에 남는다.
    def save() -> None:
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
        # 어떤 모델·프롬프트 지문으로 잰 것인지 남기려고 /health 응답(모델 이름, prompt_hash)을 먼저 기록한다.
        report["health"] = client.get("/health").json()
        for c in args.concurrency:
            res = run_load(client, items, c)
            print(json.dumps(res, ensure_ascii=False), flush=True)
            report["runs"].append(res)
            save()  # finished stages survive a later failure
        # 모든 단계가 끝난 뒤 서비스가 집계한 요청 수·지연 분포(/stats)를 함께 남겨, 클라이언트 쪽 지연과 서버
        # 쪽 지연을 견줄 수 있게 한다.
        report["service_stats"] = client.get("/stats").json()
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
