"""띄워 둔 서비스를 거쳐 품질을 잰다: 건마다 POST /summarize로 보내고, 돌아온 기록을 일반 평가 실행과 똑같이
채점한다(reports/ 형식이 같아 `compare`가 그대로 읽는다).

    uv run python -m call_summary.service_eval --data datasets/dev.jsonl --official --label s5-svc

서비스가 거절한(502) 요청이나 실패한(그 밖의 상태) 요청은 형식 실패로 채점한다.

시작할 때 서비스의 /ready를 먼저 불러(모델 서버 접속, 모델 태그, digest 확인) 그 응답을 /health와 함께
manifest에 남긴다. 공식 실행(--official)은 /ready가 200일 때만 돈다.
"""

# [파일 역할] 띄워 둔 서비스에 데이터셋의 건을 하나씩 POST /summarize로 보내고, 돌아온 기록을 evaluate.py와
# 같은 채점기로 채점해 같은 reports/ 형식(manifest.json, items.jsonl, summary.json)으로 남긴다.
# [왜 필요한가] 모델을 직접 잰 점수와 서비스(프롬프트 조립·검증·재시도·HTTP)를 거친 점수가 같은지 확인해야
# "서비스 경로가 품질을 잃지 않음"을 판정할 수 있다(docs/experiments.md 5단계 "다시 재기": 서비스 dev exact −
# 직접 잰 dev exact의 짝지은 차이 하한 > −3%p). 형식이 같으므로 compare.py --paired로 바로 짝지어 비교한다.
# [흐름] main: 인수·test 보호·공식 실행 확인 → service_identity(/ready → /health: 모델·프롬프트·가중치 지문.
# 공식 실행은 /ready 200일 때만) → run_service_items(건마다 요청·채점) → /stats → service_summary → 파일 쓰기.

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

# evaluate.py의 도우미(데이터 파일 해시, git 커밋, 깨끗한 트리 확인, 요약 표)를 그대로 써서 일반 평가 실행과
# 기록 형식·규칙을 맞춘다.
from .dataset import Item, load_items, write_jsonl
from .evaluate import _file_hash, _git, _git_strict, summary_table
from .schema import ParseResult, parse_reply
from .scoring import ItemScore, pctl, score_item


# 건마다 서비스에 요청하고 채점한다. client는 httpx.Client나 FastAPI TestClient처럼 post()를 가진 객체면
# 된다(tests/test_service_eval.py는 ScriptedProvider로 만든 앱의 TestClient를 넘긴다).
# 200이 아닌 응답(502 거절, 413, 503 등)과 전송 실패는 형식 실패로 채점한다(5단계 확정 규칙: 거절은 형식
# 실패로 센다). 한 건의 실패가 실행 전체를 멈추지 않게 예외를 잡아 상태를 "error:<예외 이름>"으로 남긴다.
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
        except Exception as exc:  # noqa: BLE001 - 전송 실패는 실행 중단이 아니라 그 건의 실패다
            status, body = f"error:{type(exc).__name__}", {}
        # 클라이언트가 잰 왕복 시간. 서비스가 응답에 적은 자체 지연(service_latency_s)과 함께 남긴다.
        latency = time.perf_counter() - t0
        # 응답의 기록을 JSON 글로 되돌려 parse_reply에 넣는다. evaluate.py가 모델 답 글을 읽는 것과 같은
        # 경로로 채점하기 위해서다.
        if status == 200:
            parsed = parse_reply(json.dumps(body["record"], ensure_ascii=False))
        else:
            parsed = ParseResult(json_ok=False, schema_ok=False, error=f"service {status}")
        # 분할 이름이 "-asr"로 끝나면(음성 조건) 환각 검사에 넓은 번호 규칙을 쓴다. evaluate.py와 같다.
        score = score_item(it.item_id, it.domain, it.gold(), parsed, it.transcript, it.split.endswith("-asr"))
        scores.append(score)
        # 건별 기록: 상태, 시도 횟수(재시도 비율 집계용), 지연, 예측 기록, 실패면 응답 본문(거절 사유), 점수.
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


# 서비스가 어떤 모델·프롬프트·가중치로 도는지 받아 manifest에 남길 두 값을 돌려준다. 입력: client(get()을 가진
# 객체: httpx.Client, FastAPI TestClient). 출력: (/ready 기록, /health 응답 JSON). /ready 기록은 응답
# JSON(status ready·not_ready, reason, model, model_digest, ollama_version, expected_digest)에 HTTP 상태
# http_status를 더한 dict이고, 보내지 못했거나 답이 JSON 객체가 아니면
# {"http_status": "error:<예외 이름>"}이다(/ready가 없는 예전 서비스는 404로 남는다).
# /ready를 /health보다 먼저 부르는 이유: /health의 model_digest·ollama_version은 서비스가 시작할 때 한
# 번(요청마다 2초 한도) 조회한 값이고 /ready가 모델 서버를 다시 확인할 때만 갱신된다. 서비스가 모델 서버보다
# 먼저 떴으면 null로 남아 있어, /health만 남기면 잰 모델과 서빙 모델을 digest로 맞춰 볼 수 없는 실행이 생긴다.
# /ready가 모델 서버에 닿으면 그 지문이 뒤이어 받는 /health에도 들어간다. /health 요청이 실패하면 전처럼
# 예외가 올라온다(서비스가 없으면 잴 수 없다).
def service_identity(client) -> tuple[dict, dict]:
    try:
        r = client.get("/ready")
        ready = {"http_status": r.status_code, **r.json()}
    except Exception as exc:  # noqa: BLE001 - 닿지 못했거나 JSON 객체가 아닌 답: 준비 안 됨으로 기록한다
        ready = {"http_status": f"error:{type(exc).__name__}"}
    health = client.get("/health").json()
    return ready, health


# evaluate.summary_table(지표와 구간)에 서비스 쪽 집계를 더한다: 200 응답의 왕복 지연 p50·p95(scoring.pctl),
# 상태 코드별 수, 시도 횟수별 수(재시도·거절 비율을 문서에 적는 근거, docs/experiments.md 5단계
# "재시도 0, 거절 0").
def service_summary(rows: Sequence[dict], scores: Sequence[ItemScore]) -> dict:
    table = summary_table(scores)
    lat = sorted(r["latency_s"] for r in rows if r["status"] == 200)
    table["latency_s"] = {"p50": pctl(lat, 0.5), "p95": pctl(lat, 0.95)} if lat else {}
    table["status"] = {str(k): v for k, v in sorted(Counter(str(r["status"]) for r in rows).items())}
    table["attempts"] = {
        str(k): v for k, v in sorted(Counter(r["attempts"] for r in rows if r["attempts"]).items())
    }
    return table


# 명령줄 입구. httpx는 이 명령에서만 필요해 함수 안에서 불러온다.
def main(argv: list[str] | None = None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8072")
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="service")
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--official",
        action="store_true",
        help="깨끗한 작업 트리에서만, 서비스의 /ready가 200일 때만 돌고 결과를 reports/에 쓴다",
    )
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    items = load_items(data_path)
    # test 분할은 단계에서 고른 것만 한 번 잰다는 규칙을 지키려고, --allow-test 없이는 test 건을 보내지
    # 않는다(docs/experiments.md 첫머리: test 세트는 단계가 끝날 때 고른 것만 한 번 잰다).
    if any(it.split.startswith("test") for it in items) and not args.allow_test:
        ap.error("test items: pass --allow-test once the stage's pick is fixed")
    # 공식 실행(reports/에 남김)은 전체 파일만, 그리고 추적 파일에 바뀐 것이 없는 깨끗한 작업 트리에서만 돈다.
    # 결과가 어느 커밋의 코드로 나왔는지 확실히 하려는 규칙이다(docs/experiments.md "공통"). git을 부르지
    # 못하면 깨끗하다고 넘기지 않고 멈춘다(evaluate.py와 같은 규칙).
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
        # 서비스가 어떤 모델과 프롬프트, 가중치(digest)로 돌고 있는지 /ready와 /health에서 받아 manifest에
        # 남긴다(service_identity의 주석). 서비스는 늘 p1 프롬프트를 쓰므로 prompt_version은 p1로 적는다.
        ready, health = service_identity(client)
        # /ready가 200이 아니면(모델 서버에 닿지 못함, 모델 태그 없음, digest가 기대값과 다름, 예열 중) 공식
        # 실행은 요청을 하나도 보내지 않고 멈춘다: 어느 가중치를 쟀는지 남길 수 없거나 다른 가중치를 재는
        # 실행을 공식 기록으로 두지 않는다. 시험 실행은 경고만 찍고 돈다(그 응답이 manifest에 남는다).
        # 서비스는 /ready의 조회 결과를 10초 동안 캐시하므로 모델 서버를 막 띄웠으면 잠시 뒤에 다시 돌린다.
        if ready["http_status"] != 200:
            why = f"service /ready answered {ready['http_status']} (reason: {ready.get('reason')})"
            if args.official:
                ap.error(
                    f"{why}; --official needs a ready service (model server reachable, model tag present, "
                    "digest as expected). /ready results are cached for 10 s: retry shortly"
                )
            print(f"warning: {why}; recorded in the manifest as service_ready", file=sys.stderr)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-service-{args.label}"
        # 공식 실행은 커밋하는 reports/에, 시험 실행은 커밋하지 않는 outputs/runs/에 둔다.
        out_dir = Path("reports" if args.official else "outputs/runs") / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # 데이터 경로는 evaluate와 같이 슬래시 꼴(as_posix)로 적어 다른 OS에서도 그대로 열린다(기록된 예전
        # manifest의 역슬래시 꼴은 dataset.manifest_data가 읽는다).
        manifest = {
            "run_id": run_id,
            "provider": f"service:{health.get('model')}",
            "service_health": health,
            "service_ready": ready,
            "url": args.url,
            "data": data_path.as_posix(),
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
        # 측정이 끝난 뒤 서비스의 결과별 수(ok, retried_ok, rejected, model_unavailable)를 받아 요약에 함께
        # 남긴다.
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
