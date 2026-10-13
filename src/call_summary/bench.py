"""Serving efficiency of an Ollama model: dataset requests sent one at a time, built as the service does.

    uv run --no-sync python -m call_summary.bench --data datasets/dev.jsonl --model call-summary-4b-dq:q8_0 \
        --out reports/s5c-bench-q8_0.json --reference reports/<earlier evaluate run of the same model>

The provider is the service's (num_ctx 4096, num_gpu left to Ollama, temperature 0, seed 0, no schema; prompt
p1, 0-shot). Per request it keeps the client round trip and Ollama's counters; generation tok/s is
eval_count / eval_duration. The model is unloaded first; nvidia-smi (memory.used, utilization.gpu) and the
system CPU are sampled before loading (baseline) and every 0.5 s during the run (peak); /api/ps gives the
model's size and size_vram once it is loaded, and the Ollama server log its layer placement and GPU buffer
sizes. --reference compares every reply with an earlier run of the
same model (greedy decoding: identical text is expected). The run is descriptive and picks nothing.
"""

# [파일 역할] 5단계 "효율 추가 측정"(2026-10-03) 도구다. 서빙 모델 하나에 dev 요청을 서비스와 같은 조건으로 한
# 건씩 보내며 지연, 생성 속도(토큰/초), Ollama가 보고하는 VRAM, nvidia-smi로 본 최대 GPU 메모리, 서버 로그의
# 층 배치·버퍼 크기를 한 보고서(JSON)로 남긴다.
# [왜 필요한가] 계획의 효율 지표(초당 토큰, 최대 GPU 메모리, 모델 VRAM) 가운데 5단계 첫 측정에서 빠진 것을
# 채운다(docs/experiments.md "효율 추가 측정"). 이 측정으로 README의 VRAM(q8_0 4,731 MiB 등)과 생성
# 속도(58·37·81토큰/초) 수치가 나왔다. 아무것도 고르지 않는 기술용 측정이다.
# [설계 이유] 지연은 다른 프로그램의 GPU 사용에 크게 흔들리므로, 재기 전에 CPU·GPU가 쉬는 상태를 기다리고(최대
# --wait-idle분) 그래도 아니면 그대로 재되 그 조건을 보고서에 남긴다. 첫 요청은 모델 적재가 들어가 따로
# 적는다. --reference로 예전 실행과 답이 글자까지 같은지 대 보아, 탐욕적 디코딩에서 실행기 버전이 바뀌어도
# 답이 같은지 확인한다(240/240 같았다).
# [흐름] 다른 모델이 올라가 있으면 거부 → 쉬는 상태 대기 → 기준 표본(모델 올리기 전) → 표본기를 켜고 한 건씩
# 요청 → 첫 요청 뒤 /api/ps와 서버 로그 읽기 → 모델 내리기 → 채점·답 대조 → 보고서 저장.

from __future__ import annotations

# threading·subprocess: nvidia-smi를 반복 모드로 띄워 두고 다른 스레드에서 출력 줄을 읽어 표본을 모은다.
# hashlib: 답 글의 짧은 해시를 남겨 대조를 쉽게 한다.
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path

# 서비스와 같은 요청을 만들려고 프롬프트(build_messages, p1 0-shot)와 서비스의 NUM_CTX(4,096)를 그대로
# 가져온다. 정확도 확인용으로 채점기(score_item, exact_rate)도 쓴다. 백분위는 scoring.pctl(evaluate와 같은
# 계산: p50은 가운데 원소, p95는 int(0.95·(n−1)) 위치)이라 이전 실행 기록의 지연과 그대로 견줄 수 있다.
from .dataset import load_items
from .evaluate import _git
from .prompts import PROMPT_VERSION, build_messages, prompt_hash
from .providers import OllamaProvider
from .schema import parse_reply, record_json_schema
from .scoring import exact_rate, pctl, score_item
from .service import NUM_CTX

# 쉬는 상태 기준(효율 추가 측정 규칙): 1분 표본에서 시스템 CPU 평균 15% 미만, GPU 사용률 평균 50% 미만, 모델을
# 올리기 전 다른 프로그램의 GPU 메모리 최대 2,560 MiB(2.5 GB) 미만. 배경화면 프로그램이 GPU 사용률 30~40%,
# 메모리 약 1 GB를 늘 써서, 이것은 "사용 중"으로 치지 않도록 기준을 정했다(같은 규칙).
# Idle rule (docs/experiments.md, stage 5 efficiency): system CPU and GPU utilization over a 60 s sample,
# and GPU memory held by everything else before the model loads.
IDLE_CPU = 15.0
IDLE_GPU_UTIL = 50.0
IDLE_OTHER_MIB = 2560


# 토큰 수 ÷ 시간(나노초를 초로 바꿈) = 초당 토큰. 시간이 0이거나 없으면 None.
def _rate(tokens: int, ns: int) -> float | None:
    return round(tokens / (ns / 1e9), 1) if ns else None


# 요청별 기록을 요약한다. 첫 요청은 모델 적재·GPU 준비가 들어가 따로 적고 나머지 지표에서 뺀다.
# 생성 속도는 전체 eval 토큰 합 ÷ eval 시간 합(요청별 속도의 p50도 함께). 프롬프트 쪽은 속도가 아니라 요청당
# 처리 시간 p50으로 적는다: 서버가 공통 앞부분(시스템 프롬프트)을 캐시에서 재사용하는데 Ollama는 전체 길이를
# 세어 보고하므로, 나누면 속도가 부풀기 때문이다(docstring, 규칙).
def summarize_rows(rows: Sequence[dict]) -> dict:
    """Rows in request order. The first request loads the model and warms the GPU kernels (its prompt pass
    took 40 s in a trial), so it is reported on its own and left out of every other figure.

    Prompt figures are times, not rates: Ollama counts the whole prompt in prompt_eval_count while the
    server reuses the cached shared prefix (the system prompt): tokens / prompt_eval_duration overstates."""
    warm_rows = list(rows[1:])
    warm = sorted(r["wall_s"] for r in warm_rows)
    gen_n = sum(r["eval_count"] or 0 for r in warm_rows)
    gen_ns = sum(r["eval_duration_ns"] or 0 for r in warm_rows)
    per_req = sorted(
        r["eval_count"] / (r["eval_duration_ns"] / 1e9)
        for r in warm_rows
        if r["eval_count"] and r["eval_duration_ns"]
    )
    prompt_ms = sorted(r["prompt_eval_duration_ns"] / 1e6 for r in warm_rows if r["prompt_eval_duration_ns"])
    comp = sorted(r["eval_count"] or 0 for r in warm_rows)
    first = rows[0] if rows else {}
    out: dict = {
        "n": len(rows),
        "first_request": {
            "wall_s": first.get("wall_s"),
            "load_s": round((first.get("load_duration_ns") or 0) / 1e9, 2),
            "prompt_eval_s": round((first.get("prompt_eval_duration_ns") or 0) / 1e9, 2),
        },
        "latency_s": None,
        "gen_tok_s": _rate(gen_n, gen_ns),
        "gen_tok_s_per_request_p50": round(pctl(per_req, 0.5), 1) if per_req else None,
        "prompt_eval_ms_p50": round(pctl(prompt_ms, 0.5), 1) if prompt_ms else None,
        "prompt_tokens_mean": (
            round(sum(r["prompt_eval_count"] or 0 for r in warm_rows) / len(warm_rows), 1)
            if warm_rows
            else None
        ),
        "completion_tokens": (
            {"mean": round(gen_n / len(warm_rows), 1), "p95": pctl(comp, 0.95)} if warm_rows else None
        ),
        # Ollama 카운터가 빠진 요청 수. 0이 아니면 속도 수치를 믿기 어렵다는 표시다.
        "missing_counters": sum(1 for r in rows if r["eval_count"] is None or r["eval_duration_ns"] is None),
    }
    if warm:
        out["latency_s"] = {
            "n": len(warm),
            "p50": round(pctl(warm, 0.5), 3),
            "p95": round(pctl(warm, 0.95), 3),
            "mean": round(statistics.fmean(warm), 3),
            "max": round(warm[-1], 3),
        }
    return out


# Ollama 서버 로그에서 뽑을 줄: GPU에 올린 층 수(offloaded), 모델·KV 캐시·계산 버퍼 크기, Flash Attention 사용
# 여부. README의 "KV 캐시 576 MiB 포함" 같은 VRAM 내역이 여기서 나왔다.
LOG_PATTERNS = ("offloaded", "model buffer size", "KV buffer size", "compute buffer size", "Flash Attention")


# 측정 시작 때 기록해 둔 로그 크기(바이트 위치) 뒤에 새로 쓰인 줄만 읽어, 이번 모델 적재에 대한 줄만 고른다.
# 로그가 없으면 빈 목록.
def server_log_lines(path: Path | None, offset: int) -> list[str]:
    """Lines about layer placement and GPU buffers that Ollama logged after `offset` (a byte position)."""
    if path is None or not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", errors="replace")
    return [ln.strip() for ln in text.splitlines() if any(p in ln for p in LOG_PATTERNS)]


# nvidia-smi CSV 한 줄("사용 메모리 MiB, 사용률 %")을 숫자 쌍으로 바꾼다. 형식이 다른 줄은 None으로 버린다.
def parse_smi_line(line: str) -> tuple[int, int] | None:
    """'1005, 36' -> (memory.used MiB, utilization.gpu %); None for anything else."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


# 표본의 최소·평균·최대. 표본이 없으면 None.
def _stats(vals: Sequence[float]) -> dict | None:
    if not vals:
        return None
    return {"min": min(vals), "mean": round(statistics.fmean(vals), 1), "max": max(vals)}


# GPU·CPU 표본기. nvidia-smi를 -lms(밀리초 반복) 모드로 한 번 띄워 0.5초마다 나오는 줄을 읽고, 줄마다 psutil의
# 시스템 CPU 사용률을 함께 적는다.
# Windows(WDDM)에서는 프로세스별 GPU 메모리가 나오지 않아 전체 사용량을 재고, 모델을 올리기 전 기준값을 빼서
# 이 모델이 더한 양을 낸다(규칙 "최대 GPU 메모리").
class Sampler:
    """nvidia-smi in loop mode plus psutil's system CPU, one sample per nvidia-smi line."""

    def __init__(self, period_ms: int = 500) -> None:
        self.period_ms = period_ms
        self.mem: list[int] = []
        self.util: list[int] = []
        self.cpu: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Sampler:
        # psutil이 없으면 CPU는 기록하지 않고 GPU만 잰다. cpu_percent(None)을 한 번 먼저 불러 기준점을 잡아야
        # 다음 호출부터 의미 있는 값이 나온다.
        try:
            import psutil

            psutil.cpu_percent(None)
            cpu = psutil.cpu_percent
        except ImportError:  # CPU is then not recorded
            cpu = None
        self._proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.period_ms),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )

        # nvidia-smi 표준 출력을 줄 단위로 읽는 스레드 본문. daemon 스레드라 주 프로그램이 끝나면 함께 끝난다.
        def read() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                got = parse_smi_line(line)
                if got:
                    self.mem.append(got[0])
                    self.util.append(got[1])
                    if cpu is not None:
                        self.cpu.append(cpu(None))

        self._thread = threading.Thread(target=read, daemon=True)
        self._thread.start()
        return self

    # nvidia-smi를 끝내고 읽기 스레드를 기다린 뒤 표본 수와 메모리·사용률·CPU 통계를 돌려준다.
    def stop(self) -> dict:
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._thread is not None:
            self._thread.join(timeout=10)
        return {
            "samples": len(self.mem),
            "gpu_mem_used_mib": _stats(self.mem),
            "gpu_util_pct": _stats(self.util),
            "cpu_pct": _stats(self.cpu),
        }


# seconds초 동안만 표본을 모은다(쉬는 상태 확인과 기준값 측정에 쓴다).
def sample_for(seconds: float) -> dict:
    s = Sampler().start()
    time.sleep(seconds)
    return s.stop()


# 표본이 쉬는 상태 기준 세 가지를 모두 만족하는지. CPU 값이 없는 등 표본이 불완전하면 쉬지 않는 것으로 친다.
def is_idle(sample: dict) -> bool:
    try:
        return (
            sample["cpu_pct"]["mean"] < IDLE_CPU
            and sample["gpu_util_pct"]["mean"] < IDLE_GPU_UTIL
            and sample["gpu_mem_used_mib"]["max"] < IDLE_OTHER_MIB
        )
    except (KeyError, TypeError):
        return False


# 1분 표본을 poll_s(3분)마다 되풀이하며 쉬는 상태를 기다린다. max_minutes가 지나면 마지막 표본을 그대로
# 돌려주고 측정은 계속한다(쉬지 않은 조건은 보고서의 idle_check에 남는다). 실제 측정에서는 60분 동안 기준을
# 맞추지 못해 그대로 재고 조건을 문서에 적었다(docs/experiments.md "효율 추가 측정 결과").
# 기본값은 규칙의 60분이고 --wait-idle로 바꿀 수 있다.
def wait_idle(max_minutes: float, poll_s: float = 180.0, sample_s: float = 60.0) -> dict:
    """Sample for a minute; repeat every `poll_s` until idle or `max_minutes` pass. Returns the last one."""
    deadline = time.monotonic() + max_minutes * 60
    tries = 0
    while True:
        tries += 1
        sample = sample_for(sample_s)
        idle = is_idle(sample)
        print(f"idle check {tries}: {json.dumps(sample)} idle={idle}", file=sys.stderr, flush=True)
        if idle or time.monotonic() + poll_s > deadline:
            return {"idle": idle, "checks": tries, "sample": sample, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        time.sleep(poll_s)


# Ollama API GET 도우미(/api/ps, /api/version).
def ollama_get(base: str, path: str) -> dict:
    import httpx

    r = httpx.get(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


# keep_alive 0으로 요청해 모델을 내리고, /api/ps가 빌 때까지 최대 60초 기다린다. 측정이 끝난 뒤 GPU를 다른
# 작업에 돌려준다.
def unload(base: str, model: str) -> None:
    import httpx

    httpx.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=120).raise_for_status()
    for _ in range(60):
        if not ollama_get(base, "/api/ps").get("models"):
            return
        time.sleep(1)


# 명령줄 진입점. --server-log의 기본값은 Windows Ollama의 로그 위치(%LOCALAPPDATA%\Ollama\server.log)다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="Ollama model name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="earlier evaluate run dir of the same model (items.jsonl)")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--wait-idle", type=float, default=60.0, help="minutes to wait for an idle GPU and CPU")
    ap.add_argument("--limit", type=int)
    ap.add_argument(
        "--server-log",
        default=str(Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "server.log"),
        help="Ollama server log, read for layer placement and GPU buffer sizes",
    )
    args = ap.parse_args(argv)

    items = load_items(args.data)[: args.limit]
    log_path = Path(args.server_log) if args.server_log else None
    # 다른 모델이 이미 올라가 있으면 재지 않는다(GPU 작업은 한 번에 하나). 대기 중에 누가 모델을 올려도 다시
    # 확인해 멈춘다.
    if ollama_get(args.ollama, "/api/ps").get("models"):
        ap.error("another model is loaded in Ollama; stop it first (one GPU job at a time)")
    idle = wait_idle(args.wait_idle)
    if ollama_get(args.ollama, "/api/ps").get("models"):
        ap.error("a model was loaded in Ollama while waiting; not measuring next to another job")
    # 모델을 올리기 전 5초 표본이 기준값(다른 프로그램이 쓰는 GPU 메모리)이다. 서버 로그도 지금 크기를 기억해
    # 두어 이번 적재 뒤의 줄만 읽는다.
    baseline = sample_for(5)  # nothing of ours is loaded yet
    log_offset = log_path.stat().st_size if log_path and log_path.exists() else 0

    # 서비스와 같은 실행기 설정: num_ctx 4,096, num_gpu는 Ollama가 정함, temperature 0, seed 0, 스키마 강제
    # 없음(use_schema 기본값 False라 아래에서 넘기는 schema는 쓰이지 않는다).
    provider = OllamaProvider(model=args.model, base_url=args.ollama, num_ctx=NUM_CTX)  # as in the service
    schema = record_json_schema()
    rows: list[dict] = []
    ps: dict | None = None
    load_log: list[str] = []
    sampler = Sampler().start()
    t_run = time.perf_counter()
    # 요청은 한 건씩 차례로 보낸다(동시 요청은 loadtest가 잰다). 오류가 나도 finally에서 표본기를 반드시
    # 멈춘다.
    try:
        for n, it in enumerate(items, 1):
            t0 = time.perf_counter()
            reply = provider.generate(build_messages(it.domain, it.transcript), json_schema=schema)
            wall = time.perf_counter() - t0
            tm = reply.timings or {}
            rows.append(
                {
                    "item_id": it.item_id,
                    "wall_s": round(wall, 3),
                    "prompt_eval_count": reply.prompt_tokens,
                    "eval_count": reply.completion_tokens,
                    "prompt_eval_duration_ns": tm.get("prompt_eval_duration"),
                    "eval_duration_ns": tm.get("eval_duration"),
                    "load_duration_ns": tm.get("load_duration"),
                    "total_duration_ns": tm.get("total_duration"),
                    # 답의 짧은 해시와 원문을 남겨, 다른 실행과 답이 같은지 바로 비교할 수 있게 한다.
                    "reply_sha": hashlib.sha256(reply.text.encode("utf-8")).hexdigest()[:16],
                    "reply": reply.text,
                }
            )
            # 첫 요청이 끝나 모델이 올라간 직후에 한 번만 /api/ps(size, size_vram, context_length)와 서버
            # 로그를 읽는다.
            if ps is None:
                ps = ollama_get(args.ollama, "/api/ps")
                load_log = server_log_lines(log_path, log_offset)
            if n % 20 == 0 or n == len(items):
                print(f"[{n}/{len(items)}] last={wall:.2f}s", file=sys.stderr, flush=True)
    finally:
        during = sampler.stop()
    run_s = time.perf_counter() - t_run
    unload(args.ollama, args.model)

    # 속도 측정이지만 같은 답이 나왔는지 확인하려고 채점도 한다(exact가 예전 실행과 같아야 한다).
    scores = [
        score_item(
            it.item_id,
            it.domain,
            it.gold(),
            parse_reply(r["reply"]),
            it.transcript,
            it.split.endswith("-asr"),
        )
        for it, r in zip(items, rows, strict=True)
    ]
    ref = None
    # 예전 evaluate 실행의 답과 글자까지 같은 건수를 센다. 탐욕적 디코딩이라 같은 가중치면 같은 답이
    # 기대된다(2026-10-03에는 Ollama 0.34.2 → 0.35.0에서도 240/240 같았다).
    if args.reference:
        ref_rows = {
            json.loads(line)["item_id"]: json.loads(line)["reply"]
            for line in (Path(args.reference) / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        same = sum(1 for r in rows if ref_rows.get(r["item_id"]) == r["reply"])
        ref = {"run": Path(args.reference).name, "identical_replies": same, "n": len(rows)}

    # /api/ps 목록에서 이 모델의 항목만 고른다(size_vram이 size와 같으면 모두 GPU에 올라간 것).
    loaded = [m for m in (ps or {}).get("models", []) if m.get("name", "").startswith(args.model)]
    report = {
        "model": args.model,
        "data": args.data,
        "n_items": len(items),
        "options": {
            "num_ctx": provider.num_ctx,
            "num_gpu": provider.num_gpu,
            "temperature": provider.temperature,
            "seed": provider.seed,
            "num_predict": provider.num_predict,
            "schema": provider.use_schema,
        },
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "ollama_version": ollama_get(args.ollama, "/api/version").get("version"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - run_s)),
        "run_s": round(run_s, 1),
        "idle_check": idle,
        "baseline": baseline,
        "during": during,
        # 최대 GPU 메모리 = 실행 중 nvidia-smi 최대값 − 모델을 올리기 전 기준 최대값. 다른 프로그램의 몫을 빼
        # 이 모델이 더한 양에 가깝게 잰다.
        "peak_over_baseline_mib": (
            during["gpu_mem_used_mib"]["max"] - baseline["gpu_mem_used_mib"]["max"]
            if during["gpu_mem_used_mib"] and baseline["gpu_mem_used_mib"]
            else None
        ),
        "api_ps": loaded[0] if loaded else None,
        "api_ps_all": ps,
        "server_log": load_log,
        "summary": summarize_rows(rows),
        "exact": round(exact_rate(scores), 4),
        "reference": ref,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    # 화면에는 핵심만(최대 메모리, exact, 답 대조, 요약, size_vram) 한 줄 JSON으로 찍는다. 요청별 기록까지 든
    # 전체 보고서는 --out 파일에 있다.
    brief = {k: report[k] for k in ("model", "peak_over_baseline_mib", "exact", "reference")}
    brief |= {"summary": report["summary"], "size_vram": (report["api_ps"] or {}).get("size_vram")}
    print(json.dumps(brief, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
