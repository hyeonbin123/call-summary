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

from __future__ import annotations

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

from .dataset import load_items
from .evaluate import _git
from .prompts import PROMPT_VERSION, build_messages, prompt_hash
from .providers import OllamaProvider
from .schema import parse_reply, record_json_schema
from .scoring import exact_rate, score_item
from .service import NUM_CTX

# Idle rule (docs/experiments.md, stage 5 efficiency): system CPU and GPU utilization over a 60 s sample,
# and GPU memory held by everything else before the model loads.
IDLE_CPU = 15.0
IDLE_GPU_UTIL = 50.0
IDLE_OTHER_MIB = 2560


def pctl(sorted_vals: Sequence[float], q: float) -> float:
    """evaluate's convention: p50 = v[n // 2], p95 = v[int(0.95 * (n - 1))]."""
    n = len(sorted_vals)
    return sorted_vals[n // 2] if q == 0.5 else sorted_vals[int(q * (n - 1))]


def _rate(tokens: int, ns: int) -> float | None:
    return round(tokens / (ns / 1e9), 1) if ns else None


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


LOG_PATTERNS = ("offloaded", "model buffer size", "KV buffer size", "compute buffer size", "Flash Attention")


def server_log_lines(path: Path | None, offset: int) -> list[str]:
    """Lines about layer placement and GPU buffers that Ollama logged after `offset` (a byte position)."""
    if path is None or not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(offset)
        text = f.read().decode("utf-8", errors="replace")
    return [ln.strip() for ln in text.splitlines() if any(p in ln for p in LOG_PATTERNS)]


def parse_smi_line(line: str) -> tuple[int, int] | None:
    """'1005, 36' -> (memory.used MiB, utilization.gpu %); None for anything else."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def _stats(vals: Sequence[float]) -> dict | None:
    if not vals:
        return None
    return {"min": min(vals), "mean": round(statistics.fmean(vals), 1), "max": max(vals)}


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


def sample_for(seconds: float) -> dict:
    s = Sampler().start()
    time.sleep(seconds)
    return s.stop()


def is_idle(sample: dict) -> bool:
    try:
        return (
            sample["cpu_pct"]["mean"] < IDLE_CPU
            and sample["gpu_util_pct"]["mean"] < IDLE_GPU_UTIL
            and sample["gpu_mem_used_mib"]["max"] < IDLE_OTHER_MIB
        )
    except (KeyError, TypeError):
        return False


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


def ollama_get(base: str, path: str) -> dict:
    import httpx

    r = httpx.get(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def unload(base: str, model: str) -> None:
    import httpx

    httpx.post(f"{base}/api/generate", json={"model": model, "keep_alive": 0}, timeout=120).raise_for_status()
    for _ in range(60):
        if not ollama_get(base, "/api/ps").get("models"):
            return
        time.sleep(1)


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
    if ollama_get(args.ollama, "/api/ps").get("models"):
        ap.error("another model is loaded in Ollama; stop it first (one GPU job at a time)")
    idle = wait_idle(args.wait_idle)
    if ollama_get(args.ollama, "/api/ps").get("models"):
        ap.error("a model was loaded in Ollama while waiting; not measuring next to another job")
    baseline = sample_for(5)  # nothing of ours is loaded yet
    log_offset = log_path.stat().st_size if log_path and log_path.exists() else 0

    provider = OllamaProvider(model=args.model, base_url=args.ollama, num_ctx=NUM_CTX)  # as in the service
    schema = record_json_schema()
    rows: list[dict] = []
    ps: dict | None = None
    load_log: list[str] = []
    sampler = Sampler().start()
    t_run = time.perf_counter()
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
                    "reply_sha": hashlib.sha256(reply.text.encode("utf-8")).hexdigest()[:16],
                    "reply": reply.text,
                }
            )
            if ps is None:
                ps = ollama_get(args.ollama, "/api/ps")
                load_log = server_log_lines(log_path, log_offset)
            if n % 20 == 0 or n == len(items):
                print(f"[{n}/{len(items)}] last={wall:.2f}s", file=sys.stderr, flush=True)
    finally:
        during = sampler.stop()
    run_s = time.perf_counter() - t_run
    unload(args.ollama, args.model)

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
    if args.reference:
        ref_rows = {
            json.loads(line)["item_id"]: json.loads(line)["reply"]
            for line in (Path(args.reference) / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        same = sum(1 for r in rows if ref_rows.get(r["item_id"]) == r["reply"])
        ref = {"run": Path(args.reference).name, "identical_replies": same, "n": len(rows)}

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
    brief = {k: report[k] for k in ("model", "peak_over_baseline_mib", "exact", "reference")}
    brief |= {"summary": report["summary"], "size_vram": (report["api_ps"] or {}).get("size_vram")}
    print(json.dumps(brief, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
