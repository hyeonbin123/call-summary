import httpx

from call_summary.bench import is_idle, parse_smi_line, pctl, server_log_lines, summarize_rows
from call_summary.providers import OllamaProvider


def _row(wall, prompt=900, gen=200, gen_ns=2_000_000_000, prompt_ns=300_000_000, load_ns=0):
    return {
        "wall_s": wall,
        "prompt_eval_count": prompt,
        "eval_count": gen,
        "prompt_eval_duration_ns": prompt_ns,
        "eval_duration_ns": gen_ns,
        "load_duration_ns": load_ns,
    }


def test_pctl_matches_evaluate_convention():
    v = sorted(float(i) for i in range(1, 241))
    assert pctl(v, 0.5) == v[120] and pctl(v, 0.95) == v[int(0.95 * 239)]


def test_summary_reports_the_loading_request_on_its_own():
    first = _row(105.0, gen=190, gen_ns=10_000_000_000, prompt_ns=40_000_000_000, load_ns=62_000_000_000)
    rows = [first] + [_row(2.0 + i / 100) for i in range(10)]
    s = summarize_rows(rows)
    assert s["first_request"] == {"wall_s": 105.0, "load_s": 62.0, "prompt_eval_s": 40.0}
    assert s["latency_s"]["n"] == 10 and s["latency_s"]["max"] == 2.09
    # the 10 warm requests: 200 tokens in 2 s each; the slow first request does not count
    assert s["gen_tok_s"] == 100.0 and s["gen_tok_s_per_request_p50"] == 100.0
    assert s["prompt_eval_ms_p50"] == 300.0 and s["completion_tokens"]["mean"] == 200.0
    assert s["missing_counters"] == 0


def test_summary_counts_missing_counters():
    s = summarize_rows([_row(3.0), _row(2.0), _row(2.0, gen=None, gen_ns=None)])
    assert s["missing_counters"] == 1 and s["gen_tok_s"] == 100.0


def test_server_log_lines(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("load_tensors: offloaded 1/1 layers to GPU\n", encoding="utf-8")
    offset = log.stat().st_size
    with log.open("a", encoding="utf-8") as f:
        f.write("[GIN] 200 /api/chat\nload_tensors: offloaded 37/37 layers to GPU\n")
        f.write("llama_kv_cache:      CUDA0 KV buffer size =   576.00 MiB\n")
    got = server_log_lines(log, offset)
    assert got == [
        "load_tensors: offloaded 37/37 layers to GPU",
        "llama_kv_cache:      CUDA0 KV buffer size =   576.00 MiB",
    ]
    assert server_log_lines(tmp_path / "missing.log", 0) == []


def test_parse_smi_line():
    assert parse_smi_line("1005, 36\n") == (1005, 36)
    assert parse_smi_line("[N/A], 3") is None and parse_smi_line("") is None


def test_idle_rule():
    def sample(cpu, util, mem):
        return {"cpu_pct": {"mean": cpu}, "gpu_util_pct": {"mean": util}, "gpu_mem_used_mib": {"max": mem}}

    assert is_idle(sample(10.0, 35.0, 1000))
    assert not is_idle(sample(20.0, 35.0, 1000))  # busy CPU
    assert not is_idle(sample(10.0, 90.0, 1000))  # another GPU job or a game
    assert not is_idle(sample(10.0, 35.0, 3000))  # memory held by other processes
    assert not is_idle({"cpu_pct": None, "gpu_util_pct": None, "gpu_mem_used_mib": None})


def test_ollama_reply_keeps_server_timings(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {"content": "{}"},
                "prompt_eval_count": 886,
                "eval_count": 198,
                "eval_duration": 2_000_000_000,
                "load_duration": 5,
            }

    monkeypatch.setattr(httpx, "post", lambda url, **kw: Resp())
    r = OllamaProvider(model="m").generate([{"role": "user", "content": "x"}])
    assert (r.prompt_tokens, r.completion_tokens) == (886, 198)
    assert r.timings["eval_duration"] == 2_000_000_000 and r.timings["prompt_eval_duration"] is None
