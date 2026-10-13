# [테스트 묶음] 효율 추가 측정 도구 call_summary.bench의 GPU·Ollama 없이 도는 부분을 검증한다: 백분위 계산,
# 요청별 기록의 요약(첫 요청 따로), 서버 로그에서 필요한 줄 고르기, nvidia-smi 한 줄 읽기, 유휴 판단 규칙,
# 그리고 OllamaProvider가 Ollama의 시간 카운터를 Reply에 남기는지.
# [왜 중요한가] 이 측정은 아무것도 고르지 않는 기술 측정이지만 문서 표(VRAM, 생성 속도, 지연 p50/p95, 첫
# 요청)의 숫자가 이 계산에서 나온다(docs/experiments.md "효율 추가 측정"). 계산 규칙(첫 요청 제외, 백분위
# 방식)이 바뀌면 이전 실행과 비교할 수 없게 된다.

import httpx

# httpx는 OllamaProvider가 Ollama에 요청할 때 쓰는 함수(httpx.post)를 monkeypatch로 가짜 응답으로 바꾸려고
# 가져온다.
from call_summary.bench import is_idle, parse_smi_line, pctl, server_log_lines, summarize_rows
from call_summary.providers import OllamaProvider


# 요청 하나의 기록을 흉내 낸다: 클라이언트 왕복 시간과 Ollama 카운터(프롬프트·생성 토큰 수, 각 처리 시간 ns,
# 모델 적재 시간). 기본값은 생성 200토큰을 2초에 = 100 tok/s.
def _row(wall, prompt=900, gen=200, gen_ns=2_000_000_000, prompt_ns=300_000_000, load_ns=0):
    return {
        "wall_s": wall,
        "prompt_eval_count": prompt,
        "eval_count": gen,
        "prompt_eval_duration_ns": prompt_ns,
        "eval_duration_ns": gen_ns,
        "load_duration_ns": load_ns,
    }


# 백분위가 evaluate.py와 같은 방식(p50 = v[n // 2], p95 = v[int(0.95·(n−1))], 보간 없음)인지 확인한다. 규칙이
# "evaluate와 같은 백분위 계산"으로 정했고, 같아야 09-30 실행 기록의 지연과 나란히 적을 수 있다.
def test_pctl_matches_evaluate_convention():
    v = sorted(float(i) for i in range(1, 241))
    assert pctl(v, 0.5) == v[120] and pctl(v, 0.95) == v[int(0.95 * 239)]


# 첫 요청은 모델 적재와 GPU 준비가 들어가 따로 적고 나머지 지표에서 뺀다(규칙: 시험 실행에서 첫 요청의
# 프롬프트 처리에 40초가 걸렸다). 느린 첫 요청이 지연 p50, 생성 속도, 요청별 생성 속도 p50, 프롬프트 처리 시간
# p50, 평균 생성 토큰에 섞이지 않는지 본다.
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


# Ollama가 카운터를 주지 않은 요청은 missing_counters로 세고, 생성 속도 계산에서는 빠져 나머지 값만으로
# 계산된다.
def test_summary_counts_missing_counters():
    s = summarize_rows([_row(3.0), _row(2.0), _row(2.0, gen=None, gen_ns=None)])
    assert s["missing_counters"] == 1 and s["gen_tok_s"] == 100.0


# 서버 로그에서 측정 시작 뒤(바이트 위치 offset 이후)의 줄 중 층 배치·버퍼 크기 줄만 고르는지 확인한다. 측정
# 전에 이미 있던 줄과 요청 로그([GIN])는 빼고, 로그 파일이 없으면 빈 목록이다. 문서의 "가중치 / KV 캐시 / 계산
# 버퍼" 열이 이 줄에서 나온다.
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


# nvidia-smi 반복 출력 한 줄("사용 MiB, 사용률 %")을 읽고, [N/A] 같은 값이나 빈 줄은 None으로 버린다.
def test_parse_smi_line():
    assert parse_smi_line("1005, 36\n") == (1005, 36)
    assert parse_smi_line("[N/A], 3") is None and parse_smi_line("") is None


# 유휴 기준(규칙: 1분 평균 CPU 15% 미만, GPU 사용률 평균 50% 미만, 다른 프로그램의 GPU 메모리 2.5 GB 미만)을
# 하나씩 어겨 본다. 배경화면 프로그램 수준(사용률 35%, 약 1 GB)은 유휴로 치고, 표본이 없으면 유휴가 아니다.
def test_idle_rule():
    def sample(cpu, util, mem):
        return {"cpu_pct": {"mean": cpu}, "gpu_util_pct": {"mean": util}, "gpu_mem_used_mib": {"max": mem}}

    assert is_idle(sample(10.0, 35.0, 1000))
    assert not is_idle(sample(20.0, 35.0, 1000))  # busy CPU
    assert not is_idle(sample(10.0, 90.0, 1000))  # another GPU job or a game
    assert not is_idle(sample(10.0, 35.0, 3000))  # memory held by other processes
    assert not is_idle({"cpu_pct": None, "gpu_util_pct": None, "gpu_mem_used_mib": None})


# OllamaProvider.generate가 Ollama 응답의 토큰 수와 시간 카운터를 Reply.timings에 그대로 남기는지
# 확인한다(커밋 0ee468b). 응답에 없는 카운터는 0이 아니라 None으로 남아 missing으로 셀 수 있다.
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
