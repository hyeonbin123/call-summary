# [테스트 묶음] 효율 추가 측정 도구 call_summary.bench의 GPU·Ollama 없이 도는 부분을 검증한다: 백분위 계산,
# 요청별 기록의 요약(첫 요청 따로), 서버 로그에서 필요한 줄 고르기, nvidia-smi 한 줄 읽기, 유휴 판단 규칙,
# OllamaProvider가 Ollama의 시간 카운터를 Reply에 남기는지, 그리고 psutil이 없을 때의 동작(기본은 바로 멈춤,
# --allow-no-cpu면 쉬는 상태를 기다리지 않고 보고서에 cpu_unavailable을 남김).
# [왜 중요한가] 이 측정은 아무것도 고르지 않는 기술 측정이지만 문서 표(VRAM, 생성 속도, 지연 p50/p95, 첫
# 요청)의 숫자가 이 계산에서 나온다(docs/experiments.md "효율 추가 측정"). 계산 규칙(첫 요청 제외, 백분위
# 방식)이 바뀌면 이전 실행과 비교할 수 없게 된다. 쉬는 상태 규칙의 CPU 조건은 psutil로만 잴 수 있어, psutil이
# 없을 때 조용히 넘어가면 --wait-idle(기본 60분)을 헛되이 다 기다린 뒤 CPU 없이 재게 된다.

import json
import sys
import types
from pathlib import Path

# httpx는 OllamaProvider가 Ollama에 요청할 때 쓰는 함수(httpx.post)를 monkeypatch로 가짜 응답으로 바꾸려고
# 가져온다. sys·types는 psutil이 없거나 있는 상황을 sys.modules로 흉내 내는 데 쓴다.
import httpx
import pytest

from call_summary import bench
from call_summary.bench import is_idle, parse_smi_line, pctl, server_log_lines, summarize_rows
from call_summary.providers import OllamaProvider

# 저장소 맨 위 폴더. bench.main을 끝까지 돌리는 시험이 커밋된 datasets/dev.jsonl의 첫 건을 쓴다.
ROOT = Path(__file__).resolve().parents[1]


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
    # 모델이 올라간 뒤의 요청 10개: 각각 2초에 200토큰. 느린 첫 요청은 세지 않는다
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
    assert not is_idle(sample(20.0, 35.0, 1000))  # CPU가 바쁨
    assert not is_idle(sample(10.0, 90.0, 1000))  # 다른 프로그램이 GPU를 쓰는 중
    assert not is_idle(sample(10.0, 35.0, 3000))  # 다른 프로세스가 GPU 메모리를 잡고 있음
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


# ---- psutil이 없을 때 (시스템 CPU를 잴 수 없을 때) ----
# sys.modules["psutil"]에 None을 넣으면 `import psutil`이 ImportError(ModuleNotFoundError)를 내므로, psutil이
# 설치된 환경에서도 "없는" 상황을 흉내 낼 수 있다. 반대로 가짜 모듈을 넣으면 psutil이 없는 CI에서도 "있는"
# 상황이 된다. monkeypatch.setitem이라 시험이 끝나면 원래대로 돌아간다.


# 불리면 그 자리에서 시험을 실패시키는 가짜. 데이터 읽기·Ollama 호출·쉬는 상태 확인·표본·잠자기가 일어나지
# 않았음을 확인하는 데 쓴다.
def _must_not_be_called(what):
    def fail(*args, **kwargs):
        raise AssertionError(f"{what} must not be called here")

    return fail


# 시스템 CPU 사용률을 늘 5%(쉬는 상태 기준 15% 미만)로 돌려주는 가짜 psutil 모듈.
def _fake_psutil():
    def cpu_percent(interval=None):
        return 5.0

    mod = types.ModuleType("psutil")
    mod.cpu_percent = cpu_percent
    return mod


# 기본 경로: psutil을 import할 수 없으면 데이터를 읽거나 Ollama에 묻거나 쉬는 상태를 기다리기 전에 이유와 함께
# 바로 멈춘다(argparse 오류, 종료 코드 2). 고치기 전에는 CPU 표본이 없어 is_idle이 늘 False라
# --wait-idle(기본 60분)을 끝까지 기다린 뒤 CPU 없이 쟀다.
def test_main_stops_at_once_without_psutil(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "psutil", None)
    for name in ("load_items", "ollama_get", "wait_idle", "sample_for", "Sampler", "unload"):
        monkeypatch.setattr(bench, name, _must_not_be_called(name))
    with pytest.raises(SystemExit) as exc:
        bench.main(["--data", "unused.jsonl", "--model", "m", "--out", "unused.json"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "psutil" in err and "--allow-no-cpu" in err


# --allow-no-cpu로 연 경우의 wait_idle: 규칙의 CPU 조건을 판정할 수 없으므로 1분 표본도 재지 않고 바로
# idle None(판정 못 함)과 이유를 돌려준다. 표본을 재거나 잠들면 가짜가 시험을 실패시킨다.
def test_wait_idle_without_psutil_returns_at_once(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(bench, "sample_for", _must_not_be_called("sample_for"))
    monkeypatch.setattr(bench.time, "sleep", _must_not_be_called("time.sleep"))
    got = bench.wait_idle(60.0)
    assert got["idle"] is None and got["checks"] == 0 and got["sample"] is None
    assert "psutil" in got["reason"]


# psutil이 있으면 wait_idle은 전처럼 1분 표본을 재서 쉬는 상태를 판정한다(여기서는 첫 표본이 쉬는 상태라 바로
# 끝난다).
def test_wait_idle_with_psutil_samples_as_before(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil())
    seconds = []
    sample = {"cpu_pct": {"mean": 5.0}, "gpu_util_pct": {"mean": 10.0}, "gpu_mem_used_mib": {"max": 1000}}
    monkeypatch.setattr(bench, "sample_for", lambda s: seconds.append(s) or sample)
    monkeypatch.setattr(bench.time, "sleep", _must_not_be_called("time.sleep"))
    got = bench.wait_idle(60.0)
    assert got["idle"] is True and got["checks"] == 1 and got["sample"] == sample
    assert "reason" not in got and seconds == [60.0]


# bench.main을 GPU·nvidia-smi·Ollama 없이 끝까지 돌린다: datasets/dev.jsonl의 첫 건 하나, 가짜
# /api/ps(올라간 모델 없음)·/api/version, 가짜 /api/chat 응답, 가짜 표본기. sample_for가 받은 초(쉬는 상태
# 확인 60, 기준값 5)를 순서대로 남기고, 쓴 보고서(JSON)와 그 목록을 돌려준다. cpu는 가짜 표본의 cpu_pct
# 값이다.
def _run_main(monkeypatch, tmp_path, extra_args, cpu):
    seconds = []
    sample = {
        "samples": 2,
        "gpu_mem_used_mib": {"min": 900, "mean": 950.0, "max": 1000},
        "gpu_util_pct": {"min": 0, "mean": 5.0, "max": 10},
        "cpu_pct": cpu,
    }

    class FakeSampler:
        def start(self):
            return self

        def stop(self):
            return {"samples": 0, "gpu_mem_used_mib": None, "gpu_util_pct": None, "cpu_pct": None}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}, "prompt_eval_count": 900, "eval_count": 2}

    def fake_get(base, path):
        return {"version": "0.35.1"} if path == "/api/version" else {"models": []}

    monkeypatch.setattr(bench, "sample_for", lambda s: seconds.append(s) or sample)
    monkeypatch.setattr(bench, "Sampler", FakeSampler)
    monkeypatch.setattr(bench, "ollama_get", fake_get)
    monkeypatch.setattr(bench, "unload", lambda base, model: None)
    monkeypatch.setattr(bench, "_git", lambda *args: "")
    monkeypatch.setattr(httpx, "post", lambda url, **kw: Resp())
    monkeypatch.setattr(bench.time, "sleep", _must_not_be_called("time.sleep"))
    out = tmp_path / "bench.json"
    argv = ["--data", str(ROOT / "datasets" / "dev.jsonl"), "--limit", "1", "--model", "m", "--out", str(out)]
    argv += ["--server-log", str(tmp_path / "missing.log"), *extra_args]
    assert bench.main(argv) == 0
    return json.loads(out.read_text(encoding="utf-8")), seconds


# psutil이 있으면 보고서에 cpu_unavailable false가 남고, 쉬는 상태 확인(60초 표본)과 기준값(5초 표본)을 전처럼
# 잰다.
def test_main_records_cpu_available_with_psutil(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil())
    report, seconds = _run_main(monkeypatch, tmp_path, [], cpu={"min": 4.0, "mean": 5.0, "max": 6.0})
    assert report["cpu_unavailable"] is False
    assert report["idle_check"]["idle"] is True and report["idle_check"]["checks"] == 1
    assert seconds == [60.0, 5] and report["n_items"] == 1


# --allow-no-cpu: psutil 없이도 재되, 쉬는 상태는 기다리지 않고(60초 표본 없음, 기준값 5초 표본만) 보고서에
# cpu_unavailable true와 판정 못 한 이유(idle None)를 남긴다.
def test_main_allow_no_cpu_measures_without_waiting(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "psutil", None)
    report, seconds = _run_main(monkeypatch, tmp_path, ["--allow-no-cpu"], cpu=None)
    assert report["cpu_unavailable"] is True
    assert report["idle_check"]["idle"] is None and "psutil" in report["idle_check"]["reason"]
    assert seconds == [5] and report["n_items"] == 1 and report["baseline"]["cpu_pct"] is None
