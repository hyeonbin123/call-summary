# [테스트 묶음] 평가 실행기 call_summary.evaluate와 OllamaProvider의 "측정 조건을 지키는 장치"를 모델 없이
# 검증한다: 공식 실행의 깨끗한 트리 검사, 모델 지문(어댑터 해시, 바탕 스냅샷, Ollama digest·버전), Ollama
# 요청에 넣는 옵션과 생각 끄기, 고정 디코딩 설정 보호, --option 값 해석, run_checks(잘린 답·생각 글·컨텍스트
# 채움) 기록, 기본값 유지, 업종별 예시 부족 시 멈춤.
# [왜 중요한가] 이 프로젝트는 규칙을 먼저 커밋하고 그 코드로 잰다. 실행 기록(manifest)이 어떤
# 가중치·옵션·커밋으로 돌았는지 정확히 남겨야 결과를 믿을 수 있고, 조건이 어긋난 실행(생각 모드가 켜짐, 예시
# 없이 2-shot 이름으로 돎)은 조용히 지나가면 안 된다.

import argparse
import hashlib
import json
import subprocess

# httpx.post·httpx.get을 monkeypatch로 가짜로 바꿔 Ollama 서버 없이 요청 본문과 응답 처리를 본다. _item은
# tests/test_scoring.py의 가짜 건이다.
import httpx
import pytest

from call_summary import evaluate, providers
from call_summary.dataset import read_jsonl, write_jsonl
from call_summary.evaluate import model_fingerprint
from call_summary.prompts import record_to_json
from call_summary.providers import HFProvider, OllamaProvider, Reply
from tests.test_scoring import _item

# Ollama 요청에 넣을 최소 메시지.
MSGS = [{"role": "user", "content": "x"}]


# httpx.post 대용: 보낸 JSON 본문을 sent에 모으고 정해 둔 payload를 응답으로 준다.
def _fake_post(sent: list, payload: dict):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    def post(url, json=None, **kw):
        sent.append(json)
        return Resp()

    return post


# --official은 깨끗한 작업 트리에서만 돈다. 예전에는 git을 부르지 못하면(git 없음, 저장소 아님, "dubious
# ownership" 오류) 출력이 비어 깨끗한 트리로 통과했다(커밋 6d9defc). 이제는 git 실패 두 종류 모두에서 argparse
# 오류(종료 코드 2)로 멈추고, 모델을 만들기 전에 멈추는지도 본다. 깨끗한 트리 보호가 조용히 통과하면 안 된다.
@pytest.mark.parametrize("failure", [FileNotFoundError("git"), subprocess.CalledProcessError(128, ["git"])])
def test_official_run_stops_when_git_cannot_check_the_tree(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [_item(0).to_dict()])

    def git_fails(*a, **kw):
        raise failure

    def no_model(*a, **kw):
        raise AssertionError("the run got past the clean-tree check")

    monkeypatch.setattr(evaluate.subprocess, "run", git_fails)
    monkeypatch.setattr(evaluate, "OllamaProvider", no_model)
    with pytest.raises(SystemExit) as stop:
        evaluate.main(["--data", str(data), "--model", "m", "--official"])
    assert stop.value.code == 2


# 어댑터 경로는 다시 학습해도 같을 수 있으므로 지문은 파일 바이트의 해시다(커밋 f89df9f). 바이트가 바뀌면
# adapter_sha도 바뀌고, 옆의 train_config.json 해시도 남는다. 캐시에 없는 바탕 모델은 base_revision이 None이고
# 실행은 멈추지 않는다.
def test_fingerprint_of_an_adapter_follows_its_bytes(tmp_path):
    final = tmp_path / "run" / "final"
    final.mkdir(parents=True)
    (final / "adapter_model.safetensors").write_bytes(b"A")
    (tmp_path / "run" / "train_config.json").write_text("{}", encoding="utf-8")
    p = HFProvider(model_id="not/a-cached-model", adapter=str(final), load_4bit=True)
    fp = model_fingerprint(p)
    assert fp["adapter_sha"] == hashlib.sha256(b"A").hexdigest()[:12]
    assert fp["train_config_sha"] and fp["base_revision"] is None
    (final / "adapter_model.safetensors").write_bytes(b"B")
    assert model_fingerprint(p)["adapter_sha"] != fp["adapter_sha"]


# 바탕 모델의 스냅샷 커밋은 캐시된 config.json 경로의 부모 폴더 이름에서 읽는다(snapshot_download는 일부만
# 받은 스냅샷을 거부해 쓰지 않음). huggingface_hub의 함수를 가짜로 바꿔 "abc123"이 나오는지 본다.
def test_fingerprint_names_the_cached_base_snapshot(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda repo, name: f"/hub/snapshots/abc123/{name}")
    assert model_fingerprint(HFProvider(model_id="Qwen/Qwen3-4B")) == {"base_revision": "abc123"}


# Ollama 모델의 지문은 /api/tags의 digest와 /api/version의 서버 버전이다(같은 가중치라도 실행기가 바뀌면
# 조건이 다르다). 목록에 없는 태그는 digest None, 서버가 꺼져 있으면 둘 다 None이고 지문 때문에 실행이 멈추지
# 않는다.
def test_fingerprint_of_an_ollama_model_is_its_digest_and_server_version(monkeypatch):
    class Resp:
        def __init__(self, url):
            self.url = url

        def json(self):
            if self.url.endswith("/api/version"):
                return {"version": "0.35.1"}
            return {"models": [{"name": "m:q8_0", "model": "m:q8_0", "digest": "abc"}]}

    monkeypatch.setattr(httpx, "get", lambda url, **kw: Resp(url))
    assert model_fingerprint(OllamaProvider(model="m:q8_0")) == {
        "ollama_digest": "abc",
        "ollama_version": "0.35.1",
    }
    assert model_fingerprint(OllamaProvider(model="other"))["ollama_digest"] is None

    def refused(url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refused)
    down = model_fingerprint(OllamaProvider(model="m:q8_0"))
    assert down == {"ollama_digest": None, "ollama_version": None}


# Ollama 지문 조회는 서비스(/health·/ready·시작 로그)와 같은 providers.ollama_fingerprint다. 같은 가짜
# 서버에서 evaluate의 manifest 값(model_fingerprint)과 공용 함수의 값이 같고 키가 ollama_digest·ollama_version
# 둘뿐이어야 기록된 manifest와 대조가 깨지지 않는다. 요청 시간 한도는 전처럼 10초이고 바꿀 수 있다. 낮은 수준
# 함수 ollama_model_digest는 서버에 닿지 못하면 예외를 내고(서비스가 no_model_server를 가르는 데 쓴다) 태그가
# 없으면 None이다. 공용 함수는 둘 다 None으로 삼킨다.
def test_ollama_fingerprint_is_shared_between_evaluate_and_the_service(monkeypatch):
    seen: list[tuple[str, float]] = []

    def fake_get(url, timeout=None, **kw):
        seen.append((url, timeout))
        req = httpx.Request("GET", url)
        if url.endswith("/api/version"):
            return httpx.Response(200, json={"version": "0.35.1"}, request=req)
        return httpx.Response(200, json={"models": [{"name": "m:q8_0", "digest": "abc"}]}, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    shared = providers.ollama_fingerprint("http://h:1", "m:q8_0")
    assert shared == model_fingerprint(OllamaProvider(model="m:q8_0", base_url="http://h:1"))
    assert list(shared) == ["ollama_digest", "ollama_version"] and shared["ollama_digest"] == "abc"
    assert seen[:2] == [("http://h:1/api/tags", 10), ("http://h:1/api/version", 10)]
    assert providers.ollama_model_digest("http://h:1", "other") is None
    assert providers.ollama_fingerprint("http://h:1", "m:q8_0", timeout_s=2.0) == shared
    assert seen[-1] == ("http://h:1/api/version", 2.0)

    def refused(url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refused)
    with pytest.raises(httpx.ConnectError):
        providers.ollama_model_digest("http://h:1", "m:q8_0")
    assert providers.ollama_fingerprint("http://h:1", "m:q8_0") == {
        "ollama_digest": None,
        "ollama_version": None,
    }


# keep_alive 값 해석: Ollama가 받는 꼴은 Go 시간 문자열("30m", "1h30m", "1.5h", "-1m", "500ms")과 초 단위
# 숫자다. 정수 글은 숫자로 바꿔 보낸다(문자열 "-1"은 Ollama의 시간 해석이 거절하지만 숫자 -1은 계속 상주, 0은
# 바로 내림). 앞뒤 공백은 지운다.
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30m", "30m"),
        ("1h30m", "1h30m"),
        ("1.5h", "1.5h"),
        ("-1m", "-1m"),
        ("500ms", "500ms"),
        ("-1", -1),
        ("0", 0),
        ("600", 600),
        (" 30m ", "30m"),
    ],
)
def test_keep_alive_values_follow_the_ollama_rules(raw, expected):
    got = providers.parse_keep_alive(raw)
    assert got == expected and type(got) is type(expected)


# Ollama가 받지 않는 keep_alive 글(단위 없는 소수, 낱말, 빈 글, 단위 앞 공백, 모르는 단위)은 ValueError다.
@pytest.mark.parametrize("raw", ["forever", "1.5", "", "10 m", "-1x", "m", "1h 30m"])
def test_keep_alive_rejects_values_ollama_would_not_accept(raw):
    with pytest.raises(ValueError):
        providers.parse_keep_alive(raw)


# OllamaProvider의 keep_alive는 문자열(기본 "30m")이나 정수를 그대로 요청 본문에 넣는다: 정수는 JSON 숫자로
# 간다. 기본값의 요청 본문은 전과 같다(기록된 실행과 같은 "30m").
def test_ollama_request_sends_keep_alive_as_given(monkeypatch):
    sent: list = []
    monkeypatch.setattr(httpx, "post", _fake_post(sent, {"message": {"content": "{}"}}))
    OllamaProvider(model="m").generate(MSGS)
    assert sent[-1]["keep_alive"] == "30m"
    OllamaProvider(model="m", keep_alive=-1).generate(MSGS)
    assert sent[-1]["keep_alive"] == -1 and type(sent[-1]["keep_alive"]) is int


# Ollama 답의 생각 글(message.thinking)과 멈춘 이유(done_reason)를 Reply에 남기는지 확인한다. 생각 모드가
# 실제로 꺼졌는지, 답이 num_predict에서 잘렸는지를 실행 기록으로 보이기 위한 값이다(커밋 5741656).
def test_ollama_reply_keeps_thinking_text_and_stop_reason(monkeypatch):
    sent: list = []
    payload = {
        "message": {"content": "{}", "thinking": "생각 중"},
        "done_reason": "length",
        "eval_count": 1024,
    }
    monkeypatch.setattr(httpx, "post", _fake_post(sent, payload))
    r = OllamaProvider(model="m").generate(MSGS)
    assert r.thinking == "생각 중" and r.done_reason == "length"
    monkeypatch.setattr(
        httpx, "post", _fake_post(sent, {"message": {"content": "{}"}, "done_reason": "stop"})
    )
    r = OllamaProvider(model="m").generate(MSGS)
    assert r.thinking is None and r.done_reason == "stop"


# 요청 본문에 더한 옵션(presence_penalty)과 컨텍스트 길이가 들어가고 고정 디코딩(temperature 0, seed 0)은
# 그대로인지 본다. think는 기본 False로 보내고, None이면 키를 아예 보내지 않으며(모델 기본값), True면 켠다.
def test_ollama_request_carries_extra_options_and_the_think_setting(monkeypatch):
    sent: list = []
    monkeypatch.setattr(httpx, "post", _fake_post(sent, {"message": {"content": "{}"}}))
    OllamaProvider(model="m", num_ctx=8192, extra_options={"presence_penalty": 0.0}).generate(MSGS)
    opts = sent[-1]["options"]
    assert opts["presence_penalty"] == 0.0 and opts["num_ctx"] == 8192
    assert opts["temperature"] == 0.0 and opts["seed"] == 0 and sent[-1]["think"] is False
    OllamaProvider(model="m", think=None).generate(MSGS)
    assert "think" not in sent[-1]
    OllamaProvider(model="m", think=True).generate(MSGS)
    assert sent[-1]["think"] is True


# extra_options로 고정 디코딩 설정을 몰래 바꾸지 못하게 provider가 만들 때 ValueError로 거절한다. 이 값들은
# provider의 필드로만 바꾸고 manifest에 따로 남는다.
@pytest.mark.parametrize("key", ["temperature", "seed", "num_ctx", "num_predict", "num_gpu"])
def test_extra_options_cannot_replace_the_fixed_decoding_settings(key):
    with pytest.raises(ValueError):
        OllamaProvider(model="m", extra_options={key: 1})


# --option KEY=VALUE의 값이 정수·실수·불리언으로 바뀌어 Ollama에 문자열이 아닌 숫자로 가는지, '=' 없는 입력은
# argparse 오류인지 확인한다.
def test_option_values_are_typed():
    assert evaluate.parse_option("presence_penalty=0") == ("presence_penalty", 0)
    assert evaluate.parse_option("repeat_penalty=1.05") == ("repeat_penalty", 1.05)
    assert evaluate.parse_option("penalize_newline=false") == ("penalize_newline", False)
    with pytest.raises(argparse.ArgumentTypeError):
        evaluate.parse_option("presence_penalty")


# 가짜 OllamaProvider로 evaluate.main을 끝까지 돌려, 넘긴 --num-ctx·--option·think off가 provider와 manifest에
# 남는지, 건별 기록에 done_reason과 생각 글 길이가 남는지, summary.json의 run_checks가 잘린 답 1·생각 글
# 1·컨텍스트를 채운 답 1(프롬프트 7000 + 답 1192 = 8192)을 세는지 본다(2026 비교 대상 규칙: 이 수들을
# summary에 적는다). 생각 끄기나 컨텍스트 크기가 지켜지지 않은 실행이 summary.json에 드러나게 하려는 것이다.
def test_run_records_context_options_and_run_checks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    items = [_item(i) for i in range(3)]
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict() for it in items])
    made: dict = {}
    replies = iter(
        [
            Reply(record_to_json(items[0].gold()), 0.1, 900, 200, done_reason="stop"),
            Reply("{", 0.1, 900, 1024, done_reason="length"),
            Reply(record_to_json(items[2].gold()), 0.1, 7000, 1192, thinking="생각", done_reason="stop"),
        ]
    )

    class FakeOllama:
        def __init__(self, **kw):
            made.update(kw)
            self.name = "fake"

        def generate(self, messages, json_schema=None):
            return next(replies)

    monkeypatch.setattr(evaluate, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(evaluate, "model_fingerprint", lambda p: {})
    argv = ["--data", str(data), "--model", "m", "--num-ctx", "8192", "--label", "t"]
    evaluate.main(argv + ["--option", "presence_penalty=0", "--option", "repeat_penalty=1"])
    assert made["num_ctx"] == 8192 and made["think"] is False
    assert made["extra_options"] == {"presence_penalty": 0, "repeat_penalty": 1}
    run = next((tmp_path / "outputs" / "runs").iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_ctx"] == 8192 and manifest["think"] == "off"
    assert manifest["ollama_options"] == {"presence_penalty": 0, "repeat_penalty": 1}
    rows = list(read_jsonl(run / "items.jsonl"))
    assert [r["done_reason"] for r in rows] == ["stop", "length", "stop"]
    assert [r["thinking_chars"] for r in rows] == [0, 0, 2]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_checks"] == {"length_stops": 1, "thinking_items": 1, "context_full": 1}


# 새 옵션을 넣지 않은 기본 실행은 예전과 같은 조건(num_ctx 4096, 생각 끔, 더한 옵션 없음)이어야 한다. 그래야
# 옵션을 더하기 전에 잰 실행과 같은 조건으로 비교할 수 있다.
def test_default_run_keeps_the_earlier_ollama_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [_item(0).to_dict()])
    made: dict = {}

    class FakeOllama:
        def __init__(self, **kw):
            made.update(kw)
            self.name = "fake"

        def generate(self, messages, json_schema=None):
            return Reply("{", 0.1)

    monkeypatch.setattr(evaluate, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(evaluate, "model_fingerprint", lambda p: {})
    evaluate.main(["--data", str(data), "--model", "m"])
    assert made["num_ctx"] == 4096 and made["think"] is False and not made["extra_options"]


# 예시 풀(train)에 데이터의 업종 예시가 모자라면 실행을 시작하지 않고 멈춘다. 예전에는 test-c(카드, train에
# 없음)를 --shots 2로 돌리면 2-shot 이름을 단 채 조용히 0-shot으로 돌았다(커밋 5741656).
def test_shots_must_exist_for_every_domain_in_the_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pool = tmp_path / "train.jsonl"
    cats = ["주문 취소", "환불 문의", "주문 취소", "환불 문의"]
    write_jsonl(pool, [_item(i, category=c).to_dict() for i, c in enumerate(cats)])
    data = tmp_path / "test.jsonl"
    write_jsonl(data, [_item(0, domain="card", category=None).to_dict()])

    def no_model(*a, **kw):
        raise AssertionError("the run started without shots for every domain")

    monkeypatch.setattr(evaluate, "OllamaProvider", no_model)
    with pytest.raises(SystemExit) as stop:
        evaluate.main(["--data", str(data), "--model", "m", "--shots", "2", "--shot-pool", str(pool)])
    assert stop.value.code == 2
