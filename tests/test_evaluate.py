import argparse
import hashlib
import json
import subprocess

import httpx
import pytest

from call_summary import evaluate
from call_summary.dataset import read_jsonl, write_jsonl
from call_summary.evaluate import model_fingerprint
from call_summary.prompts import record_to_json
from call_summary.providers import HFProvider, OllamaProvider, Reply
from tests.test_scoring import _item

MSGS = [{"role": "user", "content": "x"}]


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


@pytest.mark.parametrize("failure", [FileNotFoundError("git"), subprocess.CalledProcessError(128, ["git"])])
def test_official_run_stops_when_git_cannot_check_the_tree(tmp_path, monkeypatch, failure):
    # No git, not a repository or "dubious ownership": the clean-tree guard must not pass silently.
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


def test_fingerprint_names_the_cached_base_snapshot(monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(hub, "try_to_load_from_cache", lambda repo, name: f"/hub/snapshots/abc123/{name}")
    assert model_fingerprint(HFProvider(model_id="Qwen/Qwen3-4B")) == {"base_revision": "abc123"}


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


@pytest.mark.parametrize("key", ["temperature", "seed", "num_ctx", "num_predict", "num_gpu"])
def test_extra_options_cannot_replace_the_fixed_decoding_settings(key):
    with pytest.raises(ValueError):
        OllamaProvider(model="m", extra_options={key: 1})


def test_option_values_are_typed():
    assert evaluate.parse_option("presence_penalty=0") == ("presence_penalty", 0)
    assert evaluate.parse_option("repeat_penalty=1.05") == ("repeat_penalty", 1.05)
    assert evaluate.parse_option("penalize_newline=false") == ("penalize_newline", False)
    with pytest.raises(argparse.ArgumentTypeError):
        evaluate.parse_option("presence_penalty")


def test_run_records_context_options_and_run_checks(tmp_path, monkeypatch):
    # A reply cut at num_predict, a reply with thinking text and one that filled the context are counted,
    # so a run whose thinking switch or context size did not hold shows it in summary.json.
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


def test_shots_must_exist_for_every_domain_in_the_data(tmp_path, monkeypatch):
    # test-c (card) has no examples in train: asking for shots must stop, not run 0-shot under a 2-shot label.
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
