import hashlib
import subprocess

import httpx
import pytest

from call_summary import evaluate
from call_summary.dataset import write_jsonl
from call_summary.evaluate import model_fingerprint
from call_summary.providers import HFProvider, OllamaProvider
from tests.test_scoring import _item


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
