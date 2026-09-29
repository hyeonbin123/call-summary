import hashlib

import httpx
import pytest

from call_summary.evaluate import model_fingerprint
from call_summary.providers import HFProvider, OllamaProvider


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


def test_fingerprint_of_an_ollama_model_is_its_digest(monkeypatch):
    class Resp:
        def json(self):
            return {"models": [{"name": "m:q8_0", "model": "m:q8_0", "digest": "abc"}]}

    monkeypatch.setattr(httpx, "get", lambda url, **kw: Resp())
    assert model_fingerprint(OllamaProvider(model="m:q8_0")) == {"ollama_digest": "abc"}
    assert model_fingerprint(OllamaProvider(model="other")) == {"ollama_digest": None}

    def refused(url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refused)
    assert model_fingerprint(OllamaProvider(model="m:q8_0")) == {"ollama_digest": None}
