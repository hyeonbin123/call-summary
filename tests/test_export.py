import json
import subprocess

import pytest

from call_summary import export
from call_summary.evaluate import adapter_hash


@pytest.fixture
def fake_tools(monkeypatch):
    """merge writes merged/config.json, the GGUF converter writes --outfile, ollama and git succeed."""
    merges = []

    def fake_merge(base, adapter, out):
        merges.append(adapter)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")

    def fake_run(cmd, **kw):
        if any(str(c).endswith("convert_hf_to_gguf.py") for c in cmd):
            open(cmd[cmd.index("--outfile") + 1], "wb").close()
        return subprocess.CompletedProcess(cmd, 0, stdout="x")

    monkeypatch.setattr(export, "merge", fake_merge)
    monkeypatch.setattr(export.subprocess, "run", fake_run)
    return merges


def _adapter(tmp_path, name, weights):
    d = tmp_path / name
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(weights)
    return str(d)


def _export(adapter, out):
    return export.main(["--base", "Qwen/Qwen3-4B", "--adapter", adapter, "--name", "cs", "--out", str(out)])


def test_export_reuses_its_own_merge(tmp_path, fake_tools):
    a = _adapter(tmp_path, "a", b"A")
    out = tmp_path / "export"
    assert _export(a, out) == 0 and _export(a, out) == 0
    assert fake_tools == [a]
    assert json.loads((out / "export.json").read_text(encoding="utf-8"))["adapter_sha"] == adapter_hash(a)


def test_export_refuses_weights_merged_from_another_adapter(tmp_path, fake_tools):
    out = tmp_path / "export"
    _export(_adapter(tmp_path, "a", b"A"), out)
    with pytest.raises(SystemExit):
        _export(_adapter(tmp_path, "b", b"B"), out)
    assert len(fake_tools) == 1
