import json
import subprocess

import pytest

from call_summary import export
from call_summary.evaluate import adapter_hash


@pytest.fixture
def fake_tools(monkeypatch):
    """merge writes merged/config.json, the GGUF converter writes --outfile, ollama and git succeed."""

    class Merges(list):
        pass

    merges = Merges()
    calls = []

    def fake_merge(base, adapter, out):
        merges.append(adapter)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")

    def fake_run(cmd, **kw):
        calls.append([str(c) for c in cmd])
        if any(str(c).endswith("convert_hf_to_gguf.py") for c in cmd):
            open(cmd[cmd.index("--outfile") + 1], "wb").close()
        if str(cmd[0]).endswith("llama-quantize.exe"):
            open(cmd[2], "wb").close()
        return subprocess.CompletedProcess(cmd, 0, stdout="x")

    monkeypatch.setattr(export, "merge", fake_merge)
    monkeypatch.setattr(export.subprocess, "run", fake_run)
    merges.calls = calls  # type: ignore[attr-defined]
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


def test_an_interrupted_build_is_not_taken_for_the_older_stamp(tmp_path, fake_tools, monkeypatch):
    a = _adapter(tmp_path, "a", b"A")
    out = tmp_path / "export"
    _export(a, out)
    # As the refusal message says: delete merged/ and the F16 GGUF (source.json stays behind) ...
    for f in (out / "merged").iterdir():
        f.unlink()
    (out / "merged").rmdir()
    (out / "model-f16.gguf").unlink()

    def crashing_merge(base, adapter, target):  # ... then adapter b's merge dies half-way
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("out of memory")

    monkeypatch.setattr(export, "merge", crashing_merge)
    with pytest.raises(RuntimeError):
        _export(_adapter(tmp_path, "b", b"B"), out)
    with pytest.raises(SystemExit):  # b's half-merged weights must not pass for a's
        _export(a, out)


def test_export_refuses_weights_merged_from_another_adapter(tmp_path, fake_tools):
    out = tmp_path / "export"
    _export(_adapter(tmp_path, "a", b"A"), out)
    with pytest.raises(SystemExit):
        _export(_adapter(tmp_path, "b", b"B"), out)
    assert len(fake_tools) == 1


def test_quantized_models_come_from_llama_quantize_not_ollama(tmp_path, fake_tools):
    out = tmp_path / "export"
    _export(_adapter(tmp_path, "a", b"A"), out)
    creates = [c for c in fake_tools.calls if c[:2] == ["ollama", "create"]]
    assert [c[2] for c in creates] == ["cs:f16", "cs:q8_0", "cs:q4_k_m"]
    assert all("--quantize" not in c for c in creates)
    quantize = [c for c in fake_tools.calls if c[0].endswith("llama-quantize.exe")]
    assert [c[3] for c in quantize] == ["Q8_0", "Q4_K_M"]
    assert "model-q4_k_m.gguf" in (out / "Modelfile.q4_K_M").read_text(encoding="utf-8")
    info = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert set(info["gguf"]) == {"f16", "q8_0", "q4_K_M"}
