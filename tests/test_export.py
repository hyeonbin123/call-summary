# [테스트 묶음] 5단계 내보내기 call_summary.export(어댑터 합치기 → GGUF f16 변환 → llama-quantize → ollama
# create)의 판단 로직을 모델·도구 없이 검증한다: 같은 어댑터면 합친 가중치를 다시 쓰고, 다른 어댑터의 가중치는
# 거절하며, 중간에 멈춘 빌드를 예전 것으로 착각하지 않고, 양자화는 llama-quantize로 하며, QLoRA 어댑터는 4bit
# 바탕을 풀어 합친다.
# [왜 중요한가] 합친 가중치와 f16 GGUF를 다시 쓰는 것은 시간을 아끼지만, 다른 어댑터의 가중치를 재사용하면
# 엉뚱한 모델을 서빙·측정하게 된다(커밋 ed76ea6: source.json 도장이 맞을 때만 재사용). QLoRA 어댑터를 fp16
# 바탕에 합치면 약 7%p 떨어진 것도 실제로 겪었다(docs/experiments.md 5단계 발견 2).

import json
import subprocess

import pytest

# adapter_hash: 어댑터 파일 바이트의 해시(evaluate의 모델 지문과 같은 함수). export.json에 남는 값과 비교한다.
from call_summary import export
from call_summary.evaluate import adapter_hash


# 진짜 합치기·변환·양자화·ollama 호출 대신 쓰는 가짜 도구(fixture). merge는 merged/config.json만 쓰고 어떤
# 어댑터로, 4bit 바탕으로 합쳤는지 기록한다. subprocess.run은 명령을 기록하고 변환기·llama-quantize가 만들
# 출력 파일만 빈 파일로 만든다. monkeypatch라 테스트가 끝나면 원래 함수로 돌아간다.
@pytest.fixture
def fake_tools(monkeypatch):
    """merge writes merged/config.json, the GGUF converter writes --outfile, ollama and git succeed."""

    class Merges(list):
        pass

    merges = Merges()
    calls = []

    def fake_merge(base, adapter, out, from_4bit=False):
        merges.append(adapter)
        merges.from_4bit = from_4bit  # type: ignore[attr-defined]
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


# weights 바이트로 된 가짜 어댑터 폴더. 바이트가 다르면 adapter_sha가 달라 다른 어댑터로 취급된다.
def _adapter(tmp_path, name, weights):
    d = tmp_path / name
    d.mkdir()
    (d / "adapter_model.safetensors").write_bytes(weights)
    return str(d)


# export.main을 한 번 부른다(Ollama 이름 cs, 기본 양자화 q8_0·q4_K_M).
def _export(adapter, out):
    return export.main(["--base", "Qwen/Qwen3-4B", "--adapter", adapter, "--name", "cs", "--out", str(out)])


# 같은 어댑터로 두 번 내보내면 합치기는 한 번만 일어나고(source.json 도장이 같아 재사용), export.json에 어댑터
# 해시가 남는다. 이 해시가 평가 manifest의 지문과 이어진다.
def test_export_reuses_its_own_merge(tmp_path, fake_tools):
    a = _adapter(tmp_path, "a", b"A")
    out = tmp_path / "export"
    assert _export(a, out) == 0 and _export(a, out) == 0
    assert fake_tools == [a]
    assert json.loads((out / "export.json").read_text(encoding="utf-8"))["adapter_sha"] == adapter_hash(a)


# 거절 메시지대로 merged/와 f16을 지워도 a의 source.json은 남는다. 그 상태에서 b의 합치기가 반쯤 쓰고 죽으면,
# 다음에 a로 내보낼 때 남은 a 도장 때문에 b의 반쯤 합친 가중치를 a의 것으로 재사용할 위험이 있다. 빌드를
# 시작할 때 예전 도장을 먼저 지우므로 a 내보내기가 거절(SystemExit)되는지 확인한다.
def test_an_interrupted_build_is_not_taken_for_the_older_stamp(tmp_path, fake_tools, monkeypatch):
    a = _adapter(tmp_path, "a", b"A")
    out = tmp_path / "export"
    _export(a, out)
    # As the refusal message says: delete merged/ and the F16 GGUF (source.json stays behind) ...
    for f in (out / "merged").iterdir():
        f.unlink()
    (out / "merged").rmdir()
    (out / "model-f16.gguf").unlink()

    def crashing_merge(base, adapter, target, from_4bit=False):  # ... then adapter b's merge dies half-way
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("out of memory")

    monkeypatch.setattr(export, "merge", crashing_merge)
    with pytest.raises(RuntimeError):
        _export(_adapter(tmp_path, "b", b"B"), out)
    with pytest.raises(SystemExit):  # b's half-merged weights must not pass for a's
        _export(a, out)


# 같은 --out에 다른 어댑터(b)로 내보내려 하면 a의 가중치를 쓰지 않고 거절하며, 합치기도 다시 일어나지 않는다.
def test_export_refuses_weights_merged_from_another_adapter(tmp_path, fake_tools):
    out = tmp_path / "export"
    _export(_adapter(tmp_path, "a", b"A"), out)
    with pytest.raises(SystemExit):
        _export(_adapter(tmp_path, "b", b"B"), out)
    assert len(fake_tools) == 1


# Ollama 0.34부터 GGUF를 가져올 때 양자화하지 못해(export.py 모듈 설명, 커밋 bd33a2a) q8_0·q4_K_M 파일은
# llama.cpp의 llama-quantize로 만든다. ollama create는 이미 양자화된 파일을 등록만 하고 --quantize를 쓰지
# 않는지, Modelfile이 그 양자화 파일을 가리키는지, export.json에 세 GGUF가 모두 적히는지 본다.
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


# 어댑터 옆 train_config.json이 qlora true면 학습 때처럼 바탕을 4bit로 불러 fp16으로 풀어낸 뒤 합친다(QLoRA
# 어댑터는 nf4 바탕의 오차까지 맞춰 학습되어 원래 fp16 바탕에 얹으면 다른 모델이 된다, docs/experiments.md
# 5단계 발견 2). export.json에도 base_from_4bit가 남는다. qlora false(일반 LoRA)는 보통 합치기 그대로다.
def test_a_qlora_adapter_is_merged_into_the_dequantized_4bit_base(tmp_path, fake_tools):
    a = _adapter(tmp_path, "a", b"A")
    (tmp_path / "train_config.json").write_text('{"qlora": true}', encoding="utf-8")
    out = tmp_path / "export"
    _export(a, out)
    assert fake_tools.from_4bit is True
    assert json.loads((out / "export.json").read_text(encoding="utf-8"))["base_from_4bit"] is True
    # a LoRA adapter trained on the fp16 base keeps the plain merge
    (tmp_path / "train_config.json").write_text('{"qlora": false}', encoding="utf-8")
    _export(_adapter(tmp_path, "b", b"B"), tmp_path / "export-b")
    assert fake_tools.from_4bit is False
