# [테스트 묶음] 5단계 내보내기 call_summary.export(어댑터 합치기 → GGUF f16 변환 → llama-quantize → ollama
# create)의 판단 로직을 모델·도구 없이 검증한다: 같은 어댑터면 합친 가중치를 다시 쓰고, 다른 어댑터의 가중치는
# 거절하며, 중간에 멈춘 빌드를 예전 것으로 착각하지 않고, 양자화는 llama-quantize로 하며, QLoRA 어댑터는 4bit
# 바탕을 풀어 합친다.
# [왜 중요한가] 합친 가중치와 f16 GGUF를 다시 쓰는 것은 시간을 아끼지만, 다른 어댑터의 가중치를 재사용하면
# 엉뚱한 모델을 서빙·측정하게 된다(커밋 ed76ea6: source.json 도장이 맞을 때만 재사용). QLoRA 어댑터를 fp16
# 바탕에 합치면 약 7%p 떨어진 것도 실제로 겪었다(docs/experiments.md 5단계 발견 2).

import hashlib
import json
import subprocess
import sys

import pytest

# adapter_hash: 어댑터 파일 바이트의 해시(evaluate의 모델 지문과 같은 함수). export.json에 남는 값과 비교한다.
# BASE_REVISIONS: 바탕 모델을 고정해 불러오는 커밋 revision 표.
from call_summary import export
from call_summary.evaluate import adapter_hash
from call_summary.providers import BASE_REVISIONS


# 진짜 합치기·변환·양자화·ollama 호출 대신 쓰는 가짜 도구(fixture). merge는 merged/config.json만 쓰고 어떤
# 어댑터로, 4bit 바탕으로, 어느 revision의 바탕으로 합쳤는지 기록한다. subprocess.run과 subprocess.Popen은
# 명령을 기록하고 변환기·llama-quantize가 만들 출력 파일만 빈 파일로 만든다. 양자화는 stderr를 한 줄씩 받아야
# 해서 export가 Popen으로 부르고, 그 밖의 도구(변환기, ollama, git)는 run으로 부른다. 어느 쪽으로 불러도 같은
# 가짜가 돌도록 둘 다 바꾼다. monkeypatch라 테스트가 끝나면 원래 함수로 돌아간다.
@pytest.fixture
def fake_tools(monkeypatch):
    """merge는 merged/config.json을, GGUF 변환기는 --outfile을, llama-quantize는 출력 GGUF를 쓰며, ollama와
    git은 늘 성공한다. 가짜 llama-quantize가 stderr에 낼 줄과 종료 코드는 돌려받은 객체의
    quantizer_stderr(기본: 줄 없음)와 quantizer_returncode(기본 0)로 정한다. 종료 코드가 0이 아니면 출력
    GGUF를 만들지 않는다."""

    class Merges(list):
        pass

    merges = Merges()
    calls = []
    merges.quantizer_stderr = []  # type: ignore[attr-defined]
    merges.quantizer_returncode = 0  # type: ignore[attr-defined]

    def fake_merge(base, adapter, out, from_4bit=False, revision=None):
        merges.append(adapter)
        merges.from_4bit = from_4bit  # type: ignore[attr-defined]
        merges.revision = revision  # type: ignore[attr-defined]
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")

    # 명령 하나를 가짜로 돌린다: 명령을 기록하고, 변환기면 --outfile을, llama-quantize면 출력 GGUF(인수
    # 셋째)를 빈 파일로 만든다. 종료 코드를 돌려준다(llama-quantize만 quantizer_returncode, 나머지는 0).
    def run_tool(cmd):
        calls.append([str(c) for c in cmd])
        if any(str(c).endswith("convert_hf_to_gguf.py") for c in cmd):
            open(cmd[cmd.index("--outfile") + 1], "wb").close()
        if str(cmd[0]).endswith("llama-quantize.exe"):
            if merges.quantizer_returncode:  # type: ignore[attr-defined]
                return merges.quantizer_returncode  # type: ignore[attr-defined]
            open(cmd[2], "wb").close()
        return 0

    # subprocess.run 대신: check=True에 종료 코드가 0이 아니면 진짜처럼 CalledProcessError를 낸다.
    def fake_run(cmd, check=False, **kw):
        code = run_tool(cmd)
        if check and code:
            raise subprocess.CalledProcessError(code, cmd)
        return subprocess.CompletedProcess(cmd, code, stdout="x")

    # subprocess.Popen 대신(with 문으로 쓴다): stderr는 quantizer_stderr의 줄을 차례로 내고, wait()는 종료
    # 코드를 돌려준다. 받은 키워드 인수(stderr=PIPE, encoding 등)는 무시한다.
    class FakePopen:
        def __init__(self, cmd, **kw):
            self.args = cmd
            self.returncode = run_tool(cmd)
            self.stderr = iter(list(merges.quantizer_stderr))  # type: ignore[attr-defined]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def wait(self):
            return self.returncode

    monkeypatch.setattr(export, "merge", fake_merge)
    monkeypatch.setattr(export.subprocess, "run", fake_run)
    monkeypatch.setattr(export.subprocess, "Popen", FakePopen)
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
    # 거절 메시지대로 merged/와 F16 GGUF를 지운다(source.json은 남는다) ...
    for f in (out / "merged").iterdir():
        f.unlink()
    (out / "merged").rmdir()
    (out / "model-f16.gguf").unlink()

    # ... 그다음 어댑터 b의 합치기가 반쯤 쓰고 죽는다
    def crashing_merge(base, adapter, target, from_4bit=False, revision=None):
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        raise RuntimeError("out of memory")

    monkeypatch.setattr(export, "merge", crashing_merge)
    with pytest.raises(RuntimeError):
        _export(_adapter(tmp_path, "b", b"B"), out)
    with pytest.raises(SystemExit):  # b의 반쯤 합친 가중치를 a의 것으로 넘기면 안 된다
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
    # fp16 바탕으로 학습한 LoRA 어댑터는 그대로 일반 합치기를 쓴다
    (tmp_path / "train_config.json").write_text('{"qlora": false}', encoding="utf-8")
    _export(_adapter(tmp_path, "b", b"B"), tmp_path / "export-b")
    assert fake_tools.from_4bit is False


# 바탕 모델은 기본으로 고정된 커밋 revision으로 불러온다(providers.BASE_REVISIONS: 지금 캐시에 있는 스냅샷과
# 같은 커밋이라 내려받기 없이 같은 파일을 읽는다). --revision으로 바꿀 수 있고 표에 없는 바탕은 None(예전
# 동작)이다. 합치기에 넘어간 revision과 export.json의 revision이 같아야 서빙 모델의 출처를 되짚을 수 있다.
def test_export_pins_the_base_revision_and_records_it(tmp_path, fake_tools):
    a = _adapter(tmp_path, "a", b"A")
    pinned = BASE_REVISIONS["Qwen/Qwen3-4B"]
    _export(a, tmp_path / "e1")
    assert fake_tools.revision == pinned
    assert json.loads((tmp_path / "e1" / "export.json").read_text(encoding="utf-8"))["revision"] == pinned
    argv = ["--adapter", a, "--name", "cs", "--out", str(tmp_path / "e2")]
    assert export.main(["--base", "Qwen/Qwen3-4B", "--revision", "abc", *argv]) == 0
    assert fake_tools.revision == "abc"
    assert json.loads((tmp_path / "e2" / "export.json").read_text(encoding="utf-8"))["revision"] == "abc"
    argv[-1] = str(tmp_path / "e3")
    assert export.main(["--base", "other/model", *argv]) == 0
    assert fake_tools.revision is None
    assert json.loads((tmp_path / "e3" / "export.json").read_text(encoding="utf-8"))["revision"] is None


# 어댑터 없이 바탕을 그대로 내보낼 때는 HF 캐시의 스냅샷 폴더를 변환한다. snapshot_download에도 같은 고정
# revision이 가고 local_files_only라 내려받지 않는다.
def test_export_of_the_bare_base_asks_the_cache_for_the_pinned_snapshot(tmp_path, fake_tools, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")
    seen: dict = {}

    def fake_snapshot(repo_id, **kw):
        seen.update(repo_id=repo_id, **kw)
        return str(tmp_path / "snap")

    monkeypatch.setattr(hub, "snapshot_download", fake_snapshot)
    out = tmp_path / "export"
    assert export.main(["--base", "Qwen/Qwen3-4B", "--name", "cs", "--out", str(out)]) == 0
    assert seen == {
        "repo_id": "Qwen/Qwen3-4B",
        "revision": BASE_REVISIONS["Qwen/Qwen3-4B"],
        "local_files_only": True,
    }
    info = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert info["adapter"] is None and info["revision"] == BASE_REVISIONS["Qwen/Qwen3-4B"]


# 서빙 모델을 만든 2026-09-30 내보내기 로그에서 llama-quantize가 양자화마다 stderr에 찍은 판 줄과 같은 꼴.
BUILD_LINE = "version: 0.5.0-dev (build 11255, commit 31385c9ce)\n"


# 가짜 llama-quantize 실행 파일(내용은 아무 바이트)을 tmp_path/bin에 만든다. 이름이 llama-quantize.exe라
# fake_tools가 양자화 명령으로 알아본다. 기록될 sha256은 이 바이트에서 나온다.
def _quantizer(tmp_path, content=b"quantizer build 1"):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    exe = d / "llama-quantize.exe"
    exe.write_bytes(content)
    return exe


# --llama-quantize를 정해 export.main을 한 번 부른다(나머지는 _export와 같다).
def _export_with(adapter, out, quantizer):
    argv = ["--base", "Qwen/Qwen3-4B", "--adapter", adapter, "--name", "cs", "--out", str(out)]
    return export.main([*argv, "--llama-quantize", str(quantizer)])


# 내보내기 도구의 판이 export.json에 남는지 본다: 양자화 도구는 llama.cpp 저장소 밖의 배포 실행 파일이라
# 그동안 커밋(llama_cpp 키)만으로는 어느 배포로 양자화했는지 되짚을 수 없었다. 이제 llama_quantize 키에 받은
# 경로, 실행 파일 바이트의 sha256, 양자화 때 도구가 stderr에 찍은 판 줄의 내용, 이번 실행에서 양자화한 형태가
# 남는다. 기존 키는 그대로 남고, 도구의 stderr는 실행 중에 보이도록 그대로 흘려보낸다.
def test_export_records_the_quantizer_sha256_and_its_build_line(tmp_path, fake_tools, capsys):
    exe = _quantizer(tmp_path)
    fake_tools.quantizer_stderr = [BUILD_LINE, "built with Clang 20.1.8 for Windows x86_64\n"]
    out = tmp_path / "export"
    assert _export_with(_adapter(tmp_path, "a", b"A"), out, exe) == 0
    info = json.loads((out / "export.json").read_text(encoding="utf-8"))
    assert info["llama_quantize"] == {
        "path": str(exe),
        "sha256": hashlib.sha256(b"quantizer build 1").hexdigest(),
        "version": "0.5.0-dev (build 11255, commit 31385c9ce)",
        "quantized": ["q8_0", "q4_K_M"],
    }
    kept = {"base", "revision", "adapter", "adapter_sha", "base_from_4bit", "models", "gguf", "llama_cpp"}
    assert kept <= set(info)
    assert "built with Clang 20.1.8 for Windows x86_64" in capsys.readouterr().err


# 같은 --out으로 다시 내보내면 양자화 파일이 이미 있어 양자화를 다시 하지 않는다. 그때 기록은 "이번 실행에서
# 양자화한 것 없음"(빈 quantized, 판 줄 없음)이어야 한다. sha256은 지금 설정된 도구의 것일 뿐 그 파일을 만든
# 도구라는 보장이 없다는 것을 읽는 사람이 빈 목록으로 알 수 있다.
def test_a_rerun_that_reuses_the_quantized_files_records_that_nothing_was_quantized(tmp_path, fake_tools):
    exe = _quantizer(tmp_path)
    fake_tools.quantizer_stderr = [BUILD_LINE]
    a = _adapter(tmp_path, "a", b"A")
    out = tmp_path / "export"
    assert _export_with(a, out, exe) == 0 and _export_with(a, out, exe) == 0
    record = json.loads((out / "export.json").read_text(encoding="utf-8"))["llama_quantize"]
    assert record["quantized"] == [] and record["version"] is None
    assert record["sha256"] == hashlib.sha256(b"quantizer build 1").hexdigest()
    assert len([c for c in fake_tools.calls if c[0] == str(exe)]) == 2  # 첫 실행의 q8_0, q4_K_M뿐


# 양자화 도구가 실패하면(종료 코드 1) 예전처럼 CalledProcessError로 멈추고, 그 실행의 export.json은 쓰지
# 않는다.
def test_a_failing_quantizer_stops_the_export_before_it_is_recorded(tmp_path, fake_tools):
    exe = _quantizer(tmp_path)
    fake_tools.quantizer_returncode = 1
    out = tmp_path / "export"
    with pytest.raises(subprocess.CalledProcessError):
        _export_with(_adapter(tmp_path, "a", b"A"), out, exe)
    assert not (out / "export.json").exists()


# 기록 함수만 따로: --llama-quantize에 경로가 아니라 PATH에 있는 이름을 주면 subprocess처럼 PATH에서 찾아 그
# 파일의 sha256을 남기고, 어디에도 없으면 sha256은 null이다(예: 도구가 없는 CI에서 양자화 파일을 다시 쓸 때).
def test_the_quantizer_record_finds_a_name_on_path_and_is_null_when_missing(tmp_path, monkeypatch):
    exe = _quantizer(tmp_path, b"on path")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", str(exe.parent))
    monkeypatch.chdir(tmp_path)  # 현재 폴더에는 그 이름의 파일이 없다
    record = export.quantizer_record("llama-quantize.exe", None, [])
    assert record == {
        "path": "llama-quantize.exe",
        "sha256": hashlib.sha256(b"on path").hexdigest(),
        "version": None,
        "quantized": [],
    }
    missing = str(tmp_path / "nowhere" / "llama-quantize.exe")
    assert export.quantizer_record(missing, None, [])["sha256"] is None


# run_quantize를 가짜 Popen 없이 진짜 자식 프로세스로 돌려 본다: 이 테스트의 파이썬이 실행 파일 자리에서 작은
# 스크립트를 돌린다(명령 = [파이썬, 스크립트, 출력 파일, 형태]). 스크립트는 stderr에 진행 줄·판 줄·UTF-8이
# 아닌 바이트(cp949의 '가')를 쓰고 출력 파일을 만든 뒤, 형태가 Q8_0이 아니면 종료 코드 3으로 끝난다. 판 줄의
# 내용을 돌려받는지, stderr가 이 프로세스의 stderr로 흘러나오는지, 못 읽는 바이트에서 멈추지 않는지, 종료
# 코드가 0이 아니면 CalledProcessError인지 본다.
def test_run_quantize_streams_a_real_child_stderr_and_returns_its_build_line(tmp_path, capsys):
    script = tmp_path / "quantize.py"
    script.write_text(
        "import sys\n"
        "out, qtype = sys.argv[1], sys.argv[2]\n"
        "sys.stderr.write('main: build info follows\\n')\n"
        "sys.stderr.write('version: 9.9.9 (build 1, commit abc1234)\\n')\n"
        "sys.stderr.flush()\n"
        "sys.stderr.buffer.write(b'path \\xb0\\xa1\\n')\n"
        "open(out, 'wb').close()\n"
        "sys.exit(0 if qtype == 'Q8_0' else 3)\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.gguf"
    assert export.run_quantize(sys.executable, script, dst, "Q8_0") == "9.9.9 (build 1, commit abc1234)"
    assert dst.exists()
    assert "main: build info follows" in capsys.readouterr().err
    with pytest.raises(subprocess.CalledProcessError):
        export.run_quantize(sys.executable, script, tmp_path / "other.gguf", "Q4_K_M")
