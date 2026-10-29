"""LoRA 어댑터를 바탕 모델에 합치고 GGUF(F16과 양자화본)로 Ollama에 등록한다.

단계: 합치기(CPU, fp16) -> llama.cpp convert_hf_to_gguf.py(F16) -> llama-quantize(Q8_0, Q4_K_M, ...)
-> GGUF마다 `ollama create`. Ollama 0.34는 GGUF를 가져올 때 더는 양자화하지 않으므로 양자화 파일은
llama.cpp 자체 도구(work/llama-bin의 공식 배포 바이너리, 커밋하지 않음)로 만든다.
llama.cpp 체크아웃은 work/llama.cpp에 있고(커밋하지 않음) 그 커밋은 export.json에 기록한다.
llama-quantize는 실행 파일의 sha256과 양자화 때 찍는 판 줄을 export.json에 기록한다.
--out의 합친 가중치와 F16 GGUF는 바탕과 어댑터 가중치가 같을 때만 다시 쓴다(out/source.json).
export.json에는 evaluate의 manifest가 기록하는 어댑터 해시가 들어간다.
"""

# [파일 역할] 5단계(서빙)의 내보내기 도구다. 학습한 LoRA 어댑터를 바탕 모델에 합치고, llama.cpp로 GGUF(f16)를
# 만든 뒤 q8_0·q4_K_M으로 양자화해 Ollama에 모델로 등록한다. 서빙 모델 call-summary-4b-dq:q8_0이 이 도구로
# 만들어졌다(README 실행 명령).
# [왜 필요한가] 학습 결과는 transformers용 어댑터라 그대로는 Ollama·FastAPI 서비스로 서빙할 수 없다. GGUF로
# 바꾸면 Ollama로 가볍게 서빙하고 양자화 단계별(f16, q8_0, q4_K_M) 정확도·지연·메모리를 비교할 수 있다(계획
# 질문 5).
# [핵심 결정] QLoRA 어댑터는 원래 fp16 바탕이 아니라, 학습 때와 같은 설정으로 4bit로 불러 fp16으로 풀어낸
# 바탕에 합친다. fp16 바탕에 합치면 dev exact가 약 7%p 떨어졌다(docs/experiments.md 5단계 "발견 2": 어댑터가
# nf4 바탕의 오차까지 맞춰 학습됐기 때문). 고친 뒤 q8_0이 학습한 모델 그대로와 같은 94.6을 냈다.
# [흐름] 출처 도장 확인(source.json) → 합치기(merged/) → convert_hf_to_gguf.py로 f16 GGUF → llama-quantize로
# 양자화 → 양자화마다 Modelfile을 쓰고 ollama create → export.json 기록.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# adapter_hash: 어댑터 가중치 파일의 해시. evaluate의 실행 기록(model_fingerprint)과 같은 함수로 계산해,
# 내보낸 모델이 어떤 어댑터에서 나왔는지 맞춰 볼 수 있게 한다. revision_for: 바탕 모델의 고정
# revision(학습·평가와 같은 규칙).
from .evaluate import adapter_hash
from .providers import revision_for

# llama.cpp 소스(변환 스크립트)와 공식 배포 실행 파일(llama-quantize)의 위치. 둘 다 work/ 아래에 두고 커밋하지
# 않는다. Ollama 0.34부터 GGUF를 가져올 때 양자화해 주지 않아 llama.cpp의 양자화 도구를 따로 쓴다(커밋
# bd33a2a).
LLAMA_CPP = Path("work/llama.cpp")
LLAMA_QUANTIZE = Path("work/llama-bin/llama-quantize.exe")

# llama-quantize가 양자화를 시작할 때 stderr에 찍는 판 줄의 머리. llama.cpp common의 build-info가
# "version: 0.5.0-dev (build 11255, commit 31385c9ce)" 꼴로 쓴다(서빙 모델을 만든 2026-09-30 내보내기 로그에
# 양자화마다 찍혔다). build 번호와 commit이 그 실행 파일이 나온 llama.cpp 릴리스를 가리킨다.
BUILD_LINE_PREFIX = "version:"


# 어댑터 옆의 train_config.json(train.py가 남김)에서 qlora 값을 읽어 4bit 바탕으로 학습한 어댑터인지 판단한다.
# 이 값으로 합치는 방식을 자동으로 고르므로 사람이 옵션을 잊어도 잘못 합치지 않는다.
def trained_on_4bit(adapter: str | None) -> bool:
    """어댑터를 4bit 바탕(QLoRA)으로 학습했는지. train.py가 남긴 train_config.json에서 읽는다."""
    if not adapter:
        return False
    cfg = Path(adapter).parent / "train_config.json"
    return cfg.exists() and bool(json.loads(cfg.read_text(encoding="utf-8")).get("qlora"))


# QLoRA 학습 때와 같은 설정(nf4, double quant, fp16 계산)으로 바탕을 GPU에 4bit로 올리고, 4bit 선형층마다
# 가중치를 fp16으로 풀어(dequantize) CPU의 fp16 모델 같은 자리에 덮어쓴다. 바꾼 층 수를 돌려준다(서빙
# 모델에서는 252개, docs/experiments.md 5단계).
# 이렇게 하면 합친 모델의 바탕 가중치가 학습 때 어댑터가 본 값과 같아진다. revision은 merge가 fp16 바탕을
# 불러온 것과 같은 값이어야 한다(다른 revision의 4bit 바탕을 풀어 넣으면 두 바탕이 섞인다).
def load_dequantized_4bit_into(model, base: str, revision: str | None = None) -> int:
    """QLoRA 학습 때처럼(nf4, double quant, fp16 계산) 바탕을 올리고, 양자화된 선형층마다 가중치를
    fp16으로 풀어 그때그때 `model`(fp16, CPU)에 써 넣는다. 바꾼 층 수를 돌려준다."""
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # train.py의 QLoRA 설정(nf4, fp16 계산, double quant)과 같아야 한다. 하나라도 다르면 풀어낸 바탕이 학습
    # 때의 바탕과 달라진다.
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    q = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=torch.float16, quantization_config=quant, device_map={"": 0}
    )
    n = 0
    # 4bit 층 이름으로 fp16 모델의 같은 층을 찾아 바꾼다. 모양이 다르면 다른 구조의 모델이므로 바로 멈춘다.
    # 4bit로 양자화되지 않은 층(임베딩, 정규화 층 등)은 원래 fp16 값 그대로 둔다.
    for name, module in q.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
            target = model.get_submodule(name).weight
            if tuple(w.shape) != tuple(target.shape):
                raise ValueError(f"{name}: {tuple(w.shape)} vs {tuple(target.shape)}")
            target.data = w.to(torch.float16).cpu()
            n += 1
    # 4bit 모델을 지우고 GPU 캐시를 비워, 이어지는 작업에 GPU 메모리를 돌려준다.
    del q
    torch.cuda.empty_cache()
    return n


# 바탕에 어댑터를 합쳐 out에 저장한다(가중치는 safetensors, 토크나이저도 함께). 합치기는 CPU에서 fp16으로
# 한다. revision은 바탕 가중치·토크나이저·4bit 바탕 모두에 같은 값으로 넘긴다(None이면 refs/main).
# 호출 위치: main(어댑터가 있고 merged/가 아직 없을 때).
def merge(base: str, adapter: str, out: Path, from_4bit: bool = False, revision: str | None = None) -> None:
    """from_4bit: QLoRA 학습 때와 똑같이(nf4, double quant, fp16 계산) 바탕을 올리고, 그 4bit 가중치를
    fp16으로 풀어 낸 뒤 합친다. QLoRA 어댑터를 원래 fp16 바탕에 합치면 다른 모델이 된다(5단계: dev exact
    93 -> 86)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base, revision=revision, dtype=torch.float16, device_map={"": "cpu"}
    )
    if from_4bit:
        # 한 층씩 옮긴다: 모델 전체를 GPU에서 fp16으로 풀면 4bit 사본 옆에 들어가지 않는다.
        n = load_dequantized_4bit_into(model, base, revision)
        print(f"{n} linear layers taken from the dequantized 4-bit base", file=sys.stderr)
    # merge_and_unload: LoRA의 저랭크 행렬 곱(B·A·scale)을 바탕 가중치에 더해 넣고 PEFT 껍데기를 벗긴다.
    # 결과는 어댑터 없이 도는 일반 모델이라 llama.cpp가 변환할 수 있다.
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, revision=revision).save_pretrained(out)


# Ollama Modelfile 글. GGUF 경로, temperature 0(탐욕적), 컨텍스트 크기만 정한다. 서빙 모델은 --num-ctx
# 4096으로 만들었다(README: 4,096 토큰 안에 답까지 들어가도록 서비스가 4,000자 넘는 전사를 413으로 거절한다).
def modelfile(gguf: Path, num_ctx: int) -> str:
    # 채팅 템플릿은 GGUF 메타데이터에서 오므로 여기서는 실행 기본값만 정한다.
    return f"FROM {gguf.resolve()}\nPARAMETER temperature 0\nPARAMETER num_ctx {num_ctx}\n"


# llama-quantize로 src(f16 GGUF)를 qtype(Q8_0, Q4_K_M 등)으로 양자화해 dst에 쓴다. 도구의 stderr(텐서별 진행
# 표시 등)는 한 줄씩 그대로 이 프로세스의 stderr로 흘려보내 실행 중에 보이게 하고, 그 가운데 판 줄의 내용
# (BUILD_LINE_PREFIX 뒤, 예: "0.5.0-dev (build 11255, commit 31385c9ce)")을 돌려준다. 판 줄이 없으면 None.
# 판 줄을 여기서 받는 이유: llama-quantize는 `--version`을 모르는 인수로 보고 사용법만 찍고 끝나므로(llama.cpp
# tools/quantize/quantize.cpp의 인수 처리), 판은 실제로 양자화할 때 찍는 이 줄로만 알 수 있다.
# stdout은 그대로 콘솔에 두고 stderr만 파이프로 받는다: 둘 다 파이프로 받고 한쪽만 읽으면 다른 쪽이
# 차서 도구가 멈출 수 있다. 바이트는 UTF-8로 읽고 못 읽는 바이트는 바꿔 보인다(Windows에서 도구가 ANSI
# 코드 페이지로 찍은 한글 경로는 깨져 보일 수 있으나 보이는 글만 바뀐다).
# 종료 코드가 0이 아니면 예전의 subprocess.run(check=True)처럼 CalledProcessError를 낸다(도구가 없으면
# FileNotFoundError).
def run_quantize(exe: str, src: Path, dst: Path, qtype: str) -> str | None:
    """llama-quantize를 돌리고, 도구가 stderr에 찍은 판 줄("version: ...")의 내용을 돌려준다."""
    cmd = [exe, str(src), str(dst), qtype]
    version = None
    with subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding="utf-8", errors="replace") as proc:
        for line in proc.stderr:
            sys.stderr.write(line)
            if version is None and line.startswith(BUILD_LINE_PREFIX):
                version = line[len(BUILD_LINE_PREFIX) :].strip()
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)
    return version


# export.json의 llama_quantize 기록을 만든다.
# - path: --llama-quantize로 받은 그대로
# - sha256: 그 실행 파일 바이트의 sha256(16진 64자리). 그 경로에 파일이 없으면 subprocess처럼 PATH에서
#   이름으로 찾고(shutil.which), 그래도 없으면 None
# - version: 이번 실행의 양자화에서 받은 판 줄의 내용(run_quantize). 양자화를 하지 않았으면 None
# - quantized: 이번 실행에서 이 도구로 만든 양자화 형태(--quant의 표기 그대로, 예: ["q8_0", "q4_K_M"])
# quantized가 비어 있으면 양자화 파일은 예전 실행이 만든 것을 다시 쓴 것이다. 그때 sha256은 지금 설정된 도구를
# 가리킬 뿐 그 파일을 만든 도구라는 보장이 없다.
# sha256과 판 줄을 함께 남기는 이유: llama.cpp 공식 Windows 배포의 llama-quantize.exe는 양자화 코드가 든
# llama-quantize-impl.dll을 부르는 작은 실행 파일이라(llama.cpp tools/quantize/CMakeLists.txt) 실행 파일의
# sha256만으로는 양자화 코드의 판까지 가리키지 못한다. 판 줄은 실제로 양자화를 한 코드가 찍은 것이고, sha256은
# 받은 배포(릴리스 zip)의 같은 파일과 바이트로 대 볼 수 있다.
def quantizer_record(exe: str, version: str | None, quantized: list[str]) -> dict:
    """llama-quantize 실행 파일의 경로·sha256과 이번 실행의 판 줄·양자화 형태를 묶는다."""
    path = Path(exe)
    if not path.is_file():
        found = shutil.which(exe)
        path = Path(found) if found else None
    sha256 = None
    if path is not None:
        with path.open("rb") as f:
            sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": exe, "sha256": sha256, "version": version, "quantized": quantized}


# 명령줄 진입점. --quant로 만들 양자화 목록을 받고(f16은 늘 만든다), --name 뒤에 ":<양자화>"를 붙여 Ollama
# 모델로 등록한다(예: call-summary-4b-dq:q8_0).
# --fp16-base는 예전 방식(QLoRA 어댑터를 원래 fp16 바탕에 합치기)을 비교용으로 다시 만들 때만 쓴다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="HF 바탕 모델 id")
    # 바탕 모델의 Hub revision. 기본은 providers.BASE_REVISIONS의 고정 커밋(학습·평가와 같은 규칙), 표에 없는
    # 바탕은 고정하지 않는다. 쓴 값은 export.json의 revision에 남는다.
    ap.add_argument(
        "--revision", help="바탕 모델의 HF Hub revision. 기본: 고정 표의 커밋, 표에 없는 모델은 고정 없음"
    )
    ap.add_argument("--adapter", help="PEFT 어댑터 폴더. 빼면 바탕 모델을 그대로 내보낸다")
    ap.add_argument("--name", required=True, help="Ollama 모델 이름 앞부분. 예: call-summary-qwen3-1.7b")
    ap.add_argument("--out", required=True, help="합친 가중치와 GGUF 파일을 둘 작업 폴더")
    ap.add_argument("--quant", nargs="+", default=["q8_0", "q4_K_M"])
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument(
        "--llama-quantize",
        default=str(LLAMA_QUANTIZE),
        help="llama.cpp 양자화 도구(llama-quantize) 실행 파일. 경로 또는 PATH의 이름. "
        "sha256과 양자화 때 찍힌 판 줄이 export.json의 llama_quantize에 남는다",
    )
    ap.add_argument(
        "--fp16-base",
        action="store_true",
        help="QLoRA 어댑터라도 fp16 바탕에 합친다(예전 방식)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 바탕 모델을 불러올 revision(학습·평가와 같은 규칙). 도장(source.json)에는 넣지 않는다: 기록된 내보내기
    # 폴더의 도장이 어긋나 같은 어댑터의 재사용이 거절되고, 지금 캐시에는 바탕마다 스냅샷이 하나뿐이라 도장
    # 없이도 revision이 달라질 수 없다. 대신 export.json에 남겨 서빙 모델의 출처를 되짚을 수 있게 한다.
    revision = revision_for(args.base, args.revision)
    # 합치기와 f16 변환은 오래 걸려 결과를 다시 쓴다. 다만 같은 바탕·같은 어댑터 가중치·같은 합치기 방식일
    # 때만 merged/와 F16 GGUF를 다시 쓰도록 source.json 도장으로 확인한다(커밋 ed76ea6).
    from_4bit = trained_on_4bit(args.adapter) and not args.fp16_base
    source = {
        "base": args.base,
        "adapter_sha": adapter_hash(args.adapter) if args.adapter else None,
        "base_from_4bit": from_4bit,
    }
    # 어댑터 폴더에 가중치 파일이 없으면 해시가 None이 되므로 잘못된 경로로 보고 멈춘다.
    if args.adapter and source["adapter_sha"] is None:
        ap.error(f"no adapter weights in {args.adapter}")
    stamp = out / "source.json"
    merged = out / "merged"
    f16 = out / "model-f16.gguf"
    # 이미 결과가 있는데 도장이 없거나 지금 출처와 다르면, 다른 가중치의 결과를 실수로 다시 쓰지 않도록
    # 멈춘다. 새 --out을 쓰거나 지우라고 알려 준다.
    if (merged.exists() or f16.exists()) and (
        not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")) != source
    ):
        ap.error(f"{out} holds weights from another source; use a new --out or delete merged/ and {f16.name}")
    if not f16.exists() or (args.adapter and not (merged / "config.json").exists()):
        # 이제 새로 만든다: 예전 도장을 먼저 지워, 합치기나 변환이 중간에 끊겨도 다음 실행에서 그 결과를
        # 예전 출처의 가중치로 잘못 보지 않게 한다.
        stamp.unlink(missing_ok=True)
        # 양자화 파일은 자기를 만든 f16에 딸린 것이므로, f16을 새로 만들 때는 예전 양자화 파일도 지운다.
        for old in out.glob("model-*.gguf"):  # 양자화 파일은 자기를 만든 F16에 딸린다
            if old != f16:
                old.unlink()
    if args.adapter:
        if not (merged / "config.json").exists():
            print("merging adapter ...", file=sys.stderr)
            merge(args.base, args.adapter, merged, from_4bit=from_4bit, revision=revision)
        src = str(merged)
    else:
        # 어댑터가 없으면 바탕 모델 자체를 내보낸다(HF 캐시의 같은 revision 스냅샷 폴더를 그대로 변환한다.
        # local_files_only라 내려받지 않는다).
        from huggingface_hub import snapshot_download

        src = snapshot_download(args.base, revision=revision, local_files_only=True)

    # llama.cpp의 convert_hf_to_gguf.py로 f16 GGUF를 만든다. 스크립트가 같은 저장소의 gguf-py 패키지를 쓰므로
    # PYTHONPATH에 넣고, 이 프로젝트의 파이썬(sys.executable)으로 돌린다.
    if not f16.exists():
        env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP / "gguf-py")}
        subprocess.run(
            [
                sys.executable,
                str(LLAMA_CPP / "convert_hf_to_gguf.py"),
                src,
                "--outtype",
                "f16",
                "--outfile",
                str(f16),
            ],
            check=True,
            env=env,
        )
    stamp.write_text(json.dumps(source, indent=2), encoding="utf-8")  # 합치기와 변환이 끝난 뒤에만 쓴다
    # 양자화마다: GGUF가 없으면 llama-quantize로 f16에서 만들고, Modelfile을 써서 ollama create로 등록한다.
    # 모델 이름과 파일 크기를 기록한다(README 5단계 표의 파일 크기 4.3 GB, 2.5 GB 등). 이번 실행에서 양자화한
    # 형태와 도구가 찍은 판 줄(처음 받은 것)은 llama_quantize 기록에 쓴다.
    created = {}
    ggufs = {}
    quantized: list[str] = []
    quantizer_version = None
    for q in ["f16", *args.quant]:
        gguf = f16 if q == "f16" else out / f"model-{q.lower()}.gguf"
        if not gguf.exists():
            version = run_quantize(args.llama_quantize, f16, gguf, q.upper())
            quantizer_version = quantizer_version or version
            quantized.append(q)
        name = f"{args.name}:{q.lower()}"
        mf = out / f"Modelfile.{q}"
        mf.write_text(modelfile(gguf, args.num_ctx), encoding="utf-8")
        subprocess.run(["ollama", "create", name, "-f", str(mf)], check=True)
        created[q] = name
        ggufs[q] = {"file": gguf.name, "bytes": gguf.stat().st_size}
    # 변환에 쓴 llama.cpp 커밋을 남긴다. llama.cpp는 커밋하지 않는 work/ 아래에 있어 이 기록이 없으면 변환
    # 조건을 되짚을 수 없다.
    llama_rev = subprocess.run(
        ["git", "-C", str(LLAMA_CPP), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    # export.json: 바탕과 그 revision, 어댑터 경로와 해시, 4bit를 풀어 합쳤는지, 등록한 Ollama 모델 이름, GGUF
    # 파일과 크기, llama.cpp 커밋, llama-quantize의 실행 파일 sha256·판 줄·이번에 양자화한 형태
    # (quantizer_record). 서빙 측정의 출처 기록이다(revision 키나 llama_quantize 키가 없는 export.json은 그
    # 값을 남기기 전에 만든 것이다. 서빙 모델을 만든 도구 판은 README "실행"의 내보내기 도구 문단).
    info = {
        "base": args.base,
        "revision": revision,
        "adapter": args.adapter,
        "adapter_sha": source["adapter_sha"],
        "base_from_4bit": from_4bit,
        "models": created,
        "gguf": ggufs,
        "llama_cpp": llama_rev,
        "llama_quantize": quantizer_record(args.llama_quantize, quantizer_version, quantized),
    }
    (out / "export.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
