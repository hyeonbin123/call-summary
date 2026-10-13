"""Merge a LoRA adapter into its base model and register it with Ollama as GGUF (F16 and quantized).

Steps: merge (CPU, fp16) -> llama.cpp convert_hf_to_gguf.py (F16) -> llama-quantize (Q8_0, Q4_K_M, ...)
-> `ollama create` from each GGUF. Ollama 0.34 no longer quantizes GGUF imports, so the quantized files are
made with llama.cpp's own tool (official release binaries in work/llama-bin, not committed).
The llama.cpp checkout lives in work/llama.cpp (not committed); its commit is recorded in export.json.
Merged weights and the F16 GGUF in --out are reused only for the same base and adapter weights
(out/source.json); export.json carries the adapter hash that evaluate manifests record.
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
import json
import os
import subprocess
import sys
from pathlib import Path

# adapter_hash: 어댑터 가중치 파일의 해시. evaluate의 실행 기록(model_fingerprint)과 같은 함수로 계산해,
# 내보낸 모델이 어떤 어댑터에서 나왔는지 맞춰 볼 수 있게 한다.
from .evaluate import adapter_hash

# llama.cpp 소스(변환 스크립트)와 공식 배포 실행 파일(llama-quantize)의 위치. 둘 다 work/ 아래에 두고 커밋하지
# 않는다. Ollama 0.34부터 GGUF를 가져올 때 양자화해 주지 않아 llama.cpp의 양자화 도구를 따로 쓴다(커밋
# bd33a2a).
LLAMA_CPP = Path("work/llama.cpp")
LLAMA_QUANTIZE = Path("work/llama-bin/llama-quantize.exe")


# 어댑터 옆의 train_config.json(train.py가 남김)에서 qlora 값을 읽어 4bit 바탕으로 학습한 어댑터인지 판단한다.
# 이 값으로 합치는 방식을 자동으로 고르므로 사람이 옵션을 잊어도 잘못 합치지 않는다.
def trained_on_4bit(adapter: str | None) -> bool:
    """Whether the adapter was trained on a 4-bit base (QLoRA), read from train.py's train_config.json."""
    if not adapter:
        return False
    cfg = Path(adapter).parent / "train_config.json"
    return cfg.exists() and bool(json.loads(cfg.read_text(encoding="utf-8")).get("qlora"))


# QLoRA 학습 때와 같은 설정(nf4, double quant, fp16 계산)으로 바탕을 GPU에 4bit로 올리고, 4bit 선형층마다
# 가중치를 fp16으로 풀어(dequantize) CPU의 fp16 모델 같은 자리에 덮어쓴다. 바꾼 층 수를 돌려준다(서빙
# 모델에서는 252개, docs/experiments.md 5단계).
# 이렇게 하면 합친 모델의 바탕 가중치가 학습 때 어댑터가 본 값과 같아진다.
def load_dequantized_4bit_into(model, base: str) -> int:
    """Load the base as QLoRA training did (nf4, double quant, fp16 compute) and write every quantized linear
    layer's weight, expanded back to fp16, into `model` (fp16, on the CPU) as it goes. Returns the count."""
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
        base, dtype=torch.float16, quantization_config=quant, device_map={"": 0}
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
# 한다.
# 호출 위치: main(어댑터가 있고 merged/가 아직 없을 때).
def merge(base: str, adapter: str, out: Path, from_4bit: bool = False) -> None:
    """from_4bit: load the base exactly as QLoRA training did (nf4, double quant, fp16 compute), expand those
    4-bit weights back to fp16, then merge. A QLoRA adapter merged into the original fp16 base gives a
    different model (stage 5: dev exact 93 -> 86)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map={"": "cpu"})
    if from_4bit:
        # One layer at a time: the whole model expanded on the GPU does not fit next to its 4-bit copy.
        n = load_dequantized_4bit_into(model, base)
        print(f"{n} linear layers taken from the dequantized 4-bit base", file=sys.stderr)
    # merge_and_unload: LoRA의 저랭크 행렬 곱(B·A·scale)을 바탕 가중치에 더해 넣고 PEFT 껍데기를 벗긴다.
    # 결과는 어댑터 없이 도는 일반 모델이라 llama.cpp가 변환할 수 있다.
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


# Ollama Modelfile 글. GGUF 경로, temperature 0(탐욕적), 컨텍스트 크기만 정한다. 서빙 모델은 --num-ctx
# 4096으로 만들었다(README: 4,096 토큰 안에 답까지 들어가도록 서비스가 4,000자 넘는 전사를 413으로 거절한다).
def modelfile(gguf: Path, num_ctx: int) -> str:
    # The chat template comes from the GGUF metadata; only runtime defaults are set here.
    return f"FROM {gguf.resolve()}\nPARAMETER temperature 0\nPARAMETER num_ctx {num_ctx}\n"


# 명령줄 진입점. --quant로 만들 양자화 목록을 받고(f16은 늘 만든다), --name 뒤에 ":<양자화>"를 붙여 Ollama
# 모델로 등록한다(예: call-summary-4b-dq:q8_0).
# --fp16-base는 예전 방식(QLoRA 어댑터를 원래 fp16 바탕에 합치기)을 비교용으로 다시 만들 때만 쓴다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="HF base model id")
    ap.add_argument("--adapter", help="PEFT adapter dir; omit to export the base model as is")
    ap.add_argument("--name", required=True, help="Ollama model name prefix, e.g. call-summary-qwen3-1.7b")
    ap.add_argument("--out", required=True, help="work dir for merged weights and GGUF files")
    ap.add_argument("--quant", nargs="+", default=["q8_0", "q4_K_M"])
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--llama-quantize", default=str(LLAMA_QUANTIZE), help="llama.cpp quantize tool")
    ap.add_argument(
        "--fp16-base",
        action="store_true",
        help="merge a QLoRA adapter into the fp16 base anyway (old behaviour)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # 합치기와 f16 변환은 오래 걸려 결과를 다시 쓴다. 다만 같은 바탕·같은 어댑터 가중치·같은 합치기 방식일
    # 때만 다시 쓰도록 source.json 도장으로 확인한다(커밋 ed76ea6).
    # merged/ and the F16 GGUF are reused only when they were built from the same weights.
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
        # A build is about to start: drop an older stamp so an interrupted merge or convert is never
        # taken for that older source's weights on the next run.
        stamp.unlink(missing_ok=True)
        # 양자화 파일은 자기를 만든 f16에 딸린 것이므로, f16을 새로 만들 때는 예전 양자화 파일도 지운다.
        for old in out.glob("model-*.gguf"):  # quantized files belong to the F16 they came from
            if old != f16:
                old.unlink()
    if args.adapter:
        if not (merged / "config.json").exists():
            print("merging adapter ...", file=sys.stderr)
            merge(args.base, args.adapter, merged, from_4bit=from_4bit)
        src = str(merged)
    else:
        # 어댑터가 없으면 바탕 모델 자체를 내보낸다(HF 캐시의 스냅샷 폴더를 그대로 변환한다.
        # local_files_only라 내려받지 않는다).
        from huggingface_hub import snapshot_download

        src = snapshot_download(args.base, local_files_only=True)

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
    stamp.write_text(json.dumps(source, indent=2), encoding="utf-8")  # only once merge and convert finished
    # 양자화마다: GGUF가 없으면 llama-quantize로 f16에서 만들고, Modelfile을 써서 ollama create로 등록한다.
    # 모델 이름과 파일 크기를 기록한다(README 5단계 표의 파일 크기 4.3 GB, 2.5 GB 등).
    created = {}
    ggufs = {}
    for q in ["f16", *args.quant]:
        gguf = f16 if q == "f16" else out / f"model-{q.lower()}.gguf"
        if not gguf.exists():
            subprocess.run([args.llama_quantize, str(f16), str(gguf), q.upper()], check=True)
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
    # export.json: 바탕, 어댑터 경로와 해시, 4bit를 풀어 합쳤는지, 등록한 Ollama 모델 이름, GGUF 파일과 크기,
    # llama.cpp 커밋. 서빙 측정의 출처 기록이다.
    info = {
        "base": args.base,
        "adapter": args.adapter,
        "adapter_sha": source["adapter_sha"],
        "base_from_4bit": from_4bit,
        "models": created,
        "gguf": ggufs,
        "llama_cpp": llama_rev,
    }
    (out / "export.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
