"""Merge a LoRA adapter into its base model and register it with Ollama as GGUF (F16 and quantized).

Steps: merge (CPU, fp16) -> llama.cpp convert_hf_to_gguf.py (F16) -> llama-quantize (Q8_0, Q4_K_M, ...)
-> `ollama create` from each GGUF. Ollama 0.34 no longer quantizes GGUF imports, so the quantized files are
made with llama.cpp's own tool (official release binaries in work/llama-bin, not committed).
The llama.cpp checkout lives in work/llama.cpp (not committed); its commit is recorded in export.json.
Merged weights and the F16 GGUF in --out are reused only for the same base and adapter weights
(out/source.json); export.json carries the adapter hash that evaluate manifests record.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .evaluate import adapter_hash

LLAMA_CPP = Path("work/llama.cpp")
LLAMA_QUANTIZE = Path("work/llama-bin/llama-quantize.exe")


def trained_on_4bit(adapter: str | None) -> bool:
    """Whether the adapter was trained on a 4-bit base (QLoRA), read from train.py's train_config.json."""
    if not adapter:
        return False
    cfg = Path(adapter).parent / "train_config.json"
    return cfg.exists() and bool(json.loads(cfg.read_text(encoding="utf-8")).get("qlora"))


def load_dequantized_4bit_into(model, base: str) -> int:
    """Load the base as QLoRA training did (nf4, double quant, fp16 compute) and write every quantized linear
    layer's weight, expanded back to fp16, into `model` (fp16, on the CPU) as it goes. Returns the count."""
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
    for name, module in q.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            w = bnb.functional.dequantize_4bit(module.weight.data, module.weight.quant_state)
            target = model.get_submodule(name).weight
            if tuple(w.shape) != tuple(target.shape):
                raise ValueError(f"{name}: {tuple(w.shape)} vs {tuple(target.shape)}")
            target.data = w.to(torch.float16).cpu()
            n += 1
    del q
    torch.cuda.empty_cache()
    return n


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
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)


def modelfile(gguf: Path, num_ctx: int) -> str:
    # The chat template comes from the GGUF metadata; only runtime defaults are set here.
    return f"FROM {gguf.resolve()}\nPARAMETER temperature 0\nPARAMETER num_ctx {num_ctx}\n"


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
    # merged/ and the F16 GGUF are reused only when they were built from the same weights.
    from_4bit = trained_on_4bit(args.adapter) and not args.fp16_base
    source = {
        "base": args.base,
        "adapter_sha": adapter_hash(args.adapter) if args.adapter else None,
        "base_from_4bit": from_4bit,
    }
    if args.adapter and source["adapter_sha"] is None:
        ap.error(f"no adapter weights in {args.adapter}")
    stamp = out / "source.json"
    merged = out / "merged"
    f16 = out / "model-f16.gguf"
    if (merged.exists() or f16.exists()) and (
        not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")) != source
    ):
        ap.error(f"{out} holds weights from another source; use a new --out or delete merged/ and {f16.name}")
    if not f16.exists() or (args.adapter and not (merged / "config.json").exists()):
        # A build is about to start: drop an older stamp so an interrupted merge or convert is never
        # taken for that older source's weights on the next run.
        stamp.unlink(missing_ok=True)
        for old in out.glob("model-*.gguf"):  # quantized files belong to the F16 they came from
            if old != f16:
                old.unlink()
    if args.adapter:
        if not (merged / "config.json").exists():
            print("merging adapter ...", file=sys.stderr)
            merge(args.base, args.adapter, merged, from_4bit=from_4bit)
        src = str(merged)
    else:
        from huggingface_hub import snapshot_download

        src = snapshot_download(args.base, local_files_only=True)

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
    llama_rev = subprocess.run(
        ["git", "-C", str(LLAMA_CPP), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
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
