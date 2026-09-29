"""Merge a LoRA adapter into its base model and register it with Ollama as GGUF (F16 and quantized).

Steps: merge (CPU, fp16) -> llama.cpp convert_hf_to_gguf.py (F16) -> `ollama create --quantize`.
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


def merge(base: str, adapter: str, out: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float16, device_map={"": "cpu"})
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
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # merged/ and the F16 GGUF are reused only when they were built from the same weights.
    source = {"base": args.base, "adapter_sha": adapter_hash(args.adapter) if args.adapter else None}
    if args.adapter and source["adapter_sha"] is None:
        ap.error(f"no adapter weights in {args.adapter}")
    stamp = out / "source.json"
    merged = out / "merged"
    f16 = out / "model-f16.gguf"
    if (merged.exists() or f16.exists()) and (
        not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")) != source
    ):
        ap.error(f"{out} holds weights from another source; use a new --out or delete merged/ and {f16.name}")
    if args.adapter:
        if not (merged / "config.json").exists():
            print("merging adapter ...", file=sys.stderr)
            merge(args.base, args.adapter, merged)
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
    for q in ["f16", *args.quant]:
        name = f"{args.name}:{q.lower()}"
        mf = out / f"Modelfile.{q}"
        mf.write_text(modelfile(f16, args.num_ctx), encoding="utf-8")
        cmd = ["ollama", "create", name, "-f", str(mf)]
        if q != "f16":
            cmd += ["--quantize", q]
        subprocess.run(cmd, check=True)
        created[q] = name
    llama_rev = subprocess.run(
        ["git", "-C", str(LLAMA_CPP), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    info = {
        "base": args.base,
        "adapter": args.adapter,
        "adapter_sha": source["adapter_sha"],
        "models": created,
        "llama_cpp": llama_rev,
    }
    (out / "export.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
