"""Speech condition v3, pass 2 for the Qwen3-ASR arms: Qwen3-ASR-1.7B on the WAVs kept by pass 1.

It runs in this project's own environment (transformers 5.17 has the model class; numpy and torch from the
`train` group), on the audio scripts/asr_condition.py --keep-audio kept:

    uv run --no-sync python scripts/asr_qwen.py --arm Q --split dev
    uv run --no-sync python scripts/asr_qwen.py --arm Qc --split dev --context outputs/asr3/context.json

Decoding follows whisper-ko-ft stage 9 (`whisper_ko_ft.evaluate_qwen`): the Transformers classes at a pinned
revision, fp16, SDPA attention, greedy, the language forced through the official prompt
(`apply_transcription_request(..., language="Korean")`), at most 256 new tokens, left padding. Qc passes the
domain's context (`python -m call_summary.asr_arms context`) as the official `prompt` (the system message).
Batch 8 and a per-process GPU memory cap (default 9.5 GB) keep it inside the 11 GB card.

Output rows are those of asr_condition's pass 2 ({key, split, item_id, turn, arm, heard}) plus the official
parse with its repetition fix (`fixed`), the generated token count, whether the token limit was hit, and
whether any logit of the utterance was not finite (fp16 overflow). `heard` is the raw text after "<asr_text>";
the student gets it after the identifier rule (call_summary.id_itn), not the fixed one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_condition import (  # noqa: E402
    DEFAULT_AUDIO,
    LISTEN_RATE,
    WavStore,
    _append_jsonl,
    _git_commit,
    _read_jsonl,
    write_hyp_meta,
)

MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"  # downloaded 2026-10-04 (whisper-ko-ft stage 9)
LANGUAGE = "Korean"
MARKER = "<asr_text>"
GENERATE = {"do_sample": False, "num_beams": 1, "max_new_tokens": 256}
ARMS = {"Q": False, "Qc": True}  # arm -> the domain context as the prompt


def raw_transcription(text: str) -> str:
    """The official parse of one decoded output without its repetition fix (as whisper-ko-ft stage 9)."""
    text = text.strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]
    if MARKER in text:
        text = text.split(MARKER, 1)[1]
    return text.strip()


def read_wav(path: Path):
    """16-bit mono 16 kHz WAV -> float32 in [-1, 1), as faster-whisper's decode_audio gives it."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, LISTEN_RATE):
            raise ValueError(f"{path}: not 16-bit mono {LISTEN_RATE} Hz")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def todo_rows(store: WavStore, split: str, done: set[str], limit: int | None = None) -> list[dict]:
    rows = [r for r in store.split_rows(split) if r["wav"] is not None and r["key"] not in done]
    return rows[:limit] if limit else rows


def batched(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for begin in range(0, len(rows), size):
        yield rows[begin : begin + size]


def generated_lengths(generated, eos_token_id) -> list[tuple[int, bool]]:
    """(tokens before the first end token, ran into the token limit) for each row."""
    import torch

    eos = torch.tensor(eos_token_id if isinstance(eos_token_id, list | tuple) else [eos_token_id])
    ended = torch.isin(generated.cpu(), eos)
    out = []
    for row in ended:
        hits = row.nonzero()
        out.append((int(hits[0]), False) if len(hits) else (int(row.shape[0]), True))
    return out


class NonFiniteWatch:
    """Flags the rows of a batch whose logits were ever not finite (an fp16 overflow), through a hook."""

    def __init__(self, head) -> None:
        self.rows = None
        head.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        import torch

        bad = ~torch.isfinite(output).flatten(1).all(dim=1)
        self.rows = bad if self.rows is None else self.rows | bad

    def take(self, n: int) -> list[bool]:
        rows, self.rows = self.rows, None
        return [False] * n if rows is None else [bool(x) for x in rows.tolist()]


def load(device: str, dtype_name: str, max_memory_gb: float | None):
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    if device == "cuda" and max_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_memory_gb * 2**30 / total))
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(device).eval(), processor, dtype


def transcribe(
    model, processor, audio: list, prompts: list[str] | None, device: str, dtype, watch
) -> list[dict]:
    import torch

    inputs = processor.apply_transcription_request(audio, language=LANGUAGE, prompt=prompts).to(device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, **GENERATE)
    generated = output[:, inputs["input_ids"].shape[1] :]
    texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    fixed = processor.extract_transcription(texts)
    lengths = generated_lengths(generated, model.generation_config.eos_token_id)
    bad = watch.take(len(texts))
    return [
        {
            "heard": raw_transcription(t),
            "fixed": f,
            "new_tokens": n,
            "hit_token_limit": hit,
            "nonfinite_logits": b,
        }
        for t, f, (n, hit), b in zip(texts, fixed, lengths, bad, strict=True)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--hyp", help="output (default <audio>/hyp/<ARM>.jsonl)")
    ap.add_argument("--context", help="per-domain context JSON (needed by Qc)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-memory-gb", type=float, default=9.5, help="per-process GPU memory cap")
    ap.add_argument("--precision", choices=["fp16", "fp32", "bf16"], default="fp16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit", type=int, help="only the first N utterances to do (smoke)")
    args = ap.parse_args(argv)

    if ARMS[args.arm] and not args.context:
        ap.error(f"{args.arm} needs --context")
    if args.device == "cpu" and not args.hyp:
        ap.error("a CPU run is not the measured condition; give it its own --hyp")
    store = WavStore(Path(args.audio))
    hyp = Path(args.hyp) if args.hyp else store.root / "hyp" / f"{args.arm}.jsonl"
    rows = todo_rows(store, args.split, {r["key"] for r in _read_jsonl(hyp)}, args.limit)
    context = json.loads(Path(args.context).read_text(encoding="utf-8")) if ARMS[args.arm] else None
    print(json.dumps({"arm": args.arm, "split": args.split, "to_do": len(rows)}), file=sys.stderr, flush=True)
    if not rows:
        return 0

    import torch
    import transformers

    model, processor, dtype = load(args.device, args.precision, args.max_memory_gb)
    watch = NonFiniteWatch(model.lm_head)
    write_hyp_meta(
        hyp,
        {
            "arm": args.arm,
            "model": MODEL,
            "revision": REVISION,
            "hotwords": ARMS[args.arm],
            "decoding": {**GENERATE, "language": LANGUAGE, "batch_size": args.batch_size},
            "precision": args.precision,
            "device": args.device,
            "max_memory_gb": args.max_memory_gb,
            "attention": model.config._attn_implementation,
            "context": context,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "commit": _git_commit(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    n: Counter = Counter()
    t0 = time.time()
    audio_seconds = 0.0
    for batch in batched(rows, args.batch_size):
        audio = [read_wav(store.wav_path(r["key"])) for r in batch]
        prompts = [context[r["domain"]] for r in batch] if context else None
        for row, out in zip(
            batch, transcribe(model, processor, audio, prompts, args.device, dtype, watch), strict=True
        ):
            keep = {k: row[k] for k in ("key", "split", "item_id", "turn")}
            _append_jsonl(hyp, {**keep, "arm": args.arm, **out})
            n["recognised"] += 1
            n["hit_token_limit"] += out["hit_token_limit"]
            n["nonfinite"] += out["nonfinite_logits"]
            audio_seconds += row["audio_seconds"]
        if n["recognised"] % (args.batch_size * 25) < args.batch_size:
            rate = audio_seconds / max(time.time() - t0, 1e-9)
            print(
                f"[{args.arm} {args.split}] {n['recognised']}/{len(rows)}, {rate:.1f}x real time",
                file=sys.stderr,
                flush=True,
            )
    summary = {
        "arm": args.arm,
        "split": args.split,
        **n,
        "seconds": round(time.time() - t0, 1),
        "audio_seconds": round(audio_seconds, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
    }
    print(json.dumps(summary), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
