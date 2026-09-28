"""Stage 4 input condition: every turn of a dataset spoken by a TTS voice, passed through a narrowband
telephone channel, and written down again by a speech recogniser.

It runs in the speech environment of the sibling project support-agent (MeloTTS needs transformers 4.27,
this project uses 5.x), so it imports nothing from call_summary:

    set PYTHONPATH=D:/coding/support-agent/src
    D:/coding/support-agent/.venv/Scripts/python.exe scripts/asr_condition.py --split test-a

Output: data/asr/<split>.jsonl, the same items with recognised text in `turns` and the written text kept
in meta["asr"]["written"]. Every utterance is cached (outputs/asr/cache.jsonl), so a run can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

LISTEN_RATE = 16_000
MU = 255.0
_BAND = butter(4, [300, 3400], btype="bandpass", fs=LISTEN_RATE, output="sos")
SPEEDS = {"상담원": 1.0, "고객": 1.1}  # one Korean voice; a different pace per side


def mu_law_roundtrip(samples: np.ndarray) -> np.ndarray:
    """Compress to 8-bit mu-law codes and expand back (G.711 style). Same as whisper-ko-ft's channel."""
    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


def telephone(samples: np.ndarray) -> np.ndarray:
    """16 kHz audio after a 300-3400 Hz, 8 kHz, 8-bit mu-law channel, still at 16 kHz."""
    if len(samples) < 64:
        return samples.astype(np.float32)
    band = sosfiltfilt(_BAND, samples)
    narrow = resample_poly(band, 1, 2)
    wide = resample_poly(mu_law_roundtrip(narrow), 2, 1)
    return wide[: len(samples)].astype(np.float32)


def to_16k(samples: np.ndarray, rate: int) -> np.ndarray:
    from math import gcd

    g = gcd(LISTEN_RATE, rate)
    return resample_poly(np.asarray(samples, dtype=np.float32), LISTEN_RATE // g, rate // g).astype(
        np.float32
    )


class Channel:
    def __init__(self, cache_path: Path, whisper_model: str = "large-v3-turbo"):
        from support_agent.voice.speech import MeloSpeaker, WhisperListener, versions, wav_bytes
        from support_agent.voice.verbalize import verbalize

        self.verbalize, self.wav_bytes = verbalize, wav_bytes
        self.speaker = MeloSpeaker(device="cuda")
        self.listener = WhisperListener(model=whisper_model, device="cuda")
        self.identity = {
            "tts": "melotts/KR",
            "speeds": SPEEDS,
            "channel": "telephone 300-3400 Hz, 8 kHz, 8-bit mu-law",
            "stt": f"faster-whisper/{whisper_model}, beam 5, temperature 0, ko",
            "versions": versions("melotts", "torch", "faster-whisper", "ctranslate2"),
        }
        self._id = json.dumps(self.identity, sort_keys=True, ensure_ascii=False)
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.cache[e["key"]] = e

    def hear(self, text: str, speaker: str, seed: int) -> dict:
        spoken = self.verbalize(text)
        if not any("가" <= ch <= "힣" for ch in spoken):
            return {"spoken": spoken, "heard": "", "audio_seconds": 0.0}
        key = hashlib.sha256(f"{self._id}|{speaker}|{seed}|{spoken}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        import torch

        torch.manual_seed(seed)
        model = self.speaker._model
        samples = model.tts_to_file(spoken, self.speaker._speaker, None, speed=SPEEDS[speaker], quiet=True)
        audio = telephone(to_16k(samples, self.speaker.sample_rate))
        heard = self.listener.transcribe(self.wav_bytes(audio, LISTEN_RATE).wav)
        entry = {
            "key": key,
            "spoken": spoken,
            "heard": heard,
            "audio_seconds": round(len(audio) / LISTEN_RATE, 3),
        }
        self.cache[key] = entry
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--out", default="data/asr")
    ap.add_argument("--cache", default="outputs/asr/cache.jsonl")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    src = Path(args.data) / f"{args.split}.jsonl"
    out = Path(args.out) / f"{args.split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["item_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    items = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    todo = [it for it in items if it["item_id"] not in done][: args.limit]
    print(f"{len(todo)} items to hear ({len(done)} done)", file=sys.stderr, flush=True)
    channel = Channel(Path(args.cache))
    t0 = time.time()
    n_utt = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for n, it in enumerate(todo, 1):
            written = [t["text"] for t in it["turns"]]
            heard = []
            for i, t in enumerate(it["turns"]):
                seed = zlib.crc32(f"{it['item_id']}:{i}".encode())
                heard.append(channel.hear(t["text"], t["speaker"], seed)["heard"])
                n_utt += 1
            it["turns"] = [
                {"speaker": t["speaker"], "text": h} for t, h in zip(it["turns"], heard, strict=True)
            ]
            it["split"] = f"{it['split']}-asr"
            it.setdefault("meta", {})["asr"] = {**channel.identity, "written": written}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / max(n_utt, 1)
                print(f"[{n}/{len(todo)}] {n_utt} utterances, {rate:.2f}s/utt", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
