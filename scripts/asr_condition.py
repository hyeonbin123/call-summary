"""Stage 4 input condition: every turn of a dataset spoken by a TTS voice, passed through a narrowband
telephone channel, and written down again by a speech recogniser.

It runs in the speech environment of the sibling project support-agent (MeloTTS needs transformers 4.27,
this project uses 5.x), so it imports nothing from call_summary:

    set PYTHONPATH=D:/coding/support-agent/src
    D:/coding/support-agent/.venv/Scripts/python.exe scripts/asr_condition.py --split test-a

Output: data/asr-<verbalizer>/<split>.jsonl (v1: data/asr/), the same items with recognised text in `turns`
and the written text kept in meta["asr"]["written"]. Every utterance is cached (outputs/asr/cache.jsonl) by
the speech setup and the spoken text, so a run can resume and a new verbalizer only re-synthesises the
utterances it says differently. `--dry-run` reports that plan without loading a model.

Verbalizer v1 = support-agent's verbalize() alone (stage 4). v2 (2026-10-03) first spells tracking numbers
(\\d{4}-\\d{4}-\\d{4}) digit by digit with a pause between groups, as verbalize() already does for phone
numbers; v1 read them as cardinals joined by a range "에서".
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter
from collections.abc import Callable, Container, Iterable, Iterator
from pathlib import Path

LISTEN_RATE = 16_000
MU = 255.0
SPEEDS = {"상담원": 1.0, "고객": 1.1}  # one Korean voice; a different pace per side
VERBALIZERS = ("v1", "v2")
DIGIT_NAMES = "공일이삼사오육칠팔구"  # as support-agent's verbalize reads phone numbers ("공" for 0)
_TRACKING = re.compile(r"(?<![\d-])(\d{4})-(\d{4})-(\d{4})(?![\d-])")


# --- what is said --------------------------------------------------------------------------------------


def spell_tracking_ids(text: str) -> str:
    """'6291-7877-5168로' -> ' 육이구일, 칠팔칠칠, 오일육팔 로'. Text without one is returned unchanged."""
    norm = unicodedata.normalize("NFKC", text)  # verbalize() starts with the same step
    if not _TRACKING.search(norm):
        return text
    return _TRACKING.sub(
        lambda m: " " + ", ".join("".join(DIGIT_NAMES[int(c)] for c in g) for g in m.groups()) + " ", norm
    )


def spoken_form(text: str, verbalize: Callable[[str], str], verbalizer: str) -> str:
    if verbalizer == "v1":
        return verbalize(text)
    if verbalizer == "v2":
        return verbalize(spell_tracking_ids(text))
    raise ValueError(f"unknown verbalizer {verbalizer!r}")


def has_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


def utterances(items: Iterable[dict]) -> Iterator[tuple[str, int, str, str]]:
    for it in items:
        for i, t in enumerate(it["turns"]):
            yield it["item_id"], i, t["speaker"], t["text"]


def seed_of(item_id: str, index: int) -> int:
    return zlib.crc32(f"{item_id}:{index}".encode())


def identity_key(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, ensure_ascii=False)


def cache_key(identity: str, speaker: str, seed: int, spoken: str) -> str:
    return hashlib.sha256(f"{identity}|{speaker}|{seed}|{spoken}".encode()).hexdigest()


def plan(
    items: Iterable[dict],
    verbalize: Callable[[str], str],
    identity: str,
    cached: Container[str],
    verbalizer: str,
) -> dict[str, int]:
    """What a run would synthesise, by cache state. `changed` = said differently from v1;
    `unchanged_not_cached` > 0 means the speech setup no longer matches the cache."""
    n: Counter = Counter()
    for item_id, i, speaker, text in utterances(items):
        n["utterances"] += 1
        v1 = spoken_form(text, verbalize, "v1")
        spoken = v1 if verbalizer == "v1" else spoken_form(text, verbalize, verbalizer)
        if not has_hangul(spoken):
            n["silent"] += 1
            continue
        changed = spoken != v1
        n["changed"] += changed
        if cache_key(identity, speaker, seed_of(item_id, i), spoken) in cached:
            n["cached"] += 1
        else:
            n["to_synthesise"] += 1
            n["unchanged_not_cached"] += not changed
    keys = ("utterances", "silent", "changed", "cached", "to_synthesise", "unchanged_not_cached")
    return {k: n[k] for k in keys}


# --- the channel ----------------------------------------------------------------------------------------


def mu_law_roundtrip(samples):
    """Compress to 8-bit mu-law codes and expand back (G.711 style). Same as whisper-ko-ft's channel."""
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


@functools.cache
def _band():
    from scipy.signal import butter

    return butter(4, [300, 3400], btype="bandpass", fs=LISTEN_RATE, output="sos")


def telephone(samples):
    """16 kHz audio after a 300-3400 Hz, 8 kHz, 8-bit mu-law channel, still at 16 kHz."""
    import numpy as np
    from scipy.signal import resample_poly, sosfiltfilt

    if len(samples) < 64:
        return samples.astype(np.float32)
    band = sosfiltfilt(_band(), samples)
    narrow = resample_poly(band, 1, 2)
    wide = resample_poly(mu_law_roundtrip(narrow), 2, 1)
    return wide[: len(samples)].astype(np.float32)


def to_16k(samples, rate: int):
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    g = gcd(LISTEN_RATE, rate)
    return resample_poly(np.asarray(samples, dtype=np.float32), LISTEN_RATE // g, rate // g).astype(
        np.float32
    )


def speech_identity(whisper_model: str) -> dict:
    """The speech setup a cached utterance depends on (part of the cache key; the verbalizer is not)."""
    from support_agent.voice.speech import versions

    return {
        "tts": "melotts/KR",
        "speeds": SPEEDS,
        "channel": "telephone 300-3400 Hz, 8 kHz, 8-bit mu-law",
        "stt": f"faster-whisper/{whisper_model}, beam 5, temperature 0, ko",
        "versions": versions("melotts", "torch", "faster-whisper", "ctranslate2"),
    }


def load_verbalize() -> Callable[[str], str]:
    from support_agent.voice.verbalize import verbalize

    return verbalize


class Channel:
    """Speaks and hears one utterance at a time. The models load at the first cache miss."""

    def __init__(
        self,
        cache_path: Path,
        whisper_model: str = "large-v3-turbo",
        verbalizer: str = "v2",
        verbalize: Callable[[str], str] | None = None,
        identity: dict | None = None,
    ):
        if verbalizer not in VERBALIZERS:
            raise ValueError(f"unknown verbalizer {verbalizer!r}")
        self.whisper_model, self.verbalizer = whisper_model, verbalizer
        self.verbalize = verbalize or load_verbalize()
        self.identity = identity if identity is not None else speech_identity(whisper_model)
        self._id = identity_key(self.identity)
        self.speaker = self.listener = self.wav_bytes = None
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.cache[e["key"]] = e

    def _load_models(self) -> None:
        from support_agent.voice.speech import MeloSpeaker, WhisperListener, wav_bytes

        self.wav_bytes = wav_bytes
        self.speaker = MeloSpeaker(device="cuda")
        self.listener = WhisperListener(model=self.whisper_model, device="cuda")

    def plan(self, items: Iterable[dict]) -> dict[str, int]:
        return plan(items, self.verbalize, self._id, self.cache, self.verbalizer)

    def meta(self, written: list[str]) -> dict:
        return {**self.identity, "verbalizer": self.verbalizer, "written": written}

    def hear(self, text: str, speaker: str, seed: int) -> dict:
        spoken = spoken_form(text, self.verbalize, self.verbalizer)
        if not has_hangul(spoken):
            return {"spoken": spoken, "heard": "", "audio_seconds": 0.0}
        key = cache_key(self._id, speaker, seed, spoken)
        if key in self.cache:
            return self.cache[key]
        if self.speaker is None:
            self._load_models()
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
    ap.add_argument("--verbalizer", choices=VERBALIZERS, default="v2")
    ap.add_argument("--out", help="output dir (default data/asr for v1, data/asr-<verbalizer> otherwise)")
    ap.add_argument("--cache", default="outputs/asr/cache.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print the cache plan and stop (no models)")
    ap.add_argument(
        "--allow-full-run",
        action="store_true",
        help="synthesise utterances that v1 says the same way but the cache lacks "
        "(a new split, or a changed speech setup that re-runs everything)",
    )
    args = ap.parse_args(argv)

    src = Path(args.data) / f"{args.split}.jsonl"
    out_dir = args.out or ("data/asr" if args.verbalizer == "v1" else f"data/asr-{args.verbalizer}")
    out = Path(out_dir) / f"{args.split}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["item_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    items = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    todo = [it for it in items if it["item_id"] not in done][: args.limit]
    channel = Channel(Path(args.cache), verbalizer=args.verbalizer)
    p = channel.plan(todo)
    report = {"split": args.split, "verbalizer": args.verbalizer, "items": len(todo), "done": len(done), **p}
    print(json.dumps(report, ensure_ascii=False), file=sys.stderr, flush=True)
    if args.dry_run:
        return 0
    if p["unchanged_not_cached"] and not args.allow_full_run:
        print(
            f"refusing: {p['unchanged_not_cached']} utterances said as in v1 are not in the cache, so the "
            "speech setup differs from the cached one (versions?). Check, or pass --allow-full-run.",
            file=sys.stderr,
        )
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_utt = 0
    with out.open("a", encoding="utf-8", newline="\n") as f:
        for n, it in enumerate(todo, 1):
            written = [t["text"] for t in it["turns"]]
            heard = [
                channel.hear(t["text"], t["speaker"], seed_of(it["item_id"], i))["heard"]
                for i, t in enumerate(it["turns"])
            ]
            n_utt += len(heard)
            it["turns"] = [
                {"speaker": t["speaker"], "text": h} for t, h in zip(it["turns"], heard, strict=True)
            ]
            it["split"] = f"{it['split']}-asr"
            it.setdefault("meta", {})["asr"] = channel.meta(written)
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                rate = (time.time() - t0) / max(n_utt, 1)
                print(f"[{n}/{len(todo)}] {n_utt} utterances, {rate:.2f}s/utt", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
