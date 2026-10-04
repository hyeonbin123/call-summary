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

Speech condition v3 (2026-10-05) compares recognisers on the same audio, in two passes:

    pass 1  --keep-audio [--only-ids]   synthesise every utterance again (the text cache keeps no audio),
            keep the 16-bit 16 kHz WAV the base recogniser T (large-v3-turbo) is given, and T's text, in
            outputs/asr3/audio/ (index.jsonl, one row per turn; --only-ids: only turns whose written text has
            an order, receipt or tracking number, the cheap first gate)
    pass 2  --recognise ARM             an arm on the kept WAVs: Th (T with the domain context as hotwords),
            L (large-v3), N (whisper-ko-ft turbo-n, CTranslate2); same decoding as T. The Qwen3-ASR arms run
            in this project's own environment (scripts/asr_qwen.py). Output outputs/asr3/audio/hyp/<ARM>.jsonl

The datasets each arm gives the student are built by `python -m call_summary.asr_arms build`.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter
from collections.abc import Callable, Container, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

LISTEN_RATE = 16_000
MU = 255.0
SPEEDS = {"상담원": 1.0, "고객": 1.1}  # one Korean voice; a different pace per side
VERBALIZERS = ("v1", "v2")
DIGIT_NAMES = "공일이삼사오육칠팔구"  # as support-agent's verbalize reads phone numbers ("공" for 0)
_TRACKING = re.compile(r"(?<![\d-])(\d{4})-(\d{4})-(\d{4})(?![\d-])")
ID_PATTERN = re.compile(r"D\d{7}|R\d{6}|\d{4}-\d{4}-\d{4}")  # order, receipt, tracking numbers as written
DEFAULT_AUDIO = "outputs/asr3/audio"
TURBO_N = Path(__file__).resolve().parents[2] / "whisper-ko-ft" / "outputs" / "turbo-n" / "ct2"
# Pass-2 arms in faster-whisper: arm -> (model, hotwords = the domain's context). Decoding is T's own.
RECOGNISERS = {
    "T": ("large-v3-turbo", False),
    "Th": ("large-v3-turbo", True),
    "L": ("large-v3", False),
    "N": (str(TURBO_N), False),
}
DECODING = {
    "language": "ko",
    "beam_size": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "vad_filter": False,
}


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


def says_identifier(text: str) -> bool:
    """Whether a written turn has an order, receipt or tracking number (the cheap gate's turns)."""
    return bool(ID_PATTERN.search(unicodedata.normalize("NFKC", text)))


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
        self.device = "cuda"  # the measured condition; "cpu" only for a smoke run (--device cpu)
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.cache[e["key"]] = e

    def _load_models(self) -> None:
        from support_agent.voice.speech import MeloSpeaker, WhisperListener, wav_bytes

        self.wav_bytes = wav_bytes
        self.speaker = MeloSpeaker(device=self.device)
        self.listener = WhisperListener(model=self.whisper_model, device=self.device)

    def plan(self, items: Iterable[dict]) -> dict[str, int]:
        return plan(items, self.verbalize, self._id, self.cache, self.verbalizer)

    def meta(self, written: list[str]) -> dict:
        return {**self.identity, "verbalizer": self.verbalizer, "written": written}

    def _speak(self, spoken: str, speaker: str, seed: int):
        """16 kHz audio of one utterance after the telephone channel (the models are loaded)."""
        import torch

        torch.manual_seed(seed)
        model = self.speaker._model
        samples = model.tts_to_file(spoken, self.speaker._speaker, None, speed=SPEEDS[speaker], quiet=True)
        return telephone(to_16k(samples, self.speaker.sample_rate))

    def key(self, text: str, speaker: str, seed: int) -> tuple[str, str]:
        spoken = spoken_form(text, self.verbalize, self.verbalizer)
        return cache_key(self._id, speaker, seed, spoken), spoken

    def hear(self, text: str, speaker: str, seed: int) -> dict:
        spoken = spoken_form(text, self.verbalize, self.verbalizer)
        if not has_hangul(spoken):
            return {"spoken": spoken, "heard": "", "audio_seconds": 0.0}
        key = cache_key(self._id, speaker, seed, spoken)
        if key in self.cache:
            return self.cache[key]
        if self.speaker is None:
            self._load_models()
        audio = self._speak(spoken, speaker, seed)
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


# --- speech condition v3 -------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class WavStore:
    """Pass 1 output: the WAV bytes each utterance's base recogniser was given, and one index row per turn
    ({key, split, item_id, turn, domain, speaker, seed, spoken, cached_heard, audio_seconds, heard, wav,
    wav_sha256}; `heard` is T's text, `cached_heard` the text cache's for the same key; wav None: silent).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"
        self.rows: dict[str, dict] = {r["key"]: r for r in _read_jsonl(self.index_path)}

    def wav_path(self, key: str) -> Path:
        return self.root / "wav" / key[:2] / f"{key}.wav"

    def add(self, row: dict, wav: bytes | None) -> dict:
        if wav is None:
            row = {**row, "wav": None, "wav_sha256": None}
        else:
            path = self.wav_path(row["key"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wav)
            row = {
                **row,
                "wav": path.relative_to(self.root).as_posix(),
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
            }
        _append_jsonl(self.index_path, row)
        self.rows[row["key"]] = row
        return row

    def split_rows(self, split: str) -> list[dict]:
        return [r for r in self.rows.values() if r["split"] == split]

    def check_identity(self, identity: dict, write: bool = True) -> None:
        """Refuse to mix audio made in different speech setups in one store."""
        path = self.root / "identity.json"
        if path.exists():
            kept = json.loads(path.read_text(encoding="utf-8"))
            if kept != identity:
                raise SystemExit(f"{path} was made with another speech setup: {kept} != {identity}")
            return
        if not write:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(identity, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def turns_to_keep(
    channel: Channel, items: list[dict], only_ids: bool
) -> Iterator[tuple[dict, int, int, str, str]]:
    """(item, turn index, seed, key, spoken text) of every turn pass 1 keeps."""
    for it in items:
        for i, t in enumerate(it["turns"]):
            if only_ids and not says_identifier(t["text"]):
                continue
            seed = seed_of(it["item_id"], i)
            key, spoken = channel.key(t["text"], t["speaker"], seed)
            yield it, i, seed, key, spoken


def keep_plan(channel: Channel, items: list[dict], store: WavStore, only_ids: bool) -> dict[str, int]:
    """What pass 1 would do; `in_text_cache` counts the turns to synthesise whose key the v2 text cache has
    (all of them when the speech setup is the cached one)."""
    n: Counter = Counter()
    for _it, _i, _seed, key, spoken in turns_to_keep(channel, items, only_ids):
        n["turns"] += 1
        if key in store.rows:
            n["done"] += 1
        elif not has_hangul(spoken):
            n["silent"] += 1
        else:
            n["to_synthesise"] += 1
            n["in_text_cache"] += key in channel.cache
    return {k: n[k] for k in ("turns", "done", "silent", "to_synthesise", "in_text_cache")}


def save_wavs(
    channel: Channel, items: list[dict], store: WavStore, split: str, only_ids: bool = False, every: int = 50
) -> Counter:
    """Pass 1: synthesise each turn not yet in the store, keep its WAV and the base recogniser's text."""
    n: Counter = Counter()
    t0 = time.time()
    for it, i, seed, key, spoken in turns_to_keep(channel, items, only_ids):
        t = it["turns"][i]
        if key in store.rows:
            n["done"] += 1
            continue
        row = {
            "key": key,
            "split": split,
            "item_id": it["item_id"],
            "turn": i,
            "domain": it["domain"],
            "speaker": t["speaker"],
            "seed": seed,
            "spoken": spoken,
            "cached_heard": channel.cache.get(key, {}).get("heard"),
        }
        if not has_hangul(spoken):
            store.add({**row, "audio_seconds": 0.0, "heard": ""}, None)
            n["silent"] += 1
            continue
        if channel.speaker is None:
            channel._load_models()
        audio = channel._speak(spoken, t["speaker"], seed)
        wav = channel.wav_bytes(audio, LISTEN_RATE).wav
        heard = channel.listener.transcribe(wav)  # exactly the measured channel
        store.add({**row, "audio_seconds": round(len(audio) / LISTEN_RATE, 3), "heard": heard}, wav)
        n["synthesised"] += 1
        if n["synthesised"] % every == 0:
            rate = (time.time() - t0) / n["synthesised"]
            print(f"[{split}] {n['synthesised']} synthesised, {rate:.2f}s/utt", file=sys.stderr, flush=True)
    return n


def recognise(
    store: WavStore,
    split: str,
    arm: str,
    hyp_path: Path,
    transcribe: Callable[[bytes, str | None], str],
    context: dict[str, str] | None,
    every: int = 100,
) -> Counter:
    """Pass 2: one arm on every kept WAV of the split that is not yet in its output (resumable)."""
    done = {r["key"] for r in _read_jsonl(hyp_path)}
    n: Counter = Counter()
    t0 = time.time()
    for row in store.split_rows(split):
        if row["wav"] is None:
            continue
        if row["key"] in done:
            n["done"] += 1
            continue
        hotwords = context[row["domain"]] if context else None
        heard = transcribe(store.wav_path(row["key"]).read_bytes(), hotwords)
        keep = {k: row[k] for k in ("key", "split", "item_id", "turn")}
        _append_jsonl(hyp_path, {**keep, "arm": arm, "heard": heard})
        n["recognised"] += 1
        if n["recognised"] % every == 0:
            rate = (time.time() - t0) / n["recognised"]
            print(
                f"[{arm} {split}] {n['recognised']} recognised, {rate:.2f}s/utt", file=sys.stderr, flush=True
            )
    return n


def whisper_transcriber(listener) -> Callable[[bytes, str | None], str]:
    """T's own call without hotwords (support_agent WhisperListener.transcribe); with hotwords, the same
    decoding plus faster-whisper's `hotwords`."""

    def run(wav: bytes, hotwords: str | None) -> str:
        if hotwords is None:
            return listener.transcribe(wav)
        from faster_whisper.audio import decode_audio

        samples = decode_audio(io.BytesIO(wav), sampling_rate=LISTEN_RATE)
        segments, _info = listener._model.transcribe(samples, **DECODING, hotwords=hotwords)
        return " ".join(segment.text.strip() for segment in segments).strip()

    return run


def _git_commit() -> str:
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "?"
    return out.strip() + ("+dirty" if dirty.strip() else "")


def _model_files(model: str) -> dict:
    """Where the model came from: a local folder's model.bin hash, or the cached snapshot folder."""
    if os.path.isdir(model):
        h = hashlib.sha256()
        with open(os.path.join(model, "model.bin"), "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        return {"path": Path(model).as_posix(), "model_bin_sha256": h.hexdigest()}
    from faster_whisper.utils import download_model

    return {"path": Path(download_model(model, local_files_only=True)).as_posix()}


def write_hyp_meta(hyp_path: Path, meta: dict) -> None:
    """<hyp>.meta.json: written once; a resumed run must be the same arm and model."""
    path = hyp_path.with_name(hyp_path.name + ".meta.json")
    if path.exists():
        kept = json.loads(path.read_text(encoding="utf-8"))
        same = ("arm", "model", "hotwords", "decoding")
        if any(kept.get(k) != meta.get(k) for k in same):
            raise SystemExit(f"{path} belongs to another arm or model: {kept}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def add_torch_cuda_dlls() -> None:
    """CTranslate2 loads cuBLAS and cuDNN by name. support-agent's environment has them as nvidia-* wheels
    (WhisperListener adds those folders); whisper-ko-ft's environment, where arm N runs, uses the ones that
    ship with torch, as whisper_ko_ft.evaluate_ct2 does."""
    from importlib.util import find_spec

    if os.name != "nt" or find_spec("nvidia") is not None:
        return
    spec = find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return
    lib = os.path.join(next(iter(spec.submodule_search_locations)), "lib")
    if os.path.isdir(lib):
        os.add_dll_directory(lib)
        os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")


def run_pass_two(args) -> int:
    from support_agent.voice.speech import WhisperListener, versions

    add_torch_cuda_dlls()

    model, hotwords = RECOGNISERS[args.recognise]
    store = WavStore(Path(args.audio))
    if not store.split_rows(args.split):
        print(f"no kept audio for {args.split} in {store.root} (run pass 1 first)", file=sys.stderr)
        return 2
    if hotwords and not args.context:
        print(f"{args.recognise} needs --context", file=sys.stderr)
        return 2
    context = json.loads(Path(args.context).read_text(encoding="utf-8")) if hotwords else None
    hyp = Path(args.hyp) if args.hyp else store.root / "hyp" / f"{args.recognise}.jsonl"
    listener = WhisperListener(model=model, device=args.device)
    n_mels = listener._model.model.n_mels
    if listener._model.feature_extractor.mel_filters.shape[0] != n_mels:
        print(f"{model}: the feature extractor does not make {n_mels} mel bins", file=sys.stderr)
        return 2
    write_hyp_meta(
        hyp,
        {
            "arm": args.recognise,
            "model": model,
            "hotwords": hotwords,
            "decoding": DECODING,
            "device": args.device,
            "compute_type": listener.compute_type,
            "mel_bins": n_mels,
            "model_files": _model_files(model),
            "context": context,
            "versions": versions("faster-whisper", "ctranslate2", "tokenizers", "av"),
            "python": sys.executable,
            "commit": _git_commit(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    t0 = time.time()
    n = recognise(store, args.split, args.recognise, hyp, whisper_transcriber(listener), context)
    print(
        json.dumps({"arm": args.recognise, "split": args.split, **n, "seconds": round(time.time() - t0, 1)}),
        file=sys.stderr,
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", required=True)
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--verbalizer", choices=VERBALIZERS, default="v2")
    ap.add_argument("--out", help="output dir (default data/asr for v1, data/asr-<verbalizer> otherwise)")
    ap.add_argument("--cache", default="outputs/asr/cache.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="print the cache plan and stop (no models)")
    ap.add_argument("--audio", default=DEFAULT_AUDIO, help="v3: the kept audio (pass 1 writes, pass 2 reads)")
    ap.add_argument("--keep-audio", action="store_true", help="v3 pass 1: synthesise again and keep the WAVs")
    ap.add_argument(
        "--only-ids", action="store_true", help="v3 pass 1: only turns with an identifier written"
    )
    ap.add_argument("--recognise", choices=list(RECOGNISERS), help="v3 pass 2: this arm on the kept WAVs")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="v3: cpu only for a smoke run")
    ap.add_argument("--hyp", help="v3 pass 2 output (default <audio>/hyp/<ARM>.jsonl)")
    ap.add_argument(
        "--context", help="v3 pass 2: per-domain context JSON (python -m call_summary.asr_arms context)"
    )
    ap.add_argument(
        "--allow-full-run",
        action="store_true",
        help="synthesise utterances that v1 says the same way but the cache lacks "
        "(a new split, or a changed speech setup that re-runs everything)",
    )
    args = ap.parse_args(argv)
    if args.device == "cpu" and Path(args.audio) == Path(DEFAULT_AUDIO):
        ap.error("a CPU run is not the measured condition; give it its own --audio")
    if args.recognise:
        return run_pass_two(args)

    src = Path(args.data) / f"{args.split}.jsonl"
    out_dir = args.out or ("data/asr" if args.verbalizer == "v1" else f"data/asr-{args.verbalizer}")
    out = Path(out_dir) / f"{args.split}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["item_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    items = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    todo = [it for it in items if it["item_id"] not in done][: args.limit]
    channel = Channel(Path(args.cache), verbalizer=args.verbalizer)
    channel.device = args.device
    if args.keep_audio:
        if args.verbalizer != "v2":
            print("v3 keeps the v2 reading", file=sys.stderr)
            return 2
        store = WavStore(Path(args.audio))
        store.check_identity({**channel.identity, "verbalizer": args.verbalizer}, write=not args.dry_run)
        chosen = items[: args.limit]
        plan_ = keep_plan(channel, chosen, store, args.only_ids)
        print(json.dumps({"split": args.split, "items": len(chosen), **plan_}), file=sys.stderr)
        if args.dry_run:
            return 0
        t0 = time.time()
        n = save_wavs(channel, chosen, store, args.split, only_ids=args.only_ids)
        print(json.dumps({"split": args.split, **n, "seconds": round(time.time() - t0, 1)}), file=sys.stderr)
        return 0
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
