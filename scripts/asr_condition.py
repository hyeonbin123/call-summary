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

# [파일 역할] 4단계 음성 조건 데이터를 만드는 스크립트다. 대화의 발화마다 발음형으로 풀어 쓰기 → MeloTTS
# 한국어 합성 → 전화 채널(300~3400 Hz, 8 kHz, 8비트 μ-law) → faster-whisper large-v3-turbo 인식을 거쳐, 같은
# 건의 전사를 인식된 글로 바꾼 데이터(dev-asr, test-d, train-half-asr)를 쓴다. 음성 조건 v3의 1패스(소리
# 남기기)와 faster-whisper 팔의 2패스(다시 듣기)도 여기서 한다.
# [왜 필요한가] 실제 상담은 음성 인식을 거친 전사로 들어온다. 인식 오류가 섞인 전사에서 얼마나 떨어지는지,
# 그런 전사를 섞어 학습하면 나아지는지를 재려고 만든 입력 조건이다(docs/plan.md 질문 4, docs/experiments.md
# 4단계).
# [설계 이유] MeloTTS가 transformers 4.27을 요구해 이 프로젝트 환경(5.x)이 아니라 support-agent 저장소의 음성
# 환경에서 돌고, 그래서 call_summary를 하나도 import하지 않는다(docstring, README). 발화마다 고정 seed로
# 합성하고, 결과를 (음성 환경, 화자, seed, 읽을 글)의 해시로 캐시해 중간에 멈춰도 이어 하고, 읽는
# 법(verbalizer)을 바꾸면 달라진 발화만 다시 합성한다.
# [판] v1(4단계): support-agent의 verbalize만. v2(2026-10-03): 운송장번호를 숫자 하나씩 읽도록 고침(v1은 범위
# "에서"와 기수로 읽은 측정 인공물이었다). v3(2026-10-05): 같은 소리로 인식기를 비교하는 두 패스.

from __future__ import annotations

# zlib: 발화마다 crc32로 seed를 만든다(파이썬 hash()와 달리 실행마다 같다). hashlib: 캐시 키와 WAV 지문.
# functools.cache: 전화 대역 필터를 한 번만 설계한다. io: WAV 바이트를 파일처럼 넘긴다.
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

# LISTEN_RATE: 인식기 입력 표본화율(16 kHz). MU: μ-law 압축 강도(G.711의 255).
# SPEEDS: 한국어 목소리 하나로 상담원 1.0, 고객 1.1배속으로 읽어 두 화자를 조금 다르게 한다(4단계 규칙).
# DIGIT_NAMES: v2가 운송장번호를 읽을 숫자 이름. 0은 support-agent의 verbalize가 전화번호를 읽는 방식처럼
# "공"이다.
LISTEN_RATE = 16_000
MU = 255.0
SPEEDS = {"상담원": 1.0, "고객": 1.1}  # one Korean voice; a different pace per side
VERBALIZERS = ("v1", "v2")
DIGIT_NAMES = "공일이삼사오육칠팔구"  # as support-agent's verbalize reads phone numbers ("공" for 0)
# _TRACKING: 운송장번호 꼴(4-4-4). 앞뒤에 숫자나 하이픈이 붙어 더 긴 수의 일부인 경우는 잡지 않는다. 이 꼴이
# 아닌 하이픈 숫자(주소 "761-4" 등)는 v2 규칙의 범위 밖이다(docs/experiments.md 음성 조건 v2).
# ID_PATTERN: 쓴 글의 주문번호(D+7), 접수번호(R+6), 운송장번호. v3 관문은 이 번호가 든 발화만 먼저 합성한다.
_TRACKING = re.compile(r"(?<![\d-])(\d{4})-(\d{4})-(\d{4})(?![\d-])")
ID_PATTERN = re.compile(r"D\d{7}|R\d{6}|\d{4}-\d{4}-\d{4}")  # order, receipt, tracking numbers as written
# DEFAULT_AUDIO: v3 1패스가 소리를 남기는 곳(커밋하지 않음). TURBO_N: 형제 저장소 whisper-ko-ft에서 학습해
# CTranslate2로 바꾼 turbo-n(N 팔)의 위치를 이 저장소 기준 상대 경로로 찾는다.
DEFAULT_AUDIO = "outputs/asr3/audio"
TURBO_N = Path(__file__).resolve().parents[2] / "whisper-ko-ft" / "outputs" / "turbo-n" / "ct2"
# RECOGNISERS: faster-whisper로 도는 팔 → (모델, 업종 맥락을 hotwords로 줄지). T는 기준(large-v3-turbo), Th는
# T + 카탈로그 hotwords, L은 large-v3, N은 turbo-n. Qwen3-ASR 팔은 scripts/asr_qwen.py가 따로 돈다.
# DECODING: 모든 팔이 같은 디코딩(한국어, beam 5, temperature 0, 이전 글 조건 끔, VAD 끔). v2의 T와 같아
# 인식기만 바꾼 비교가 된다(docs/experiments.md 음성 조건 v3 팔 표).
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


# v2 규칙: 운송장번호를 숫자 하나씩, 묶음 사이에 쉼표를 두어 읽는 꼴로 바꾼다("6291-7877-5168로" → "육이구일,
# 칠팔칠칠, 오일육팔 로"). verbalize가 처음에 하는 것과 같은 NFKC 정규화 뒤에 찾는다. 번호가 없는 글은 한
# 글자도 바꾸지 않고 그대로 돌려줘 v1과 같은 소리·같은 캐시를 쓰게 한다(tests/test_asr_condition.py가 확인).
def spell_tracking_ids(text: str) -> str:
    """'6291-7877-5168로' -> ' 육이구일, 칠팔칠칠, 오일육팔 로'. Text without one is returned unchanged."""
    norm = unicodedata.normalize("NFKC", text)  # verbalize() starts with the same step
    if not _TRACKING.search(norm):
        return text
    return _TRACKING.sub(
        lambda m: " " + ", ".join("".join(DIGIT_NAMES[int(c)] for c in g) for g in m.groups()) + " ", norm
    )


# 판(v1/v2)에 따라 합성할 글(발음형)을 만든다. v1은 verbalize 그대로, v2는 운송장번호를 먼저 풀어 쓴 뒤
# verbalize.
def spoken_form(text: str, verbalize: Callable[[str], str], verbalizer: str) -> str:
    if verbalizer == "v1":
        return verbalize(text)
    if verbalizer == "v2":
        return verbalize(spell_tracking_ids(text))
    raise ValueError(f"unknown verbalizer {verbalizer!r}")


# 쓴 발화에 번호가 있는지(v3 관문에서 합성할 발화를 고른다).
def says_identifier(text: str) -> bool:
    """Whether a written turn has an order, receipt or tracking number (the cheap gate's turns)."""
    return bool(ID_PATTERN.search(unicodedata.normalize("NFKC", text)))


# 발음형에 한글이 하나도 없으면 합성할 말이 없는 발화(silent)로 보고 소리 없이 빈 글로 둔다.
def has_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


# 모든 건의 (건 id, 발화 번호, 화자, 쓴 글)을 차례로 내준다.
def utterances(items: Iterable[dict]) -> Iterator[tuple[str, int, str, str]]:
    for it in items:
        for i, t in enumerate(it["turns"]):
            yield it["item_id"], i, t["speaker"], t["text"]


# 발화마다 고정 seed("건id:발화번호"의 crc32). 다시 돌려도 같은 발화는 같은 seed로 합성된다.
def seed_of(item_id: str, index: int) -> int:
    return zlib.crc32(f"{item_id}:{index}".encode())


# 음성 환경(합성기, 속도, 채널, 인식기, 라이브러리 버전)을 정렬된 JSON 문자열로 만들어 캐시 키의 일부로 쓴다.
# 버전이 바뀌면 키가 달라져 예전 캐시를 잘못 쓰지 않는다.
def identity_key(identity: dict) -> str:
    return json.dumps(identity, sort_keys=True, ensure_ascii=False)


# 캐시 키 = sha256(음성 환경 | 화자 | seed | 읽을 글). verbalizer 이름은 넣지 않는다: v1과 v2가 같은 글을 읽는
# 발화는 같은 키라 v2가 그 소리를 다시 합성하지 않는다(테스트가 확인).
def cache_key(identity: str, speaker: str, seed: int, spoken: str) -> str:
    return hashlib.sha256(f"{identity}|{speaker}|{seed}|{spoken}".encode()).hexdigest()


# 모델 없이 이번 실행이 무엇을 합성할지 센다(--dry-run): 발화 수, 소리 없는 발화, v1과 다르게 읽는 발화,
# 캐시에 있는 것, 새로 합성할 것.
# unchanged_not_cached가 0보다 크면 v1과 똑같이 읽는데 캐시에 없다는 뜻이라, 음성 환경(라이브러리 버전 등)이
# 캐시를 만들 때와 달라진 것이다. main은 이 경우 멈춘다.
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


# 8비트 μ-law로 압축 → 256단계 양자화 → 복원을 한 번 거친 파형. whisper-ko-ft의 전화 채널과 같은 식이다(4단계
# 규칙). numpy·scipy는 쓰는 함수 안에서 불러온다.
def mu_law_roundtrip(samples):
    """Compress to 8-bit mu-law codes and expand back (G.711 style). Same as whisper-ko-ft's channel."""
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    compressed = np.sign(clipped) * np.log1p(MU * np.abs(clipped)) / np.log1p(MU)
    codes = np.round((compressed + 1.0) / 2.0 * 255.0)
    restored = codes / 255.0 * 2.0 - 1.0
    return (np.sign(restored) * np.expm1(np.abs(restored) * np.log1p(MU)) / MU).astype(np.float32)


# 300~3400 Hz 4차 Butterworth 대역 통과 필터(sos 형식). 처음 쓸 때 한 번만 설계해 캐시한다.
@functools.cache
def _band():
    from scipy.signal import butter

    return butter(4, [300, 3400], btype="bandpass", fs=LISTEN_RATE, output="sos")


# 16 kHz 소리를 전화 채널에 통과시킨다: 대역 제한(앞뒤로 걸러 위상 지연 0) → 8 kHz로 낮춤 → μ-law 왕복 → 16
# kHz로 되돌림. 인식기 입력을 16 kHz로 맞추되 소리는 좁은 대역 전화 음질이 된다. 너무 짧은 소리는 필터가 덧댈
# 길이가 모자라 그대로 둔다.
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


# 합성기의 표본화율을 16 kHz로 바꾼다(최대공약수로 정수비를 만들어 resample_poly).
def to_16k(samples, rate: int):
    from math import gcd

    import numpy as np
    from scipy.signal import resample_poly

    g = gcd(LISTEN_RATE, rate)
    return resample_poly(np.asarray(samples, dtype=np.float32), LISTEN_RATE // g, rate // g).astype(
        np.float32
    )


# 캐시된 발화가 의존하는 음성 환경: 합성기, 속도, 채널, 인식기와 디코딩, 그리고
# melotts·torch·faster-whisper·ctranslate2 버전. 이 dict가 캐시 키와 데이터의 meta.asr에 들어간다(4단계 규칙
# "버전은 각 건의 meta.asr에 남긴다").
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


# support-agent의 발음형 변환(verbalize)을 불러온다. 숫자·기호를 한국어 읽는 법으로 풀어 써야 TTS가 읽을 수
# 있다.
def load_verbalize() -> Callable[[str], str]:
    from support_agent.voice.verbalize import verbalize

    return verbalize


# 발화 하나를 말하고(합성 + 전화 채널) 듣는(인식) 통로. 모델은 처음 캐시에 없는 발화를 만날 때 올리므로, 모두
# 캐시에 있으면 GPU 없이 돈다. 캐시 파일(outputs/asr/cache.jsonl)을 처음에 전부 읽어 둔다.
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
        # 측정 조건은 cuda다. cpu는 스모크용이며 main이 별도 --audio를 요구해 측정 결과와 섞이지 않게 한다.
        self.device = "cuda"  # the measured condition; "cpu" only for a smoke run (--device cpu)
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                self.cache[e["key"]] = e

    # support-agent의 MeloSpeaker(합성)와 WhisperListener(인식)를 올린다. support-agent 음성 채널과 같은
    # 구성이라 그쪽 음성 조건과 비교할 수 있다(docs/experiments.md "계획과 달라진 것").
    def _load_models(self) -> None:
        from support_agent.voice.speech import MeloSpeaker, WhisperListener, wav_bytes

        self.wav_bytes = wav_bytes
        self.speaker = MeloSpeaker(device=self.device)
        self.listener = WhisperListener(model=self.whisper_model, device=self.device)

    def plan(self, items: Iterable[dict]) -> dict[str, int]:
        return plan(items, self.verbalize, self._id, self.cache, self.verbalizer)

    # 데이터 한 건의 meta.asr: 음성 환경, verbalizer 판, 원래 쓴 글 목록(written). asr_report.tracking-forms와
    # asr_arms가 written을 읽는다.
    def meta(self, written: list[str]) -> dict:
        return {**self.identity, "verbalizer": self.verbalizer, "written": written}

    # 발화 하나를 합성해 전화 채널을 거친 16 kHz 소리로 돌려준다. torch.manual_seed(seed)로 MeloTTS의
    # 무작위성을 발화마다 고정한다(다만 GPU 합성이 비트 단위로 같다는 보장은 없어, v3에서 다시 합성한 소리의 T
    # 글이 3,157발화 중 24개 달랐다).
    def _speak(self, spoken: str, speaker: str, seed: int):
        """16 kHz audio of one utterance after the telephone channel (the models are loaded)."""
        import torch

        torch.manual_seed(seed)
        model = self.speaker._model
        samples = model.tts_to_file(spoken, self.speaker._speaker, None, speed=SPEEDS[speaker], quiet=True)
        return telephone(to_16k(samples, self.speaker.sample_rate))

    # 쓴 글에서 (캐시 키, 읽을 글)을 만든다.
    def key(self, text: str, speaker: str, seed: int) -> tuple[str, str]:
        spoken = spoken_form(text, self.verbalize, self.verbalizer)
        return cache_key(self._id, speaker, seed, spoken), spoken

    # 발화 하나를 듣는다: 소리 없는 발화는 빈 글, 캐시에 있으면 캐시, 없으면 합성 → 인식하고 캐시 파일에 한 줄
    # 덧붙인다(이어 하기).
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


# JSONL 읽기(파일이 없으면 빈 목록). call_summary를 import하지 않으려고 이 스크립트 안에 따로 둔다.
# asr_qwen.py도 이것을 가져다 쓴다.
def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# JSONL에 한 줄 덧붙인다(폴더가 없으면 만든다). 한 발화마다 바로 써서 멈춰도 이어 할 수 있다.
def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# v3 1패스의 저장소: 기준 인식기 T가 들은 WAV 바이트 그대로와, 발화마다 index 한 줄. 모든 팔이 이 같은
# 바이트를 들으므로 팔끼리 소리가 같은지 따질 일이 없다(docs/experiments.md 음성 조건 v3 "소리"). 8 kHz μ-law
# 대신 16 kHz WAV로 둔 이유도 같은 절에 있다.
class WavStore:
    """Pass 1 output: the WAV bytes each utterance's base recogniser was given, and one index row per turn
    ({key, split, item_id, turn, domain, speaker, seed, spoken, cached_heard, audio_seconds, heard, wav,
    wav_sha256}; `heard` is T's text, `cached_heard` the text cache's for the same key; wav None: silent).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index_path = self.root / "index.jsonl"
        self.rows: dict[str, dict] = {r["key"]: r for r in _read_jsonl(self.index_path)}

    # 키 앞 두 글자로 하위 폴더를 나눠 한 폴더에 파일이 수천 개 몰리지 않게 한다.
    def wav_path(self, key: str) -> Path:
        return self.root / "wav" / key[:2] / f"{key}.wav"

    # index 행을 저장한다. 소리가 있으면 WAV를 쓰고 상대 경로와 sha256을, 없으면(silent) None을 적는다.
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

    # 한 저장소에 다른 음성 환경에서 만든 소리가 섞이지 않게, 처음 만든 환경(identity.json)과 지금 환경이
    # 다르면 멈춘다. dry-run에서는 파일을 새로 쓰지 않는다.
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


# 1패스가 다룰 발화마다 (건, 발화 번호, seed, 캐시 키, 읽을 글). --only-ids면 번호가 든 발화만(관문).
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


# 1패스 dry-run 계획: 이미 저장한 것, 소리 없는 것, 새로 합성할 것, 그중 v2 텍스트 캐시에 키가 있는 것. 음성
# 환경이 캐시와 같다면 합성할 발화의 키가 모두 캐시에 있어야 한다(v3 규칙의 확인).
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


# 1패스: 저장소에 없는 발화마다 다시 합성하고, T가 받은 WAV와 T의 글을 저장한다. 같은 키의 v2 캐시
# 글(cached_heard)도 함께 적어, 다시 합성한 소리의 T 글이 예전 전사와 같은지(asr_arms identity) 확인할 수 있게
# 한다.
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
        # 측정 조건 그대로의 호출로 T의 글을 얻는다(2패스 팔들은 이 WAV 바이트를 다시 듣는다).
        heard = channel.listener.transcribe(wav)  # exactly the measured channel
        store.add({**row, "audio_seconds": round(len(audio) / LISTEN_RATE, 3), "heard": heard}, wav)
        n["synthesised"] += 1
        if n["synthesised"] % every == 0:
            rate = (time.time() - t0) / n["synthesised"]
            print(f"[{split}] {n['synthesised']} synthesised, {rate:.2f}s/utt", file=sys.stderr, flush=True)
    return n


# 2패스: 팔 하나로 그 분할의 저장된 WAV를 모두 다시 듣는다. 이미 hyp에 있는 키는 건너뛴다(이어 하기). 맥락을
# 쓰는 팔은 발화의 업종에 맞는 맥락 글을 hotwords로 넘긴다.
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


# faster-whisper 팔의 인식 함수. hotwords가 없으면 T와 똑같이 WhisperListener.transcribe를 부르고, 있으면(Th)
# 같은 디코딩(DECODING)에 hotwords만 더해 모델을 직접 부른다. 조각들의 글을 공백으로 잇는다.
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


# 지금 커밋 해시. 추적 파일이 바뀌어 있으면 "+dirty"를 붙인다. git이 없으면 "?". hyp 메타에 남긴다.
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


# 모델 출처 기록: 로컬 폴더(turbo-n)면 model.bin의 sha256(16 MB씩 읽어 메모리를 아낌), 허브 모델이면 내려받지
# 않고(local_files_only) 캐시된 스냅샷 경로.
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


# hyp 옆의 .meta.json을 처음 한 번만 쓴다. 이어 돌릴 때 팔·모델·hotwords·디코딩이 다르면 다른 결과가 한 파일에
# 섞이므로 멈춘다. asr_qwen.py도 이 함수를 쓴다.
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


# Windows에서 CTranslate2가 cuBLAS·cuDNN DLL을 찾게 한다. support-agent 환경은 nvidia-* 휠이 있어 아무것도
# 하지 않고, N 팔이 도는 whisper-ko-ft 환경(tokenizers 버전 문제로 N은 그 환경에서 돈다, docs/experiments.md
# 음성 조건 v3)에서는 torch에 든 lib 폴더를 DLL 경로에 더한다.
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


# 2패스 실행: 저장된 소리가 없거나 맥락 팔에 --context가 없으면 멈춘다. 모델을 올린 뒤 특징 추출기의 멜 필터
# 수가 모델의 n_mels와 맞는지 확인한다: whisper-ko-ft의 6단계 모델을 faster-whisper로 돌린 첫 시도가 멜 필터
# 수가 맞지 않아 모든 발화에서 실패한 적이 있어(docs/experiments.md "계획과 달라진 것") 미리 막는다. 그 뒤 hyp
# 메타(모델 파일 지문, 버전, 파이썬 경로, 커밋)를 쓰고 recognise를 돈다.
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


# 명령줄 진입점. 세 갈래: --recognise(v3 2패스), --keep-audio(v3 1패스), 그 밖(4단계·v2 데이터 만들기). 기본
# 판은 v2다.
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
    # CPU 실행은 측정 조건이 아니므로 기본 저장소에 섞이지 않게 별도 --audio를 요구한다.
    if args.device == "cpu" and Path(args.audio) == Path(DEFAULT_AUDIO):
        ap.error("a CPU run is not the measured condition; give it its own --audio")
    if args.recognise:
        return run_pass_two(args)

    src = Path(args.data) / f"{args.split}.jsonl"
    out_dir = args.out or ("data/asr" if args.verbalizer == "v1" else f"data/asr-{args.verbalizer}")
    out = Path(out_dir) / f"{args.split}.jsonl"
    # 출력 파일에 이미 있는 건은 건너뛴다(이어 하기).
    done = set()
    if out.exists():
        done = {json.loads(line)["item_id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    items = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line]
    todo = [it for it in items if it["item_id"] not in done][: args.limit]
    channel = Channel(Path(args.cache), verbalizer=args.verbalizer)
    channel.device = args.device
    # v3 1패스: v2 읽기만 허용한다(v3은 v2 조건 그대로 인식기만 바꾸는 비교라서). 저장소의 음성 환경을
    # 확인하고 계획을 찍은 뒤, dry-run이 아니면 합성해 저장한다.
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
    # 4단계·v2 데이터: 먼저 계획을 찍고, v1과 똑같이 읽는데 캐시에 없는 발화가 있으면 음성 환경이 달라진
    # 것으로 보고 멈춘다(--allow-full-run으로만 넘어감). 환경이 바뀐 채 일부만 다시 합성해 조건이 섞이는 것을
    # 막는다.
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
            # 건마다 모든 발화를 듣고, turns의 글을 인식된 글로 바꾸고, 분할 이름에 "-asr"을 붙이고(평가가
            # spoken 채점 규칙을 쓰게 함), 원래 글은 meta.asr.written에 남긴다. 정답(명세)과 참고 요약은
            # 그대로다.
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
