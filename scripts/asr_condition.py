"""4단계 입력 조건: 데이터셋의 모든 발화를 TTS 목소리로 읽고, 좁은 대역의 전화 채널을 거친 뒤, 음성
인식기로 다시 받아 적는다.

형제 프로젝트 support-agent의 음성 환경에서 돌므로(MeloTTS는 transformers 4.27이 필요하고 이 프로젝트는
5.x를 쓴다) call_summary에서 아무것도 가져오지 않는다:

    set PYTHONPATH=../support-agent/src
    ../support-agent/.venv/Scripts/python.exe scripts/asr_condition.py --split test-a

출력: data/asr-<verbalizer>/<split>.jsonl(v1: data/asr/). 같은 항목에서 `turns`를 인식된 글로 바꾸고 쓴
글은 meta["asr"]["written"]에 남긴다. 모든 발화를 음성 설정과 발음형 글로 캐시하므로(outputs/asr/cache.jsonl)
실행을 이어 할 수 있고, 새 verbalizer는 다르게 읽는 발화만 다시 합성한다. `--dry-run`은 모델을 올리지 않고
그 계획만 보고한다.

Verbalizer v1 = support-agent의 verbalize()만 쓴다(4단계). v2(2026-10-03)는 먼저 운송장번호
(\\d{4}-\\d{4}-\\d{4})를 묶음 사이에 쉼을 두고 숫자 하나씩 읽게 풀어 쓴다. verbalize()가 전화번호를 이미
그렇게 읽는 것과 같다. v1은 운송장번호를 수사로 읽고 묶음을 범위 "에서"로 이었다.

verbalize()는 support-agent의 현재 파일이 아니라, 4단계 때의 파일(support-agent 커밋 a5717ce)을 옮긴 사본
scripts/verbalize_a5717ce.py다(2026-10-14부터). support-agent가 그 뒤 운송장번호 읽기를 고쳐, 현재 파일로는
v1이 4단계 읽기를 재현하지 못한다. 사본의 본문이 원본과 다르면 멈춘다.

음성 조건 v3(2026-10-05)는 같은 소리로 인식기를 비교하며, 두 패스로 돈다:

    1패스  --keep-audio [--only-ids]   모든 발화를 다시 합성하고(텍스트 캐시에는 소리가 없다), 기준 인식기
           T(large-v3-turbo)에 주는 16비트 16 kHz WAV와 T의 글을 outputs/asr3/audio/에 남긴다(index.jsonl,
           발화마다 한 줄. --only-ids: 쓴 글에 주문번호·접수번호·운송장번호가 있는 발화만, 싼 첫 관문)
    2패스  --recognise ARM             남긴 WAV에 팔 하나를 돌린다: Th(업종 맥락을 hotwords로 준 T),
           L(large-v3), N(whisper-ko-ft turbo-n, CTranslate2). 디코딩은 T와 같다. Qwen3-ASR 팔은 이
           프로젝트 자체 환경에서 돈다(scripts/asr_qwen.py). 출력 outputs/asr3/audio/hyp/<ARM>.jsonl

각 팔이 학생에게 주는 데이터셋은 `python -m call_summary.asr_arms build`가 만든다.
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
# [읽기 고정] 2026-10-14부터 verbalize는 4단계 때의 support-agent 파일(커밋 a5717ce)을 옮긴 사본만 쓴다
# (load_verbalize, docs/experiments.md "음성 조건 v2"의 2026-10-14 덧붙임). 합성·인식
# (support_agent.voice.speech)은 그대로 support-agent 음성 환경에서 가져오며, 그 라이브러리 버전은 캐시
# 키(speech_identity)가 지킨다.

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
import types
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
SPEEDS = {"상담원": 1.0, "고객": 1.1}  # 한국어 목소리 하나, 화자마다 다른 속도
VERBALIZERS = ("v1", "v2")
DIGIT_NAMES = "공일이삼사오육칠팔구"  # support-agent의 verbalize가 전화번호를 읽는 대로(0은 "공")
# 고정한 verbalize의 출처. PINNED_VERBALIZE는 support-agent의 PINNED_SOURCE를 커밋 PINNED_COMMIT 시점
# 그대로 옮긴 사본이고, 머리 주석을 뺀 본문의 git blob 해시가 PINNED_BLOB이다(`git rev-parse
# <커밋>:<경로>`가 내는 값과 같은 계산이라 옆 저장소 없이도 확인된다). 이 커밋을 고른 근거: 이 파일의
# 코드는 a5717ce(2026-09-21) 뒤 2026-10-13까지 바뀌지 않았고(56507f0은 주석만 더함), 4단계·v2·v3 데이터가
# 모두 그 사이에 만들어졌으며, 이 사본의 읽기가 기록된 캐시의 v1·v2 발화와 글자까지 같다
# (tests/test_asr_condition.py, docs/experiments.md).
PINNED_VERBALIZE = Path(__file__).resolve().with_name("verbalize_a5717ce.py")
PINNED_COMMIT = "a5717ce60ca95aaa69c34e3d3dc91af6de00603a"
PINNED_SOURCE = "src/support_agent/voice/verbalize.py"
PINNED_BLOB = "e956c969424fee355d53ccc613d2915a95530181"
# _TRACKING: 운송장번호 꼴(4-4-4). 앞뒤에 숫자나 하이픈이 붙어 더 긴 수의 일부인 경우는 잡지 않는다. 이 꼴이
# 아닌 하이픈 숫자(주소 "761-4" 등)는 v2 규칙의 범위 밖이다(docs/experiments.md 음성 조건 v2).
# ID_PATTERN: 쓴 글의 주문번호(D+7), 접수번호(R+6), 운송장번호. v3 관문은 이 번호가 든 발화만 먼저 합성한다.
_TRACKING = re.compile(r"(?<![\d-])(\d{4})-(\d{4})-(\d{4})(?![\d-])")
ID_PATTERN = re.compile(r"D\d{7}|R\d{6}|\d{4}-\d{4}-\d{4}")  # 쓴 글의 주문번호, 접수번호, 운송장번호
# DEFAULT_AUDIO: v3 1패스가 소리를 남기는 곳(커밋하지 않음). TURBO_N: 형제 저장소 whisper-ko-ft에서 학습해
# CTranslate2로 바꾼 turbo-n(N 팔)의 위치를 이 저장소 기준 상대 경로로 찾는다.
DEFAULT_AUDIO = "outputs/asr3/audio"
TURBO_N = Path(__file__).resolve().parents[2] / "whisper-ko-ft" / "outputs" / "turbo-n" / "ct2"
# RECOGNISERS: faster-whisper로 도는 팔 → (모델, 업종 맥락을 hotwords로 줄지). T는 기준(large-v3-turbo), Th는
# T + 카탈로그 hotwords, L은 large-v3, N은 turbo-n. Qwen3-ASR 팔은 scripts/asr_qwen.py가 따로 돈다.
# DECODING: 모든 팔이 같은 디코딩(한국어, beam 5, temperature 0, 이전 글 조건 끔, VAD 끔). v2의 T와 같아
# 인식기만 바꾼 비교가 된다(docs/experiments.md 음성 조건 v3 팔 표).
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


# --- 말하는 글(발음형) ---------------------------------------------------------------------------------


# v2 규칙: 운송장번호를 숫자 하나씩, 묶음 사이에 쉼표를 두어 읽는 꼴로 바꾼다("6291-7877-5168로" → "육이구일,
# 칠팔칠칠, 오일육팔 로"). verbalize가 처음에 하는 것과 같은 NFKC 정규화 뒤에 찾는다. 번호가 없는 글은 한
# 글자도 바꾸지 않고 그대로 돌려줘 v1과 같은 소리·같은 캐시를 쓰게 한다(tests/test_asr_condition.py가 확인).
def spell_tracking_ids(text: str) -> str:
    """'6291-7877-5168로' -> ' 육이구일, 칠팔칠칠, 오일육팔 로'. 번호가 없는 글은 그대로 돌려준다."""
    norm = unicodedata.normalize("NFKC", text)  # verbalize()도 같은 단계로 시작한다
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
    """쓴 발화에 주문번호·접수번호·운송장번호가 있는지(싼 관문이 고르는 발화)."""
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
    """실행이 무엇을 합성할지 캐시 상태별로 센다. `changed` = v1과 다르게 읽는 발화.
    `unchanged_not_cached` > 0이면 음성 설정이 더는 캐시와 맞지 않는다는 뜻이다."""
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


# --- 전화 채널 ------------------------------------------------------------------------------------------


# 8비트 μ-law로 압축 → 256단계 양자화 → 복원을 한 번 거친 파형. whisper-ko-ft의 전화 채널과 같은 식이다(4단계
# 규칙). numpy·scipy는 쓰는 함수 안에서 불러온다.
def mu_law_roundtrip(samples):
    """8비트 mu-law 코드로 압축했다가 다시 펼친다(G.711 방식). whisper-ko-ft의 채널과 같다."""
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
    """300-3400 Hz, 8 kHz, 8비트 mu-law 채널을 거친 16 kHz 소리(표본화율은 16 kHz 그대로)."""
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
    """캐시된 발화가 의존하는 음성 설정(캐시 키의 일부. verbalizer는 들어가지 않는다)."""
    from support_agent.voice.speech import versions

    return {
        "tts": "melotts/KR",
        "speeds": SPEEDS,
        "channel": "telephone 300-3400 Hz, 8 kHz, 8-bit mu-law",
        "stt": f"faster-whisper/{whisper_model}, beam 5, temperature 0, ko",
        "versions": versions("melotts", "torch", "faster-whisper", "ctranslate2"),
    }


# git이 파일 내용에 붙이는 blob 해시(sha1("blob <길이>\0" + 내용)). `git hash-object`·`git rev-parse
# <커밋>:<경로>`와 같은 값이라, support-agent 저장소 없이도 사본이 그 커밋의 파일과 같은지 확인할 수 있다.
def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# 고정 사본에서 원본 부분만 꺼낸다: 줄바꿈을 LF로 맞추고(Windows 체크아웃은 core.autocrlf로 CRLF가 될 수
# 있고, 원본 blob은 LF뿐이다), 맨 앞의 머리 주석(# 줄)과 빈 줄을 뺀다. 원본은 모듈 docstring으로 시작하므로
# 머리 주석과 섞이지 않는다.
def pinned_body(raw: bytes) -> bytes:
    lines = raw.replace(b"\r\n", b"\n").splitlines(keepends=True)
    start = 0
    while start < len(lines) and (lines[start].startswith(b"#") or not lines[start].strip()):
        start += 1
    return b"".join(lines[start:])


# 발음형 변환(verbalize)을 불러온다. 숫자·기호를 한국어 읽는 법으로 풀어 써야 TTS가 읽을 수 있다.
# support_agent.voice.verbalize(옆 저장소의 현재 파일)는 import하지 않는다: support-agent가 2026-10-13에
# 운송장번호 읽기를 고쳐, 그 파일로는 v1이 4단계 읽기가 아니게 되고(v2와 같은 읽기) 그 키가 캐시에 있어
# 실행도 멈추지 않는다. 대신 4단계 때의 파일을 옮긴 사본(PINNED_VERBALIZE)을 쓰고, 본문의 blob 해시가
# PINNED_BLOB이 아니면(누가 사본을 고쳤으면) 멈춘다. 확인한 바이트를 그대로 실행하므로 확인과 실행 사이에
# 파일이 바뀌어도 섞이지 않는다. 출력: verbalize(text) -> 한글·공백·`.,?!~`만 남은 읽는 글.
def load_verbalize() -> Callable[[str], str]:
    raw = PINNED_VERBALIZE.read_bytes()
    blob = git_blob_id(pinned_body(raw))
    if blob != PINNED_BLOB:
        raise SystemExit(
            f"{PINNED_VERBALIZE} is not support-agent {PINNED_COMMIT[:7]}:{PINNED_SOURCE} "
            f"(blob {blob} != {PINNED_BLOB}); v1 and v2 would no longer be the recorded readings"
        )
    module = types.ModuleType("verbalize_a5717ce")
    module.__file__ = str(PINNED_VERBALIZE)
    exec(compile(raw, str(PINNED_VERBALIZE), "exec"), module.__dict__)
    return module.verbalize


# 발화 하나를 말하고(합성 + 전화 채널) 듣는(인식) 통로. 모델은 처음 캐시에 없는 발화를 만날 때 올리므로, 모두
# 캐시에 있으면 GPU 없이 돈다. 캐시 파일(outputs/asr/cache.jsonl)을 처음에 전부 읽어 둔다.
class Channel:
    """발화를 하나씩 말하고 듣는다. 모델은 처음 캐시에 없는 발화를 만날 때 올린다."""

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
        self.device = "cuda"  # 측정 조건. "cpu"는 스모크 실행에만(--device cpu)
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
        """전화 채널을 거친 발화 하나의 16 kHz 소리(모델이 올라가 있어야 한다)."""
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


# --- 음성 조건 v3 -------------------------------------------------------------------------------------


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
    """1패스 출력: 발화마다 기준 인식기에 준 WAV 바이트와, 발화마다 index 한 줄
    ({key, split, item_id, turn, domain, speaker, seed, spoken, cached_heard, audio_seconds, heard, wav,
    wav_sha256}. `heard`는 T의 글, `cached_heard`는 같은 키의 텍스트 캐시 글, wav가 None이면 소리 없음).
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
        """다른 음성 설정으로 만든 소리를 한 저장소에 섞지 않도록 거부한다."""
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
    """1패스가 남길 모든 발화의 (건, 발화 번호, seed, 키, 읽을 글)."""
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
    """1패스가 할 일. `in_text_cache`는 합성할 발화 중 v2 텍스트 캐시에 키가 있는 수다(음성 설정이
    캐시와 같으면 전부)."""
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
    """1패스: 저장소에 아직 없는 발화마다 합성하고, 그 WAV와 기준 인식기의 글을 남긴다."""
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
        heard = channel.listener.transcribe(wav)  # 측정한 채널 그대로
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
    """2패스: 그 분할의 남긴 WAV 중 아직 출력에 없는 것마다 팔 하나를 돌린다(이어 하기 가능)."""
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
    """hotwords가 없으면 T와 같은 호출(support_agent WhisperListener.transcribe), 있으면 같은 디코딩에
    faster-whisper의 `hotwords`를 더한다."""

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
    """모델 출처: 로컬 폴더면 model.bin 해시, 아니면 캐시된 스냅샷 폴더."""
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
    """<hyp>.meta.json: 한 번만 쓴다. 이어 하는 실행은 같은 팔·같은 모델이어야 한다."""
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
    """CTranslate2는 cuBLAS와 cuDNN을 이름으로 불러온다. support-agent 환경에는 nvidia-* 휠로 들어 있고
    (WhisperListener가 그 폴더를 더한다), N 팔이 도는 whisper-ko-ft 환경은 whisper_ko_ft.evaluate_ct2처럼
    torch에 딸려 온 것을 쓴다."""
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
    ap.add_argument(
        "--verbalizer",
        choices=VERBALIZERS,
        default="v2",
        help="발음형 판. v1: 4단계 읽기(support-agent a5717ce의 verbalize 사본 그대로, 운송장번호를 기수와 "
        "범위 '에서'로 읽은 인공물 포함). v2(기본): v1에서 운송장번호만 숫자 하나씩, 묶음 사이 쉼표. 둘 다 "
        "scripts/verbalize_a5717ce.py를 쓰고 support-agent의 현재 verbalize는 쓰지 않는다",
    )
    ap.add_argument("--out", help="출력 폴더(기본: v1은 data/asr, 그 밖은 data/asr-<verbalizer>)")
    ap.add_argument("--cache", default="outputs/asr/cache.jsonl")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="캐시 계획만 찍고 멈춘다(모델 없음)")
    ap.add_argument("--audio", default=DEFAULT_AUDIO, help="v3: 남긴 소리 폴더(1패스가 쓰고 2패스가 읽음)")
    ap.add_argument("--keep-audio", action="store_true", help="v3 1패스: 다시 합성하고 WAV를 남긴다")
    ap.add_argument("--only-ids", action="store_true", help="v3 1패스: 쓴 글에 식별자가 있는 발화만")
    ap.add_argument("--recognise", choices=list(RECOGNISERS), help="v3 2패스: 남긴 WAV에 이 팔을 돌린다")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="v3: cpu는 스모크 실행에만")
    ap.add_argument("--hyp", help="v3 2패스 출력(기본 <audio>/hyp/<ARM>.jsonl)")
    ap.add_argument("--context", help="v3 2패스: 업종별 맥락 JSON(python -m call_summary.asr_arms context)")
    ap.add_argument(
        "--allow-full-run",
        action="store_true",
        help="v1과 같게 읽지만 캐시에 없는 발화도 합성한다"
        "(새 분할이거나, 음성 설정이 바뀌어 전부 다시 도는 경우)",
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
    # 계획 보고에 고정한 verbalize의 출처(커밋과 blob)도 남겨, 실행 로그만 봐도 어떤 읽기로 만들었는지 알 수
    # 있게 한다. 데이터의 meta.asr에는 넣지 않는다: 넣으면 같은 캐시로 다시 만든 데이터가 기록된 파일(실행
    # 기록에 data_sha로 남음)과 달라진다.
    p = channel.plan(todo)
    report = {
        "split": args.split,
        "verbalizer": args.verbalizer,
        "verbalize": f"support-agent {PINNED_COMMIT[:7]}:{PINNED_SOURCE} (blob {PINNED_BLOB[:12]})",
        "items": len(todo),
        "done": len(done),
        **p,
    }
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
