"""Identifiers read digit by digit, written back as digits (speech condition v3, rule "id-itn-1").

The speech condition reads order, receipt and tracking numbers digit by digit ("디 사사칠칠팔삼팔",
"육이구일, 칠팔칠칠, 오일육팔"). A recogniser may write them back as Hangul digit names, or mix names into
digits ("627일 8127 2631"). Scoring finds an identifier only by its digits (values.occurs_in_transcript), so
such a number counts as lost. `id_itn` rewrites those numbers and nothing else:

- a run: digit names (공 영 일 이 삼 사 오 육 칠 팔 구) and digits, one separator (", " "," " " "-" ".")
  allowed between two of them, not glued to a preceding letter or Hangul syllable. A letter name or letter in
  front (디/D, 알/R) is an order (7 digits) or receipt (6 digits) prefix when the digit count fits.
- words at the edges of a run are not digits:
  - head: a lone digit name before a group of two or more ("이 15,900원", "이 오팔구이, ...") is a word.
  - tail, in this order, each once: a lone last digit that starts the next word ("이 오는데", "2287 3시");
    a lone last "이" (the particle, "이이팔칠 이 맞습니다") unless only the count with it is an identifier
    length (4 card, 6 receipt, 7 order, 12 tracking); a last "이" glued to the next word ("이에요") or
    "이구" glued to "요"; a last "이" that makes the count one more than an identifier length
    ("3038이 오는데").
- a run is rewritten when it has a digit name and is a prefixed identifier of the right length, or is all
  digit names and at least 4 long, or mixes digits and names, is at least 6 long and has no day ("15일").
  Cardinal numbers (십, 백, 천, 만 ...) stop a run, so amounts and dates said as numbers stay as they are.

The rule is the same for every recogniser arm. It was fixed with tests/test_id_itn.py before any Qwen3-ASR
output was seen (docs/experiments.md "음성 조건 v3").
"""

# [파일 역할] 음성 조건 v3(인식기 바꾸기)에서 인식기가 적은 글의 번호만 숫자로 되돌리는 식별자 규칙
# id-itn-1이다. ITN은 inverse text normalization(읽은 꼴을 쓴 꼴로 되돌리기)의 줄임말이다.
# [왜 필요한가] 채점은 번호를 숫자로만 찾는다(values.occurs_in_transcript). 인식기가 "디 사사칠칠팔삼팔"처럼
# 숫자 이름으로 적으면 모델이 번호를 옮길 수 없고 채점에서도 잃은 값이 된다. 인식기마다 숫자를 적는 꼴이
# 달라(whisper-ko-ft에서 Qwen3-ASR은 숫자를 한글로 적는 일이 많았다), 모든 후보 팔에 같은 규칙을 씌운 뒤
# 학생이 실제로 받는 글로 비교한다(docs/experiments.md "음성 조건 v3").
# [설계 이유] whisper-ko-ft의 itn.to_digits는 단위가 붙은 수만 바꿔 숫자 하나씩 읽은 번호를 하나도 바꾸지 못해
# 쓰지 않았다. 날짜·금액·시각까지 바꾸는 일반 역정규화도 하지 않는다: 십·백·천·만이 덩어리를 끊어 수사로 읽은
# 값은 그대로 둔다. 규칙은 Qwen3-ASR 출력을 보기 전에 tests/test_id_itn.py와 함께 고정했다.
# [흐름] _RUN으로 숫자 덩어리를 찾음 → _rewrite가 접두(디/알)와 앞뒤 낱말을 가려냄(_trim) → 바꾸는 조건을
# 만족하면 _digits로 숫자화.
# [연결] asr_arms.py가 이름이 "+I"로 끝나는 팔의 모든 발화에 id_itn을 씌운다. scripts/asr_qwen.py의 Qwen 팔도
# 학생에게 갈 때 이 규칙을 거친다.

from __future__ import annotations

import re

# ITN_VERSION: 규칙의 판 이름. asr_arms가 만든 데이터 행의 meta에 남겨 어떤 규칙을 거친 글인지 추적한다.
# DIGIT_NAMES: 한글 숫자 이름 → 숫자. 0은 "공"과 "영" 둘 다 받는다.
ITN_VERSION = "id-itn-1"
DIGIT_NAMES = {
    "공": "0",
    "영": "0",
    "일": "1",
    "이": "2",
    "삼": "3",
    "사": "4",
    "오": "5",
    "육": "6",
    "칠": "7",
    "팔": "8",
    "구": "9",
}
# PREFIXES: 번호 앞 글자 → (붙여 쓸 글자, 그 번호의 숫자 자릿수). 주문번호는 D+7자리, 접수번호는
# R+6자리다(음성 조건은 이 글자를 "디", "알"로 읽는다).
# ID_LENGTHS: 번호로 볼 숫자 길이. 4(카드 끝자리), 6(접수), 7(주문), 12(운송장)이다. 끝의 "이"가 조사인지
# 숫자인지 가를 때 쓴다.
# PARTICLE: 숫자 2와 같은 글자인 조사 "이". 번호 끝의 "이"가 가장 헷갈리는 경우라 따로 이름을 붙였다.
PREFIXES = {"디": ("D", 7), "D": ("D", 7), "알": ("R", 6), "R": ("R", 6)}
ID_LENGTHS = frozenset({4, 6, 7, 12})
PARTICLE = "이"
# GLUED_PARTICLES: 덩어리 바로 뒤 낱말에 붙은 조사 꼴. "이구"는 다음 글자가 "요"일 때만("이구요"), "이"는 뒤에
# 한글이 오면 늘("이에요", "이라고") 숫자가 아닌 조사로 본다.
# (particle, what the next character must start with) for a particle glued to the word after the run
GLUED_PARTICLES = (("이구", "요"), (PARTICLE, ""))

# 정규식 조각들.
# - _NAME: 한글 숫자 이름 한 글자. _DIGIT: 아라비아 숫자나 한글 숫자 이름 한 글자.
# - _SEP: 두 자리 사이에 허용하는 구분자 하나(", ", ",", 공백, ".", "-").
# - _RUN: 숫자 덩어리. 앞 글자가 한글·영문·숫자면 시작하지 않아 낱말 속 글자(예: "통일"의 "일")를 숫자로 잡지
#   않는다. 앞에 접두 글자(디/알/D/R)와 공백 하나가 올 수 있다.
# - _LONE_HEAD: 덩어리 맨 앞의 외톨이 숫자 이름 뒤에 두 자 이상 묶음이 오는 꼴("이 15,900원"의 "이").
# - _LONE_TAIL: 구분자 뒤 마지막 한 자(외톨이 끝 숫자).
# - _DAY: "15일" 같은 날짜. 숫자와 이름이 섞인 덩어리에 있으면 번호가 아니라 날짜로 본다.
# - _HANGUL: 한글 음절 한 자.
_NAME = "[" + "".join(DIGIT_NAMES) + "]"
_DIGIT = "[0-9" + "".join(DIGIT_NAMES) + "]"
_SEP = r"(?:, ?|[ .\-])"
_RUN = re.compile(rf"(?<![가-힣A-Za-z0-9])(?:(?P<prefix>[디알DR]) ?)?(?P<run>{_DIGIT}(?:{_SEP}?{_DIGIT})*)")
_LONE_HEAD = re.compile(rf"{_NAME}{_SEP}(?={_DIGIT}{_DIGIT})")
_LONE_TAIL = re.compile(rf"{_SEP}({_DIGIT})$")
_DAY = re.compile(r"(?<![0-9])[0-9]{1,2}일")
_HANGUL = re.compile(r"[가-힣]")


# 덩어리의 (전체 숫자 자릿수, 그중 한글 숫자 이름 수)를 센다. 구분자는 세지 않는다.
def _count(run: str) -> tuple[int, int]:
    """(digit characters, digit names) in a run."""
    names = sum(ch in DIGIT_NAMES for ch in run)
    return names + sum(ch.isdigit() for ch in run), names


# 덩어리 끝에 붙어 잡힌 낱말 조각을 떼어 낸 앞부분을 돌려준다. after는 덩어리 바로 뒤 한 글자, lengths는
# 번호로 볼 자릿수 집합이다.
# 모듈 docstring의 순서 그대로 한 번씩만 본다. 순서를 바꾸면 "이"가 숫자로 남거나 잘려 나가는 결과가
# 달라지므로, 테스트로 고정한 순서다.
def _trim(run: str, after: str, lengths: frozenset[int]) -> str:
    """The run without the words at its end (module docstring), each step once, in this order."""
    # 1) 마지막 외톨이 숫자 바로 뒤가 한글이면 그 숫자는 다음 낱말의 첫 글자다("이 오는데", "2287 3시"). 떼어
    #    낸 조각이 새 after가 되므로 아래 3)의 한글 검사에는 걸리지 않는다.
    last = _LONE_TAIL.search(run)
    if last and _HANGUL.match(after):
        after, run = run[last.start() :], run[: last.start()]  # "이 오는데", "이 일치하지", "2287 3시"
    # 2) 구분자 뒤 외톨이 "이"는 조사로 보고 뗀다("이이팔칠 이 맞습니다"). 단 "이"까지 세야만 번호 길이가 되고
    #    빼면 번호 길이가 아닐 때는 숫자 2로 남긴다.
    last = _LONE_TAIL.search(run)
    if last and last.group(1) == PARTICLE:
        n, _ = _count(run)
        if not (n in lengths and n - 1 not in lengths):
            after, run = run[last.start() :], run[: last.start()]  # "이이팔칠 이 맞습니다"
    # 3) 덩어리 끝이 다음 한글 낱말에 바로 붙은 조사면 뗀다("이에요", "이라고", "이구요"). 덩어리 전체가 조사
    #    하나뿐이면 떼지 않는다.
    if _HANGUL.match(after):
        for particle, before in GLUED_PARTICLES:
            if after.startswith(before) and run.endswith(particle) and len(run) > len(particle):
                return run[: -len(particle)]  # "이에요", "이라고", "이구요"
    # 4) 끝의 "이"를 넣으면 번호 길이보다 하나 많고 빼면 번호 길이가 되면, 그 "이"는 조사다("3038이 오는데":
    #    13자리는 12자리 운송장번호 + 조사).
    if run.endswith(PARTICLE) and len(run) > 1:
        n, _ = _count(run)
        if n not in lengths and n - 1 in lengths:
            return run[:-1]  # "3038이 오는데": 13 digits would be 12 and a particle
    return run


# 한글 숫자 이름을 숫자로 바꾼다. 구분자 등 다른 글자는 그대로 둔다(접두 번호가 아닐 때는 "6291, 7877,
# 5168"처럼 구분자가 남는다).
def _digits(run: str) -> str:
    return "".join(DIGIT_NAMES.get(ch, ch) for ch in run)


# re.sub의 치환 함수. 덩어리 하나를 받아, 바꿀 조건이 맞으면 숫자로 바꾼 글을, 아니면 원래 글을 돌려준다.
# text는 덩어리 뒤 글자(after)를 보려고 함께 받는다.
def _rewrite(m: re.Match[str], text: str) -> str:
    prefix, run = m.group("prefix"), m.group("run")
    after = text[m.end() : m.end() + 1]
    head = m.group(0)[: m.start("run") - m.start()]  # the prefix and its space, as heard
    # 접두 번호(디/알): 그 번호 길이만 기준으로 끝 낱말을 떼고, 숫자 수가 정확히 맞으면 D/R + 숫자만(구분자
    # 제거) + 뗀 꼬리로 바꾼다. 길이가 맞지 않으면 아래 일반 규칙으로 넘어가며, 이때 접두 글자는 들린 그대로
    # head에 남는다.
    if prefix:
        letter, length = PREFIXES[prefix]
        kept = _trim(run, after, frozenset({length}))
        n, _ = _count(kept)
        if n == length:
            return letter + re.sub(r"\D", "", _digits(kept)) + run[len(kept) :]
    else:
        # 접두가 없으면 맨 앞 외톨이 숫자 이름을 낱말로 보고 head로 빼 둔다("이 15,900원"의 "이"는 조사, 뒤는
        # 금액).
        lone = _LONE_HEAD.match(run)
        if lone:
            head, run = lone.group(0), run[lone.end() :]  # "이 15,900원"
    # 모든 번호 길이를 기준으로 끝 낱말을 뗀 뒤, 바꾸는 조건을 본다: 한글 숫자 이름이 하나 이상 있고, (모두
    # 이름이고 4자 이상) 또는 (숫자와 섞였고 6자 이상이며 "15일" 같은 날이 없음). 이름이 없는 순수 숫자
    # 덩어리는 손대지 않으므로, 이미 숫자로 적힌 값은 바뀌지 않는다.
    kept = _trim(run, after, ID_LENGTHS)
    rest = run[len(kept) :]
    n, names = _count(kept)
    if names and ((names == n and n >= 4) or (names < n and n >= 6 and not _DAY.search(kept))):
        return head + _digits(kept) + rest
    return m.group(0)


# 공개 함수: 글 전체에서 숫자 덩어리마다 _rewrite를 적용한다. 같은 입력에 늘 같은 출력을 내는 순수 함수다. 쓴
# 글(dev 3,157발화, train 19,330발화)에서는 바뀌는 발화가 0개임을 측정 전에 확인했다(docs/experiments.md).
def id_itn(text: str) -> str:
    """`text` with identifiers read digit by digit written as digits."""
    return _RUN.sub(lambda m: _rewrite(m, text), text)
