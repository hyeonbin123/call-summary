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

from __future__ import annotations

import re

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
PREFIXES = {"디": ("D", 7), "D": ("D", 7), "알": ("R", 6), "R": ("R", 6)}
ID_LENGTHS = frozenset({4, 6, 7, 12})
PARTICLE = "이"
# (particle, what the next character must start with) for a particle glued to the word after the run
GLUED_PARTICLES = (("이구", "요"), (PARTICLE, ""))

_NAME = "[" + "".join(DIGIT_NAMES) + "]"
_DIGIT = "[0-9" + "".join(DIGIT_NAMES) + "]"
_SEP = r"(?:, ?|[ .\-])"
_RUN = re.compile(rf"(?<![가-힣A-Za-z0-9])(?:(?P<prefix>[디알DR]) ?)?(?P<run>{_DIGIT}(?:{_SEP}?{_DIGIT})*)")
_LONE_HEAD = re.compile(rf"{_NAME}{_SEP}(?={_DIGIT}{_DIGIT})")
_LONE_TAIL = re.compile(rf"{_SEP}({_DIGIT})$")
_DAY = re.compile(r"(?<![0-9])[0-9]{1,2}일")
_HANGUL = re.compile(r"[가-힣]")


def _count(run: str) -> tuple[int, int]:
    """(digit characters, digit names) in a run."""
    names = sum(ch in DIGIT_NAMES for ch in run)
    return names + sum(ch.isdigit() for ch in run), names


def _trim(run: str, after: str, lengths: frozenset[int]) -> str:
    """The run without the words at its end (module docstring), each step once, in this order."""
    last = _LONE_TAIL.search(run)
    if last and _HANGUL.match(after):
        after, run = run[last.start() :], run[: last.start()]  # "이 오는데", "이 일치하지", "2287 3시"
    last = _LONE_TAIL.search(run)
    if last and last.group(1) == PARTICLE:
        n, _ = _count(run)
        if not (n in lengths and n - 1 not in lengths):
            after, run = run[last.start() :], run[: last.start()]  # "이이팔칠 이 맞습니다"
    if _HANGUL.match(after):
        for particle, before in GLUED_PARTICLES:
            if after.startswith(before) and run.endswith(particle) and len(run) > len(particle):
                return run[: -len(particle)]  # "이에요", "이라고", "이구요"
    if run.endswith(PARTICLE) and len(run) > 1:
        n, _ = _count(run)
        if n not in lengths and n - 1 in lengths:
            return run[:-1]  # "3038이 오는데": 13 digits would be 12 and a particle
    return run


def _digits(run: str) -> str:
    return "".join(DIGIT_NAMES.get(ch, ch) for ch in run)


def _rewrite(m: re.Match[str], text: str) -> str:
    prefix, run = m.group("prefix"), m.group("run")
    after = text[m.end() : m.end() + 1]
    head = m.group(0)[: m.start("run") - m.start()]  # the prefix and its space, as heard
    if prefix:
        letter, length = PREFIXES[prefix]
        kept = _trim(run, after, frozenset({length}))
        n, _ = _count(kept)
        if n == length:
            return letter + re.sub(r"\D", "", _digits(kept)) + run[len(kept) :]
    else:
        lone = _LONE_HEAD.match(run)
        if lone:
            head, run = lone.group(0), run[lone.end() :]  # "이 15,900원"
    kept = _trim(run, after, ID_LENGTHS)
    rest = run[len(kept) :]
    n, names = _count(kept)
    if names and ((names == n and n >= 4) or (names < n and n >= 6 and not _DAY.search(kept))):
        return head + _digits(kept) + rest
    return m.group(0)


def id_itn(text: str) -> str:
    """`text` with identifiers read digit by digit written as digits."""
    return _RUN.sub(lambda m: _rewrite(m, text), text)
