"""Entity value kinds: how a value is normalized for comparison and how it is found in a transcript.

Every entity type in a domain has one kind. Scoring compares normalized values, and the hallucination
check asks whether a predicted value (normalized) occurs anywhere in the transcript (also normalized).
"""

from __future__ import annotations

import re
from typing import Literal

Kind = Literal["id", "amount", "date", "time", "text"]

_HANGUL_DIGITS = {
    "영": 0,
    "공": 0,
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_SMALL_UNITS = {"십": 10, "백": 100, "천": 1000}
_BIG_UNITS = {"만": 10_000, "억": 100_000_000}


def parse_korean_number(text: str) -> int | None:
    """Parse '32000', '32,000', '3만 2천', '삼만 이천', '1억 2,500만' into an int (None if not a number)."""
    s = re.sub(r"[\s,]", "", text)
    if not s:
        return None
    total = 0
    section = 0  # value below the current big unit
    num: int | None = None  # pending digit run
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            num = int(s[i:j])
            i = j
            continue
        if ch in _HANGUL_DIGITS:
            num = _HANGUL_DIGITS[ch]
        elif ch in _SMALL_UNITS:
            section += (num if num is not None else 1) * _SMALL_UNITS[ch]
            num = None
        elif ch in _BIG_UNITS:
            section += num or 0
            total += (section or 1) * _BIG_UNITS[ch]
            section = 0
            num = None
        else:
            return None
        i += 1
    return total + section + (num or 0)


def _norm_id(v: str) -> str:
    return re.sub(r"[\s\-_.·]", "", v).upper()


def _norm_text(v: str) -> str:
    return re.sub(r"\s+", "", v).lower()


def _norm_amount(v: str) -> str | None:
    s = v.strip()
    s = re.sub(r"\s*원$", "", s)
    n = parse_korean_number(s)
    return None if n is None else str(n)


_DATE_PATTERNS = [
    re.compile(r"(?:(\d{4})\s*[-./년]\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"(?<![\d\-])(?:(\d{4})[-./])?(\d{1,2})[-./](\d{1,2})(?![\d\-])"),
]


def _norm_date(v: str) -> str | None:
    for pat in _DATE_PATTERNS:
        m = pat.search(v)
        if m:
            month, day = int(m.group(2)), int(m.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{month:02d}-{day:02d}"
    return None


_TIME_KO = re.compile(r"(오전|오후|아침|저녁|밤|낮)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분|\s*(반))?")
_TIME_COLON = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")


def _time_from_match(m: re.Match[str]) -> str | None:
    period, hour_s, minute_s, half = m.group(1), m.group(2), m.group(3), m.group(4)
    hour = int(hour_s)
    minute = int(minute_s) if minute_s else (30 if half else 0)
    if period in ("오후", "저녁", "밤") and hour < 12:
        hour += 12
    if period == "낮" and hour < 6:
        hour += 12
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return f"{hour:02d}:{minute:02d}"


def _norm_time(v: str) -> str | None:
    m = _TIME_KO.search(v)
    if m:
        return _time_from_match(m)
    m = _TIME_COLON.search(v)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    return None


def normalize(kind: Kind, value: str) -> str | None:
    """Canonical form used for comparison. None when the value cannot be read as this kind."""
    if kind == "id":
        out = _norm_id(value)
    elif kind == "text":
        out = _norm_text(value)
    elif kind == "amount":
        return _norm_amount(value)
    elif kind == "date":
        return _norm_date(value)
    elif kind == "time":
        return _norm_time(value)
    else:  # pragma: no cover - Literal guards this
        raise ValueError(kind)
    return out or None


_AMOUNT_RUN = re.compile(r"([\d,영공일이삼사오육칠팔구십백천만억 ]+?)\s*원")


def values_in_transcript(kind: Kind, transcript: str) -> set[str]:
    """All normalized values of this kind that can be read from the transcript (numeric kinds only)."""
    found: set[str] = set()
    if kind == "amount":
        for m in _AMOUNT_RUN.finditer(transcript):
            tokens = m.group(1).split()
            # The run may start with an unrelated word made of number characters; try every suffix.
            for k in range(len(tokens)):
                n = parse_korean_number("".join(tokens[k:]))
                if n is not None and n > 0:
                    found.add(str(n))
    elif kind == "date":
        for pat in _DATE_PATTERNS:
            for m in pat.finditer(transcript):
                month, day = int(m.group(2)), int(m.group(3))
                if 1 <= month <= 12 and 1 <= day <= 31:
                    found.add(f"{month:02d}-{day:02d}")
    elif kind == "time":
        for m in _TIME_KO.finditer(transcript):
            t = _time_from_match(m)
            if t:
                found.add(t)
        for m in _TIME_COLON.finditer(transcript):
            hour, minute = int(m.group(1)), int(m.group(2))
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                found.add(f"{hour:02d}:{minute:02d}")
    else:
        raise ValueError(f"values_in_transcript does not enumerate kind {kind!r}")
    return found


def occurs_in_transcript(kind: Kind, value: str, transcript: str) -> bool:
    """Whether a (predicted) value can be found in the transcript. Unreadable values count as not found."""
    norm = normalize(kind, value)
    if norm is None:
        return False
    if kind == "id":
        return norm in _norm_id(transcript)
    if kind == "text":
        return norm in _norm_text(transcript)
    return norm in values_in_transcript(kind, transcript)
