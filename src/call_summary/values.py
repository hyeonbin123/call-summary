"""Entity value kinds: how a value is normalized for comparison and how it is found in a transcript.

Every entity type in a domain has one kind. Scoring compares normalized values, and the hallucination
check asks whether a predicted value (normalized) occurs anywhere in the transcript (also normalized).
"""

# [파일 역할] 핵심 값의 종류(번호·금액·날짜·시각·이름)마다 "같은 값인지"를 판단하는 정규화 규칙과, 값이
# 전사(또는 요약)에 들어 있는지 찾는 규칙을 모아 둔다.
# [왜 필요한가] 같은 금액도 "32,000원", "3만 2천 원", "삼만 이천 원"으로 말할 수 있고 명세가 일부러 말투를
# 바꿔 뽑는다(README "데이터를 만드는 방식"). 글자 그대로 비교하면 맞은 값도 틀리게 채점되므로, 채점·환각
# 검사·생성 검사·번호 표시가 모두 이 한 파일의 규칙을 쓰게 해 기준이 어긋나지 않게 한다(docs/experiments.md
# "공통").
# [주요 구성] parse_korean_number(한글·아라비아 숫자 섞인 수 읽기) → 종류별 _norm_* → normalize(공개 입구) →
# values_in_transcript(전사에서 숫자형 값 모두 뽑기) → digit_runs·occurs_in_transcript(값이 전사에 있는지).
# [연결] scoring.py(필드 채점·환각), gen.py(생성한 대화 자동 검사), finalize.py(분할 사이 번호 겹침),
# specs.py(날짜 순서), verify.py(확인이 필요한 번호), value_facts.py(요약 값 확인),
# asr_report.py·asr_arms.py(음성 조건 분석)가 가져다 쓴다.

from __future__ import annotations

import re
from typing import Literal

# 값 종류. domains.py의 업종별 핵심 값 유형마다 이 중 하나가 붙는다. id=번호(주문번호 등), amount=금액,
# date=날짜, time=시각, text=이름류(상품명, 요금제 등).
Kind = Literal["id", "amount", "date", "time", "text"]

# 한글로 읽은 숫자 한 자리("공"과 "영"은 모두 0)와 자리 단위. 작은 단위(십·백·천)는 지금 구간 안에서 더하고,
# 큰 단위(만·억)는 그때까지 모은 구간에 곱해 총합으로 넘긴다.
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


# 한글 수사와 아라비아 숫자가 섞인 금액 표현을 정수로 읽는다. 숫자로 읽을 수 없는 글자가 하나라도 있으면
# None을 돌려 "금액이 아님"을 알린다.
# 동작: total(만·억 단위로 넘긴 합), section(지금 큰 단위 아래 구간의 합), num(아직 단위를 만나지 않은 숫자)을
# 들고 왼쪽부터 읽는다. 예: "3만2천" → 3에 만을 곱해 30000을 total로, 2에 천을 곱해 section 2000 → 32000.
# 주의: 한글 숫자를 단위 없이 이어 쓴 꼴("일이삼")은 자리값으로 읽지 않고 마지막 숫자만 남는다. 금액
# 읽기용이며 번호를 숫자 하나씩 읽은 꼴은 대상이 아니다.
def parse_korean_number(text: str) -> int | None:
    """Parse '32000', '32,000', '3만 2천', '삼만 이천', '1억 2,500만' into an int (None if not a number)."""
    # 공백과 천 단위 쉼표는 값에 영향이 없으므로 먼저 지운다.
    s = re.sub(r"[\s,]", "", text)
    if not s:
        return None
    total = 0
    section = 0  # value below the current big unit
    num: int | None = None  # pending digit run
    i = 0
    while i < len(s):
        ch = s[i]
        # 아라비아 숫자는 이어진 자리 전체를 한 수로 읽는다("2,500만"의 2500).
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
            # "천"처럼 앞에 숫자가 없으면 1로 본다("천 원" = 1000).
            section += (num if num is not None else 1) * _SMALL_UNITS[ch]
            num = None
        elif ch in _BIG_UNITS:
            # 만·억: 남은 숫자까지 구간에 더한 뒤 구간 전체에 큰 단위를 곱한다. 구간이 비었으면("만 원") 1로
            # 본다.
            section += num or 0
            total += (section or 1) * _BIG_UNITS[ch]
            section = 0
            num = None
        else:
            return None
        i += 1
    return total + section + (num or 0)


# 번호 정규화: 공백·하이픈·밑줄·점·가운뎃점을 빼고 대문자로. "6291-7877-5168"과 "6291 7877 5168", "d1234567"과
# "D1234567"을 같은 번호로 본다.
def _norm_id(v: str) -> str:
    return re.sub(r"[\s\-_.·]", "", v).upper()


# 이름류 정규화: 띄어쓰기 차이("은하 탭 9"와 "은하탭9")와 영문 대소문자만 무시한다.
def _norm_text(v: str) -> str:
    return re.sub(r"\s+", "", v).lower()


# 금액 정규화: 끝의 "원"을 떼고 parse_korean_number로 읽어 숫자 문자열로 만든다. "월 6,200원"처럼 앞에 다른
# 낱말이 붙으면 읽지 못해 None이 된다(docs/experiments.md 2026 비교 대상 test 결과의 바로잡음: 모든 모델에
# 같은 규칙이라 고치지 않음).
def _norm_amount(v: str) -> str | None:
    s = v.strip()
    s = re.sub(r"\s*원$", "", s)
    n = parse_korean_number(s)
    return None if n is None else str(n)


# 날짜 패턴 두 가지. (1) "[2026년] 10월 5일" 꼴, (2) "[2026-]10-05", "10/5", "10.5" 꼴. 연도는 있어도 버리고
# 월-일만 비교한다.
# (2)의 앞뒤 조건(숫자나 하이픈이 붙어 있지 않을 것)은 운송장번호·전화번호 같은 "6291-7877-5168" 안의 조각을
# 날짜로 읽지 않으려는 것이다.
_DATE_PATTERNS = [
    re.compile(r"(?:(\d{4})\s*[-./년]\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일"),
    re.compile(r"(?<![\d\-])(?:(\d{4})[-./])?(\d{1,2})[-./](\d{1,2})(?![\d\-])"),
]


# 날짜 정규화: 처음 맞는 패턴에서 월·일을 꺼내 "MM-DD"로. 월 1~12, 일 1~31을 벗어나면 날짜로 보지 않는다.
# 문자열 비교로 날짜 순서를 볼 수 있게 두 자리로 맞춘다(specs.py가 날짜 순서 검사에 쓴다).
def _norm_date(v: str) -> str | None:
    for pat in _DATE_PATTERNS:
        m = pat.search(v)
        if m:
            month, day = int(m.group(2)), int(m.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{month:02d}-{day:02d}"
    return None


# 시각 패턴. _TIME_KO는 "오후 2시 30분", "저녁 7시 반" 같은 한국어 꼴(시간대 낱말은 선택), _TIME_COLON은
# "14:30" 꼴이다.
_TIME_KO = re.compile(r"(오전|오후|아침|저녁|밤|낮)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분|\s*(반))?")
_TIME_COLON = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")


# 한국어 시각 매치를 24시간제 "HH:MM"으로 바꾼다. "반"은 30분. 오후·저녁·밤은 12시 미만이면 12를 더하고,
# "낮"은 6시 미만일 때만 더한다("낮 2시" = 14:00, "낮 11시"는 그대로).
# 시간대 낱말이 없으면 그대로 두므로 "2시"와 "오후 2시"는 다른 값이다(docs/experiments.md 요약 판정 재보정 1:
# 시각은 오전·오후까지 맞아야 한다).
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


# 시각 정규화: 한국어 꼴을 먼저 보고, 없으면 "HH:MM" 꼴을 본다. 범위를 벗어나면 None.
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


# 값 종류에 맞는 정규화 함수로 보내는 공개 입구. 읽을 수 없는 값은 None이다. 채점(scoring._entity_keys)은
# None인 예측 값을 어떤 정답과도 맞지 않는 값(fp 하나)으로 센다.
# 번호·이름은 정규화 결과가 빈 문자열이면 None으로 바꿔, 빈 값("")을 "있는 값"으로 세지 않게 한다.
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


# 전사에서 "원" 앞에 붙은 수 덩어리(아라비아 숫자, 쉼표, 한글 수사, 공백)를 찾는 패턴. 덩어리 앞쪽에 수
# 글자로만 된 다른 낱말이 붙어 잡힐 수 있어, values_in_transcript가 뒤쪽 부분을 하나씩 다시 읽는다.
_AMOUNT_RUN = re.compile(r"([\d,영공일이삼사오육칠팔구십백천만억 ]+?)\s*원")


# 전사에 나오는 금액·날짜·시각을 모두 정규화한 집합으로 돌려준다. 예측 값이 이 집합에 있으면 "전사에 있는
# 값"이다.
# 쓰는 곳: occurs_in_transcript(숫자형 종류), gen.py(명세에 없는 금액·날짜·시각이 대화에 끼어들었는지 검사),
# asr_arms.py.
# 번호·이름은 정해진 꼴이 없어 모두 뽑을 수 없으므로 부르면 예외를 낸다(그 종류는 occurs_in_transcript가 부분
# 문자열로 찾는다).
def values_in_transcript(kind: Kind, transcript: str) -> set[str]:
    """All normalized values of this kind that can be read from the transcript (numeric kinds only)."""
    found: set[str] = set()
    if kind == "amount":
        for m in _AMOUNT_RUN.finditer(transcript):
            tokens = m.group(1).split()
            # The run may start with an unrelated word made of number characters; try every suffix.
            for k in range(len(tokens)):
                n = parse_korean_number("".join(tokens[k:]))
                # 0원은 금액으로 세지 않는다.
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


# 아래 _DIGIT_RUN은 4단계 채점 규칙(docs/experiments.md 4단계 "채점")에서 정한 것으로, 인식기가 쉼표·공백·짧은
# 낱말로 나눠 적은 번호를 한 덩어리로 다시 잇는 데 쓴다.
# Digit groups with at most 3 other characters between them, read as one number. A recognised number may
# come back split, e.g. "5533 8050 3287". (In speech condition v1 "5533-8050-3287" came back as
# "5,533, 8,050에서 3,287" because the TTS was given cardinals joined by a range "에서" - a verbalizer
# artifact, not a recogniser habit; v2 reads tracking numbers digit by digit. docs/experiments.md)
_DIGIT_RUN = re.compile(r"\d+(?:\D{1,3}\d+)*")


# 전사의 숫자 덩어리마다 숫자만 남긴 문자열 목록. "5,816, 9,659에서 8,958" →
# "581696598958"(tests/test_scoring.py).
def digit_runs(transcript: str) -> list[str]:
    return [re.sub(r"\D", "", m.group(0)) for m in _DIGIT_RUN.finditer(transcript)]


# 예측(또는 정답) 값이 전사·요약에 있는지 판단한다. 환각 지표(scoring.py), 확인이 필요한 번호의 전사 포함
# 검사(verify.py, 늘 spoken=True), 요약 값 확인(value_facts.py, spoken=False)이 같은 함수를 쓴다.
# 번호는 정규화한 전사 안의 부분 문자열로 찾는다. 그래서 정답 번호 앞뒤에 글자가 더 붙은 값("OD1902657")이
# 있어도 찾은 것으로 친다(docs/experiments.md 요약 판정 재보정에 적은 약점).
# spoken=True(음성 인식 전사)일 때만 넓은 규칙을 더한다: 번호의 숫자(4자리 이상)가 전사의 숫자 덩어리 하나
# 안에 있으면 찾은 것으로 본다. 4자리 미만은 우연히 맞기 쉬워 넓은 규칙에서 뺀다.
def occurs_in_transcript(kind: Kind, value: str, transcript: str, spoken: bool = False) -> bool:
    """Whether a (predicted) value can be found in the transcript. Unreadable values count as not found.

    spoken=True (recognised speech, stage 4): an identifier also counts as found when its digits (4 or
    more) lie inside one digit run of the transcript.
    """
    norm = normalize(kind, value)
    if norm is None:
        return False
    # 번호: 먼저 쓴 글 규칙(정규화한 부분 문자열), 안 되면 음성 조건에서만 숫자 덩어리 규칙.
    if kind == "id":
        if norm in _norm_id(transcript):
            return True
        digits = re.sub(r"\D", "", norm)
        return spoken and len(digits) >= 4 and any(digits in run for run in digit_runs(transcript))
    if kind == "text":
        return norm in _norm_text(transcript)
    # 금액·날짜·시각: 전사에서 뽑은 정규화 값 집합에 있는지 본다. 부분 문자열로 찾지 않으므로 "32,000원"이
    # "132,000원" 안에서 잡히는 일이 없다.
    return norm in values_in_transcript(kind, transcript)
