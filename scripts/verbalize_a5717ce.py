# [파일 역할] 형제 저장소 support-agent(https://github.com/hyeonbin123/support-agent)의
# src/support_agent/voice/verbalize.py를 커밋 a5717ce60ca95aaa69c34e3d3dc91af6de00603a(2026-09-21) 시점
# 그대로 옮긴 사본이다. scripts/asr_condition.py가 음성 조건 v1(4단계)과 v2의 발음형(합성기에 넘기는 글)을
# 만들 때 support-agent의 현재 파일 대신 이 사본을 쓴다(load_verbalize).
# [왜 이 커밋인가] 이 파일의 코드는 a5717ce 뒤 2026-10-13까지 바뀌지 않았다(56507f0은 주석만 더함, 구문
# 트리 동일). 4단계(2026-09-29), v2(2026-10-04), v3 1패스(2026-10-05) 데이터는 모두 이 기간에 만들었고, 이
# 사본으로 다시 만든 읽는 글이 기록된 음성 캐시의 v1·v2 발화와 글자까지 같다(docs/experiments.md "음성 조건
# v2"의 2026-10-14 덧붙임). 운송장번호(4-4-4)를 기수와 범위 "에서"로 읽는 4단계 인공물도 이 판에서 나온다.
# [왜 사본인가] support-agent는 877b691(2026-10-13)에서 이 읽기를 고쳤다. 현재 파일을 부르면 v1이 몰래 v2와
# 같은 읽기가 되고, 그 키가 캐시에 있어 실행이 멈추지도 않는다. 커밋을 정해 `git show`로 읽는 방법은 옆에
# support-agent 저장소와 그 이력이 있어야 해서, 그 저장소가 없는 GitHub CI에서는 확인할 수 없다. 이 파일은
# 표준 라이브러리(re, unicodedata)만 쓰므로 사본으로 두면 CI에서도 v1·v2 읽기를 그대로 시험할 수 있다.
# [주의] 이 머리 주석 아래는 원본과 바이트까지 같다. 그래서 원본의 영어 docstring·주석도 번역하지 않는다.
# asr_condition.load_verbalize()는 머리 주석(맨 앞의 # 줄과 빈 줄)을 뺀 본문의 git blob 해시가
# e956c969424fee355d53ccc613d2915a95530181인지 확인하고 다르면 멈춘다. 본문을 고치지 않는다: 읽기를 바꾸려면
# asr_condition.py에 새 판을 정의하고 규칙을 docs/experiments.md에 먼저 적는다.

"""Written Korean chat text -> the words a customer would say aloud (Hangul only).

A speech synthesiser reads "O-91010" or "38,900원" in ways nobody says them, so the text is spelled out
first: identifiers and phone numbers digit by digit, money and dates as Sino-Korean numbers, counted things
with native numerals, Latin letters by their letter names. Pure functions, no model.
"""

from __future__ import annotations

import re
import unicodedata

DIGIT_NAMES = "공일이삼사오육칠팔구"  # one digit at a time: phone numbers and identifiers ("공" for 0)
_SINO = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_SMALL_UNITS = ("", "십", "백", "천")
_BIG_UNITS = ("", "만", "억", "조")
_NATIVE_ONES = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
LETTER_NAMES = {
    "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "지", "h": "에이치",
    "i": "아이", "j": "제이", "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오", "p": "피",
    "q": "큐", "r": "알", "s": "에스", "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스",
    "y": "와이", "z": "지",
}  # fmt: skip
WORDS = {  # Latin words that are read as words, not letter by letter
    "example": "이그잼플", "com": "컴", "net": "넷", "kr": "케이알", "co": "씨오", "gmail": "지메일",
    "naver": "네이버", "mm": "밀리미터", "cm": "센티미터", "km": "킬로미터", "kg": "킬로그램",
    "g": "그램", "ml": "밀리리터", "l": "리터", "ok": "오케이", "vip": "브이아이피", "as": "에이에스",
}  # fmt: skip
# Things counted with native numerals ("두 개", "열 시"). Everything else takes Sino-Korean numbers.
NATIVE_COUNTERS = (
    "개월", "개", "벌", "켤레", "장", "병", "명", "시간", "시", "살", "잔", "마리", "대", "권", "통",
    "박스", "상자", "가지", "군데", "곳", "달", "번째",
)  # fmt: skip
_SINO_ONLY = {"개월"}  # listed above only so that "개" does not match the front of "개월"
_MONTHS = {6: "유", 10: "시"}  # 유월, 시월

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# 010-1234-5678, 01012345678, (010) 1234-5678, +82 10-1234-5678: the forms find_customer accepts
_PHONE = re.compile(r"(?<!\d)(?:\+82[- ]?|\(?0)(1\d)\)?[- .]?(\d{3,4})[- .]?(\d{4})(?!\d)")
_IDENT = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,3}(?:-[A-Za-z0-9]+)+")
_ISO_DATE = re.compile(r"(?<![\d./-])(\d{4})([-./])(\d{1,2})\2(\d{1,2})(?![\d/-])")
_MINUS = re.compile(r"(?<![\w가-힣)])-(?=\d)")
_DATE = re.compile(r"(?<!\d)(\d{1,2})월(\s*)(\d{1,2})일")
_MONTH = re.compile(r"(?<!\d)(\d{1,2})월")
_CLOCK = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_RANGE = re.compile(r"(?<!\d)(\d+)\s*[-~]\s*(\d+)(?=\s*[가-힣%])")
_NUMBER = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s*)(%|[가-힣]+)?")
_LATIN = re.compile(r"[A-Za-z]+")
_KEEP = re.compile(r"[^가-힣 .,?!~]")


def sino(number: int) -> str:
    """38900 -> "삼만 팔천구백". A leading "일" is dropped before 십, 백, 천 and 만, as in speech."""
    if number == 0:
        return "영"
    groups = []
    for index, big in enumerate(_BIG_UNITS):
        number, part = divmod(number, 10_000)
        if part:
            digits = "".join(
                ("" if digit == 1 and place else _SINO[digit]) + _SMALL_UNITS[place]
                for place in (3, 2, 1, 0)
                if (digit := part // 10**place % 10)
            )
            if part == 1 and index == 1:
                digits = ""  # "만 원", not "일만 원"
            groups.append(digits + big)
        if not number:
            break
    return " ".join(reversed(groups))


def native(number: int) -> str:
    """Counting form, 1..99: 2 -> "두", 20 -> "스무", 21 -> "스물한". Larger numbers are Sino-Korean."""
    if not 0 < number < 100:
        return sino(number)
    if number == 20:
        return "스무"
    tens, ones = divmod(number, 10)
    return _NATIVE_TENS[tens] + _NATIVE_ONES[ones]


def digits(text: str) -> str:
    return "".join(DIGIT_NAMES[int(ch)] for ch in text if ch.isdigit())


def letters(word: str) -> str:
    known = WORDS.get(word.lower())
    return known if known else " ".join(LETTER_NAMES[ch] for ch in word.lower())


def _email(match: re.Match[str]) -> str:
    local, domain = match.group().split("@", 1)

    def spell(part: str) -> str:
        out = []
        for piece in re.findall(r"[A-Za-z]+|\d+|[._%+-]", part):
            if piece.isdigit():
                out.append(digits(piece))
            elif piece.isalpha():
                out.append(" ".join(LETTER_NAMES[ch] for ch in piece.lower()))
            else:
                out.append({".": "점", "_": "언더바", "+": "플러스", "-": "다시", "%": "퍼센트"}[piece])
        return " ".join(out)

    names = " 점 ".join(letters(label) for label in domain.split(".")).replace(" 점 컴", " 닷컴")
    return f" {spell(local)} 골뱅이 {names} "


def _identifier(match: re.Match[str]) -> str:
    parts = []
    for part in match.group().split("-"):
        pieces = re.findall(r"[A-Za-z]+|\d+", part)
        # Letter by letter: in an identifier "G" is a letter, not grams.
        parts.append(
            " ".join(
                digits(p) if p.isdigit() else " ".join(LETTER_NAMES[c] for c in p.lower()) for p in pieces
            )
        )
    return " " + " 다시 ".join(parts) + " "  # "-" is said "다시"


def _number(match: re.Match[str]) -> str:
    whole, fraction, gap, unit = match.groups()
    plain = whole.replace(",", "")
    unit = unit or ""
    if len(plain) > 16:  # beyond 조 there is no unit here: read the digits
        return f" {digits(plain)} {unit}"
    if fraction:
        return f"{sino(int(plain))} 점 {digits(fraction)}{gap}{'퍼센트' if unit == '%' else unit}"
    if unit == "%":
        return f"{sino(int(plain))} 퍼센트"
    counter = next((c for c in NATIVE_COUNTERS if unit.startswith(c)), None)
    if counter and counter not in _SINO_ONLY:
        return f"{native(int(plain))} {unit}"
    if (
        "," not in whole
        and (len(plain) >= 5 or (len(plain) > 1 and plain[0] == "0"))
        and not unit.startswith("원")
    ):
        return f" {digits(plain)} {unit}"  # a bare long number is an identifier, read digit by digit
    return f"{sino(int(plain))}{' ' if unit else ''}{unit}"


def verbalize(text: str) -> str:
    """The spoken form of `text`: Hangul, spaces and `.,?!~` only."""
    text = unicodedata.normalize("NFKC", text)
    text = _EMAIL.sub(_email, text)
    text = _PHONE.sub(lambda m: " " + ", ".join((digits("0" + m[1]), digits(m[2]), digits(m[3]))) + " ", text)
    text = _IDENT.sub(_identifier, text)
    text = _ISO_DATE.sub(lambda m: f"{m[1]}년 {int(m[3])}월 {int(m[4])}일", text)  # the tools write dates so
    text = _DATE.sub(lambda m: f"{_MONTHS.get(int(m[1]), sino(int(m[1])))}월 {sino(int(m[3]))}일", text)
    text = _MONTH.sub(lambda m: f"{_MONTHS.get(int(m[1]), sino(int(m[1])))}월", text)
    text = _CLOCK.sub(
        lambda m: f"{native(int(m[1]))} 시" + (f" {sino(int(m[2]))} 분" if int(m[2]) else ""), text
    )
    text = _RANGE.sub(lambda m: f"{sino(int(m[1]))}에서 {m[2]}", text)
    text = _MINUS.sub("마이너스 ", text)
    text = _NUMBER.sub(_number, text)
    text = _LATIN.sub(lambda m: " " + letters(m.group()) + " ", text)
    text = _KEEP.sub(" ", text)  # quotes, brackets and other symbols are not spoken
    text = re.sub(r"\s+([.,?!~])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()
