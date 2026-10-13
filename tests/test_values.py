# [테스트 묶음] 값 정규화 규칙(call_summary.values)을 검증한다: 한글·아라비아 숫자가 섞인 수 읽기, 종류별로
# 같은 값이 같게 정규화되는지와 다른 값이 구분되는지, 전사에서 금액·날짜·시각 뽑기, 값이 전사에 있는지 찾기.
# [왜 중요한가] 명세는 같은 값을 여러 말투로 뽑고("32,000원" / "3만 2천 원"), 채점·환각 검사·생성 검사가 모두
# 이 정규화로 값을 비교한다(docs/experiments.md "공통"). 같은 값을 다르게 보거나 다른 값을 같게 보면 모든
# 지표가 틀어진다.

import pytest

from call_summary.values import normalize, occurs_in_transcript, parse_korean_number, values_in_transcript


# parse_korean_number: 숫자, 천 단위 쉼표, 만·억 단위, 한글 수사, 띄어쓰기 유무, 단위만 있는 "만"(10000)을
# 읽고, 수가 아닌 글자나 빈 글은 None이다.
@pytest.mark.parametrize(
    "text,expected",
    [
        ("32000", 32000),
        ("32,000", 32000),
        ("3만 2천", 32000),
        ("3만2천", 32000),
        ("삼만 이천", 32000),
        ("1억 2,500만", 125_000_000),
        ("만", 10000),
        ("이천오백", 2500),
        ("12만 3천 400", 123400),
        ("abc", None),
        ("", None),
    ],
)
def test_parse_korean_number(text, expected):
    assert parse_korean_number(text) == expected


# 종류마다 같은 값으로 봐야 하는 쌍: 금액의 말투, 날짜의 "9월 28일"/"09-28"/"2026-09-28"/"9/28"(연도 무시),
# 시각의 "오후 2시 30분"/"오후 2시 반"/"14:30", 번호의 대소문자·공백, 이름의 띄어쓰기.
@pytest.mark.parametrize(
    "kind,a,b",
    [
        ("amount", "32,000원", "3만 2천 원"),
        ("amount", "32000", "삼만 이천원"),
        ("date", "9월 28일", "09-28"),
        ("date", "2026-09-28", "9/28"),
        ("time", "오후 2시 30분", "14:30"),
        ("time", "오후 2시 반", "14:30"),
        ("time", "오전 10시", "10:00"),
        ("id", "D3928174", "d 392 8174"),
        ("id", "r123456", "R123456"),
        ("text", "무선 핸디 청소기", "무선핸디청소기"),
    ],
)
def test_normalize_equivalent(kind, a, b):
    assert normalize(kind, a) is not None
    assert normalize(kind, a) == normalize(kind, b)


# 다른 값은 구분해야 한다: 다른 금액, 오전·오후가 다른 시각, 다른 날짜. 수가 아닌 금액("많이")과 상대
# 날짜("내일")는 읽을 수 없어 None이다.
def test_normalize_distinguishes():
    assert normalize("amount", "32,000원") != normalize("amount", "23,000원")
    assert normalize("time", "오전 2시") != normalize("time", "오후 2시")
    assert normalize("date", "9월 28일") != normalize("date", "9월 29일")
    assert normalize("amount", "많이") is None
    assert normalize("date", "내일") is None


# 번호·금액 두 개·날짜·시각이 든 짧은 전사. 주문번호의 하이픈 묶음("4821-33")이 날짜로 잘못 읽히지 않는지도
# 함께 본다.
TRANSCRIPT = (
    "상담원: 주문번호 1015-4821-33 맞으시죠?\n"
    "고객: 네. 결제 금액은 3만 2천 원이었고요, 적립금은 1,500원이에요.\n"
    "상담원: 10월 3일 오후 2시 반에 연락드리겠습니다."
)


# 전사에서 금액(32000, 1500)을 뽑고, 날짜는 "10월 3일" 하나만(주문번호 조각은 날짜가 아님), 시각은 "오후 2시
# 반" = 14:30 하나만 뽑는다.
def test_values_in_transcript():
    amounts = values_in_transcript("amount", TRANSCRIPT)
    assert {"32000", "1500"} <= amounts
    assert values_in_transcript("date", TRANSCRIPT) == {"10-03"}
    assert values_in_transcript("time", TRANSCRIPT) == {"14:30"}


# occurs_in_transcript(쓴 글 규칙): 하이픈 없이 적은 같은 번호는 찾고 끝 숫자가 다르면 못 찾는다. 금액·날짜는
# 값으로 비교해 100원·하루만 달라도 못 찾는다. 시각은 "14:30"으로 적어도 찾고, 이름은 부분 문자열로 찾는다.
# "알 수 없음" 같은 자리 표시 값은 금액으로 읽을 수 없어 찾지 못한 값(환각)이 된다.
@pytest.mark.parametrize(
    "kind,value,found",
    [
        ("id", "1015482133", True),
        ("id", "1015-4821-34", False),
        ("amount", "32,000원", True),
        ("amount", "32,100원", False),
        ("date", "10월 3일", True),
        ("date", "10월 4일", False),
        ("time", "14:30", True),
        ("text", "적립금", True),
        ("text", "쿠폰", False),
        ("amount", "알 수 없음", False),
    ],
)
def test_occurs_in_transcript(kind, value, found):
    assert occurs_in_transcript(kind, value, TRANSCRIPT) is found
