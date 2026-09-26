import pytest

from call_summary.values import normalize, occurs_in_transcript, parse_korean_number, values_in_transcript


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
        ("id", "915-4821-33", "915 4821 33"),
        ("id", "r123456", "R123456"),
        ("text", "무선 핸디 청소기", "무선핸디청소기"),
    ],
)
def test_normalize_equivalent(kind, a, b):
    assert normalize(kind, a) is not None
    assert normalize(kind, a) == normalize(kind, b)


def test_normalize_distinguishes():
    assert normalize("amount", "32,000원") != normalize("amount", "23,000원")
    assert normalize("time", "오전 2시") != normalize("time", "오후 2시")
    assert normalize("date", "9월 28일") != normalize("date", "9월 29일")
    assert normalize("amount", "많이") is None
    assert normalize("date", "내일") is None


TRANSCRIPT = (
    "상담원: 주문번호 1015-4821-33 맞으시죠?\n"
    "고객: 네. 결제 금액은 3만 2천 원이었고요, 적립금은 1,500원이에요.\n"
    "상담원: 10월 3일 오후 2시 반에 연락드리겠습니다."
)


def test_values_in_transcript():
    amounts = values_in_transcript("amount", TRANSCRIPT)
    assert {"32000", "1500"} <= amounts
    assert values_in_transcript("date", TRANSCRIPT) == {"10-03"}
    assert values_in_transcript("time", TRANSCRIPT) == {"14:30"}


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
