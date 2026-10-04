"""call_summary.id_itn: identifiers read digit by digit, written back as digits (speech condition v3).

The forms below are written by hand from how the v2 verbalizer reads identifiers (digit names, a "디"/"알"
letter name in front, a space before the particle). They were fixed before any Qwen3-ASR output was seen.
"""

import pytest

from call_summary.id_itn import ITN_VERSION, id_itn
from call_summary.values import occurs_in_transcript


@pytest.mark.parametrize(
    "heard,want",
    [
        # order numbers: D + 7 digits
        ("주문번호는 디 사사칠칠팔삼팔 입니다.", "주문번호는 D4477838 입니다."),
        ("주문번호는 디사사칠칠팔삼팔입니다.", "주문번호는 D4477838입니다."),
        ("주문번호가 디 사사칠칠팔일팔이에요.", "주문번호가 D4477818이에요."),
        ("디 사사칠칠팔일팔 번 주문한 전기 그릴 팬", "D4477818 번 주문한 전기 그릴 팬"),
        ("주문 번호 D 사삼칠육팔공오 인 등산화", "주문 번호 D4376805 인 등산화"),
        ("디 사삼칠 육팔공오요", "D4376805요"),  # separators inside a prefixed number go
        ("디 4477838 맞으시죠?", "D4477838 맞으시죠?"),  # the letter name alone
        # receipt numbers: R + 6 digits
        ("접수번호는 알 일이삼사오육 입니다", "접수번호는 R123456 입니다"),
        # tracking numbers: 12 digits in three groups, separators kept
        (
            "운송장 번호 육이구일, 칠팔칠칠, 오일육팔 로 받은 건이에요",
            "운송장 번호 6291, 7877, 5168 로 받은 건이에요",
        ),
        (
            "운송장 번호 육이구일, 칠팔칠칠, 오일육팔로 받은 건이에요",
            "운송장 번호 6291, 7877, 5168로 받은 건이에요",
        ),
        ("오팔구이 육팔구공 이이팔칠 이 맞습니다.", "5892 6890 2287 이 맞습니다."),  # the particle "이" stays
        (
            "오팔구이육팔구공이이팔칠이 맞습니다.",
            "589268902287이 맞습니다.",
        ),  # 13 names: the 13th is the particle
        ("오팔구이-육팔구공-이이팔칠이에요", "5892-6890-2287이에요"),
        ("일공공공, 공이공삼, 공공사공 입니다.", "1000, 0203, 0040 입니다."),
        ("영일영 이삼사오", "010 2345"),  # 영 is 0 as well
        # digits and digit names mixed (the base recogniser writes these)
        ("627일 8127 2631로 보냈어요", "6271 8127 2631로 보냈어요"),
        ("5874, 351일 2946 맞나요?", "5874, 3511 2946 맞나요?"),
        ("651일 5502-2932", "6511 5502-2932"),
        ("6627, 6706, 518이 다 맞습니다.", "6627, 6706, 5182 다 맞습니다."),  # "5182가" heard as one "이"
        ("운송장 번호가 623일 469일 534일이라고 들었는데", "운송장 번호가 6231 4691 5341이라고 들었는데"),
        ("생년월일은 94121일입니다.", "생년월일은 941211입니다."),
        ("주문번호가 D247814일이었어요.", "주문번호가 D2478141이었어요."),
        # card last four digits
        ("카드 끝자리는 일이삼사요", "카드 끝자리는 1234요"),
        ("오일육칠이이에요", "51672이에요"),  # one particle comes off, not two
        ("이 오팔구이, 육팔구공, 이이팔칠 번호로", "이 5892, 6890, 2287 번호로"),  # "이" = "this"
        ("카드 끝자리 일 이 삼 사 입니다", "카드 끝자리 1 2 3 4 입니다"),  # a name per word
        (
            "육 이 구 일 칠 팔 칠 칠 오 일 육 이 맞습니다",
            "6 2 9 1 7 8 7 7 5 1 6 2 맞습니다",
        ),  # 12 with the 이
        ("육이구일, 칠팔칠칠, 오일육팔 3시에 도착", "6291, 7877, 5168 3시에 도착"),  # "3시" is a word
    ],
)
def test_identifiers_read_digit_by_digit_become_digits(heard, want):
    assert id_itn(heard) == want


@pytest.mark.parametrize(
    "heard",
    [
        "11월 29일에 도착 예정이에요.",
        "10월 15일 3시에 다시 연락드릴게요.",  # a date next to a time: 3 digit characters
        "25일 100만 원이 청구됐어요.",  # a date, then an amount: the day blocks the run
        "31일 2,500원이 빠져나갔어요",
        "1월 2일 1234번 창구",  # "2일 1234": a day in front
        "사이즈가 안 맞아서 이사 가기 전에 반품할게요",
        "일일이 확인해 드릴게요. 오일 교환은 구매처에서 해요.",
        "일이 있어서 삼십 분 뒤에 전화할게요.",
        "오십구만 오천 원입니다.",
        "이천이십육년 시월 십사일",
        "어디 사세요? 알 수 없음",
        "D4477838입니다.",
        "B4477838이라는 주문번호",  # a misheard letter is not this rule's business
        "5891-6890-2287의 배송 위치",
        "52이고 64675677",  # "이고": a particle after a number
        "삼사일 걸려요",  # three digit names: too short for an identifier
        "디 사사칠 번",  # three digits after 디: not an order number, and too short alone
        # words after a number (written dev/train text and base-recogniser transcripts, 2026-10-05)
        "운송장번호 6245-9828-3038이 오는데, 9월 10일에 받고 싶어요",
        "죄송합니다, 고객님. D8296280이 일치하지 않는 주문 번호인 것 같아요.",
        "운송장번호는 5978-1873-3477이구요, 주문하신 건 노트북이에요.",
        "네, 주문번호는 D2959626이구요. 생년월일은 871203입니다.",
        "운송장 번호가 5609, 7857, 981 팔린 상자를 받았어요",
        "고객님, 6161, 561, 90715 오배송이 접수되었습니다.",
        "데이터를 충전해 드릴까요? 이 15,900원짜리 데이터가 어떠세요?",
        "",
    ],
)
def test_everything_else_is_left_alone(heard):
    assert id_itn(heard) == heard


def test_a_prefix_of_the_wrong_length_is_kept_and_the_digits_judged_alone():
    # 디 + 8 names is not an order number; the 8 names are still a number read digit by digit
    assert id_itn("디 사사칠칠팔삼팔구 요") == "디 44778389 요"


def test_converted_identifiers_are_found_by_the_scorer():
    text = id_itn(
        "상담원: 운송장 번호 육이구일, 칠팔칠칠, 오일육팔 로 조회할게요\n고객: 디 사사칠칠팔삼팔 이에요"
    )
    assert occurs_in_transcript("id", "6291-7877-5168", text, spoken=True)
    assert occurs_in_transcript("id", "D4477838", text, spoken=True)
    assert not occurs_in_transcript("id", "6291-7877-5168", "육이구일, 칠팔칠칠, 오일육팔", spoken=True)


def test_idempotent_and_versioned():
    once = id_itn("운송장 육이구일, 칠팔칠칠, 오일육팔 이고 주문은 디 사사칠칠팔삼팔")
    assert id_itn(once) == once
    assert ITN_VERSION == "id-itn-1"
