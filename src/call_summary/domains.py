"""Fictional call-center domains: the label sets a model chooses from and the scenario templates of specs.

Everything here is invented. Company names, products and plans are made up for this project.
The card domain is held out of training (test-C): same record schema, different label sets.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .values import Kind

# ---------------------------------------------------------------------------------------------------------
# Value makers: (rng) -> surface form as it is said in the call. Gold values are these surface forms; scoring
# normalizes both sides, so "32,000원" and "3만 2천 원" are the same amount.


def _amount(lo: int, hi: int, step: int) -> Callable[[random.Random], str]:
    def make(rng: random.Random) -> str:
        n = rng.randrange(lo // step, hi // step + 1) * step
        if rng.random() < 0.5 or n < 10_000:
            return f"{n:,}원"
        man, rest = divmod(n, 10_000)
        if not rest:
            return f"{man}만 원"
        if rest % 1000 == 0:
            return f"{man}만 {rest // 1000}천 원"
        return f"{man}만 {rest:,}원"

    return make


def _date(rng: random.Random) -> str:
    month = rng.choice([9, 10, 11])
    day = rng.randint(1, 30)
    return f"{month}월 {day}일"


def _time(rng: random.Random) -> str:
    hour = rng.choice([9, 10, 11, 13, 14, 15, 16, 17])
    minute = rng.choice([0, 0, 30])
    period = "오전" if hour < 12 else "오후"
    h12 = hour if hour <= 12 else hour - 12
    return f"{period} {h12}시" + (f" {minute}분" if minute else "")


def _order_id(rng: random.Random) -> str:
    return f"D{rng.randint(1_000_000, 9_999_999)}"


def _tracking(rng: random.Random) -> str:
    return f"{rng.randint(5000, 6999)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def _receipt(rng: random.Random) -> str:
    return f"R{rng.randint(100000, 999999)}"


def _last4(rng: random.Random) -> str:
    return f"{rng.randint(0, 9999):04d}"


def _pick(options: tuple[str, ...]) -> Callable[[random.Random], str]:
    return lambda rng: rng.choice(options)


@dataclass(frozen=True)
class EntityType:
    label: str
    kind: Kind
    make: Callable[[random.Random], str]


@dataclass(frozen=True)
class Outcome:
    resolution: str
    actions: tuple[str, ...]
    follow_up: tuple[str, ...]
    facts: tuple[str, ...]  # format strings over slot names
    extra_slots: tuple[tuple[str, str], ...] = ()  # (slot, entity label) only this outcome needs
    note: str = ""  # instruction to the dialogue writer about how the call ends
    weight: float = 1.0


@dataclass(frozen=True)
class Scenario:
    category: str
    slots: tuple[tuple[str, str], ...]  # (slot name, entity label); every slot's value appears in the call
    request_fact: str  # what the customer wanted, format string over slot names
    outcomes: tuple[Outcome, ...]
    needs_verification: bool = True  # account-level request: the agent may verify identity first


@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    company: str
    entity_types: tuple[EntityType, ...]
    scenarios: tuple[Scenario, ...]
    held_out: bool = False
    extra_questions: tuple[str, ...] = field(default=())  # small unrelated questions a caller may add

    def entity_type(self, label: str) -> EntityType:
        for et in self.entity_types:
            if et.label == label:
                return et
        raise KeyError(label)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(s.category for s in self.scenarios)

    @property
    def actions(self) -> tuple[str, ...]:
        seen: dict[str, None] = {a: None for a in COMMON_ACTIONS}
        for s in self.scenarios:
            for o in s.outcomes:
                for a in o.actions:
                    seen.setdefault(a, None)
        return tuple(seen)

    @property
    def follow_up_codes(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for s in self.scenarios:
            for o in s.outcomes:
                for c in o.follow_up:
                    seen.setdefault(c, None)
        return tuple(seen)


VERIFY = "본인 확인"
TRANSFER = "전문 부서 이관"
COMMON_ACTIONS = (VERIFY,)

# ---------------------------------------------------------------------------------------------------------
# 1. Online shop

_SHOP_PRODUCTS = (
    "무선 이어폰 소리방울 2",
    "접이식 캠핑 의자",
    "스테인리스 보온 텀블러",
    "극세사 차렵이불 퀸",
    "원목 모니터 받침대",
    "남성 방수 등산화 270",
    "여성 린넨 셔츠 M",
    "무선 핸디 청소기",
    "유아용 식판 세트",
    "게이밍 기계식 키보드",
    "전기 그릴 팬",
    "러닝화 250",
)

SHOP = Domain(
    key="shop",
    label="온라인 쇼핑몰",
    company="도토리마켓",
    entity_types=(
        EntityType("주문번호", "id", _order_id),
        EntityType("상품명", "text", _pick(_SHOP_PRODUCTS)),
        EntityType("금액", "amount", _amount(8_000, 250_000, 100)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
    ),
    scenarios=(
        Scenario(
            "배송 문의",
            (("order", "주문번호"), ("product", "상품명")),
            "고객이 주문 {order}({product})의 배송 상황을 문의함",
            (
                Outcome(
                    "해결",
                    ("배송 조회",),
                    (),
                    ("상담원이 배송을 조회하고 {date}에 도착할 예정이라고 안내함",),
                    extra_slots=(("date", "날짜"),),
                ),
                Outcome(
                    "부분 해결",
                    ("배송 조회", "택배사 확인 요청"),
                    ("콜백 예약",),
                    (
                        "배송이 멈춰 있어 상담원이 택배사에 확인을 요청함",
                        "확인 뒤 {time}에 고객에게 다시 연락하기로 함",
                    ),
                    extra_slots=(("time", "시간"),),
                ),
            ),
        ),
        Scenario(
            "주문 취소",
            (("order", "주문번호"), ("product", "상품명")),
            "고객이 주문 {order}({product})의 취소를 요청함",
            (
                Outcome(
                    "해결",
                    ("주문 취소", "환불 접수"),
                    ("환불 처리 대기",),
                    ("상담원이 주문을 취소하고 {amount} 환불을 접수함",),
                    extra_slots=(("amount", "금액"),),
                    weight=2,
                ),
                Outcome(
                    "미해결",
                    ("배송 조회",),
                    (),
                    (
                        "이미 출고되어 취소할 수 없다고 안내함",
                        "상품을 받은 뒤 반품을 신청하도록 안내함",
                    ),
                    note="취소는 되지 않고 고객은 아쉬워하며 끝낸다",
                ),
            ),
        ),
        Scenario(
            "반품 신청",
            (("order", "주문번호"), ("product", "상품명")),
            "고객이 주문 {order}({product})의 반품을 원함",
            (
                Outcome(
                    "해결",
                    ("반품 접수", "수거 예약"),
                    ("수거 예정",),
                    ("상담원이 반품을 접수하고 {date}에 수거하도록 예약함",),
                    extra_slots=(("date", "날짜"),),
                    weight=2,
                ),
                Outcome(
                    "미해결",
                    (),
                    (),
                    ("반품 기한이 지나 반품을 받을 수 없다고 안내함",),
                    note="반품은 거절되고 고객은 불만을 표시한 뒤 끝낸다",
                ),
            ),
        ),
        Scenario(
            "교환 신청",
            (("order", "주문번호"), ("product", "상품명")),
            "고객이 주문 {order}({product})을 다른 옵션으로 교환하길 원함",
            (
                Outcome(
                    "해결",
                    ("교환 접수", "수거 예약"),
                    ("재배송 예정",),
                    ("상담원이 교환을 접수하고 기존 상품 수거를 예약함",),
                    weight=2,
                ),
                Outcome(
                    "부분 해결",
                    ("재고 확인",),
                    ("재입고 알림",),
                    (
                        "원하는 옵션의 재고가 없어 교환을 바로 할 수 없다고 안내함",
                        "재입고되면 알림을 보내기로 함",
                    ),
                ),
            ),
        ),
        Scenario(
            "환불 문의",
            (("order", "주문번호"), ("amount", "금액")),
            "고객이 주문 {order}의 환불 금액 {amount}이 아직 들어오지 않았다고 문의함",
            (
                Outcome(
                    "해결",
                    ("환불 상태 조회",),
                    (),
                    ("상담원이 환불 상태를 조회하고 {date}까지 환불된다고 안내함",),
                    extra_slots=(("date", "날짜"),),
                    weight=2,
                ),
                Outcome(
                    "이관",
                    ("환불 상태 조회", TRANSFER),
                    ("콜백 예약",),
                    ("환불 기록이 확인되지 않아 결제 담당 부서로 이관하고 연락을 드리기로 함",),
                ),
            ),
        ),
        Scenario(
            "상품 불량",
            (("order", "주문번호"), ("product", "상품명")),
            "고객이 받은 상품 {product}(주문 {order})이 불량이라고 알림",
            (
                Outcome(
                    "해결",
                    ("불량 접수", "수거 예약"),
                    ("재배송 예정",),
                    ("상담원이 불량을 접수하고 수거 후 새 상품을 보내기로 함",),
                    weight=2,
                ),
                Outcome(
                    "이관",
                    ("불량 접수", TRANSFER),
                    ("보상 검토",),
                    ("불량으로 다친 곳이 있어 보상 담당 부서로 이관함",),
                ),
            ),
        ),
        Scenario(
            "결제 오류",
            (("amount", "금액"), ("date", "날짜")),
            "고객이 {date}에 {amount}이 두 번 결제되었다고 문의함",
            (
                Outcome(
                    "해결",
                    ("결제 내역 확인", "결제 취소"),
                    ("환불 처리 대기",),
                    ("상담원이 중복 결제를 확인하고 한 건을 취소함",),
                    weight=2,
                ),
                Outcome(
                    "부분 해결",
                    ("결제 내역 확인",),
                    ("콜백 예약",),
                    ("결제사 확인이 필요해 {time}에 다시 연락하기로 함",),
                    extra_slots=(("time", "시간"),),
                ),
            ),
        ),
        Scenario(
            "적립금·쿠폰 문의",
            (("amount", "금액"),),
            "고객이 적립금 {amount}이 사라졌다고 문의함",
            (
                Outcome(
                    "해결",
                    ("적립금 조회", "쿠폰 발급"),
                    (),
                    (
                        "상담원이 적립금 유효기간이 지나 소멸되었다고 안내함",
                        "대신 사과의 뜻으로 할인 쿠폰을 발급함",
                    ),
                ),
                Outcome(
                    "해결",
                    ("적립금 조회",),
                    (),
                    ("상담원이 적립금이 {date}에 들어올 예정이라고 안내함",),
                    extra_slots=(("date", "날짜"),),
                ),
            ),
        ),
    ),
    extra_questions=("앱에서 주문 내역 보는 법", "회원 등급 기준", "앱 알림 끄는 법"),
)

# ---------------------------------------------------------------------------------------------------------
# 2. Mobile carrier

_PLANS = (
    "별빛 5G 슬림 55",
    "별빛 5G 스탠다드 75",
    "별빛 5G 프리미엄 95",
    "별빛 LTE 데이터 무제한 69",
    "별빛 LTE 세이브 33",
    "별빛 시니어 29",
)
_DEVICES = ("은하폰 S12", "은하폰 A7", "사과폰 16", "사과폰 15 미니", "은하 탭 9")
_ADDONS = ("음악 스트리밍 팩", "영상 무제한 팩", "휴대폰 보험 안심", "통화 연결음 서비스", "해외 로밍 패스")

TELECOM = Domain(
    key="telecom",
    label="이동통신사",
    company="별빛텔레콤",
    entity_types=(
        EntityType("요금제", "text", _pick(_PLANS)),
        EntityType("기기명", "text", _pick(_DEVICES)),
        EntityType("부가서비스", "text", _pick(_ADDONS)),
        EntityType("금액", "amount", _amount(3_000, 150_000, 100)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
        EntityType("접수번호", "id", _receipt),
    ),
    scenarios=(
        Scenario(
            "요금 문의",
            (("amount", "금액"),),
            "고객이 이번 달 청구 요금 {amount}이 평소보다 많다고 문의함",
            (
                Outcome(
                    "해결",
                    ("요금 내역 조회",),
                    (),
                    ("상담원이 요금 내역을 조회해 소액결제가 더해졌다고 설명함",),
                    weight=2,
                ),
                Outcome(
                    "부분 해결",
                    ("요금 내역 조회", "요금 이의 신청"),
                    ("콜백 예약",),
                    ("원인이 확인되지 않아 이의 신청을 접수하고 {time}에 다시 연락하기로 함",),
                    extra_slots=(("time", "시간"),),
                ),
            ),
        ),
        Scenario(
            "요금제 변경",
            (("plan_old", "요금제"), ("plan_new", "요금제")),
            "고객이 {plan_old}에서 {plan_new}로 요금제를 바꾸길 원함",
            (
                Outcome(
                    "해결",
                    ("요금제 변경",),
                    (),
                    ("상담원이 요금제를 바로 변경함",),
                    weight=2,
                ),
                Outcome(
                    "부분 해결",
                    ("요금제 변경 예약",),
                    (),
                    ("이번 달 변경이 제한되어 {date}부터 적용되도록 예약함",),
                    extra_slots=(("date", "날짜"),),
                ),
            ),
        ),
        Scenario(
            "데이터 소진",
            (("plan", "요금제"),),
            "고객이 {plan} 요금제의 데이터가 다 떨어져 속도가 느리다고 문의함",
            (
                Outcome(
                    "해결",
                    ("데이터 충전",),
                    (),
                    ("상담원이 {amount}짜리 데이터 충전을 해 줌",),
                    extra_slots=(("amount", "금액"),),
                ),
                Outcome(
                    "해결",
                    ("요금제 변경",),
                    (),
                    ("상담원 안내로 요금제를 {plan_new}로 변경함",),
                    extra_slots=(("plan_new", "요금제"),),
                ),
            ),
        ),
        Scenario(
            "해지 문의",
            (("amount", "금액"),),
            "고객이 해지를 고민하며 위약금을 문의함",
            (
                Outcome(
                    "미해결",
                    ("위약금 조회",),
                    ("콜백 예약",),
                    (
                        "상담원이 위약금이 {amount}이라고 안내함",
                        "고객이 더 생각해 보겠다고 해 {time}에 다시 연락하기로 함",
                    ),
                    extra_slots=(("time", "시간"),),
                ),
                Outcome(
                    "이관",
                    ("위약금 조회", TRANSFER),
                    (),
                    ("위약금 {amount} 안내 후 고객이 해지를 원해 해지 전담 부서로 이관함",),
                ),
            ),
        ),
        Scenario(
            "통신 장애",
            (("date", "날짜"),),
            "고객이 {date}부터 집에서 전화와 데이터가 잘 안 된다고 신고함",
            (
                Outcome(
                    "부분 해결",
                    ("장애 접수", "기사 방문 예약"),
                    ("기사 방문 예정",),
                    ("상담원이 장애를 접수하고 기사가 {visit_date} {time}에 방문하도록 예약함",),
                    extra_slots=(("visit_date", "날짜"), ("time", "시간")),
                    weight=2,
                ),
                Outcome(
                    "해결",
                    ("장애 접수", "원격 점검"),
                    (),
                    ("원격 점검으로 설정을 초기화하자 통화가 정상이 됨",),
                ),
            ),
        ),
        Scenario(
            "분실·정지",
            (("device", "기기명"),),
            "고객이 휴대폰 {device}을 잃어버렸다고 알림",
            (
                Outcome(
                    "해결",
                    ("분실 정지",),
                    (),
                    ("상담원이 회선을 분실 정지함", "기기를 찾으면 정지를 풀 수 있다고 안내함"),
                    weight=2,
                ),
                Outcome(
                    "부분 해결",
                    ("분실 정지", "임대폰 신청"),
                    ("임대폰 배송",),
                    ("회선을 분실 정지하고 임대폰을 {date}까지 보내기로 함",),
                    extra_slots=(("date", "날짜"),),
                ),
            ),
        ),
        Scenario(
            "부가서비스 해지",
            (("addon", "부가서비스"), ("amount", "금액")),
            "고객이 가입한 적 없는 {addon}이 월 {amount}씩 청구된다고 문의함",
            (
                Outcome(
                    "해결",
                    ("부가서비스 해지", "요금 환급 접수"),
                    ("환불 처리 대기",),
                    ("상담원이 부가서비스를 해지하고 청구된 요금의 환급을 접수함",),
                    weight=2,
                ),
                Outcome(
                    "해결",
                    ("부가서비스 해지",),
                    (),
                    ("가입 기록이 확인되어 환급은 안 되고 부가서비스만 해지함",),
                ),
            ),
        ),
        Scenario(
            "기기 할부 문의",
            (("device", "기기명"),),
            "고객이 {device}의 남은 할부금을 문의함",
            (
                Outcome(
                    "해결",
                    ("할부 내역 조회",),
                    (),
                    ("상담원이 남은 할부금이 {amount}이고 {date}에 끝난다고 안내함",),
                    extra_slots=(("amount", "금액"), ("date", "날짜")),
                    weight=2,
                ),
                Outcome(
                    "해결",
                    ("할부 내역 조회", "할부금 완납 접수"),
                    (),
                    ("고객이 남은 할부금 {amount}을 한 번에 내기로 해 완납을 접수함",),
                    extra_slots=(("amount", "금액"),),
                ),
            ),
        ),
    ),
    extra_questions=("가까운 대리점 위치", "멤버십 포인트 사용처", "와이파이 비밀번호 바꾸는 법"),
)

# ---------------------------------------------------------------------------------------------------------
# 3. Parcel delivery

_PARCEL_ITEMS = (
    "전자레인지",
    "노트북",
    "김치 10kg",
    "도자기 그릇 세트",
    "의류 상자",
    "책 상자",
    "전기 자전거 배터리",
    "화분",
)

PARCEL = Domain(
    key="parcel",
    label="택배사",
    company="새싹택배",
    entity_types=(
        EntityType("운송장번호", "id", _tracking),
        EntityType("물품명", "text", _pick(_PARCEL_ITEMS)),
        EntityType("금액", "amount", _amount(3_000, 1_500_000, 1000)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
    ),
    scenarios=(
        Scenario(
            "배송 조회",
            (("tracking", "운송장번호"),),
            "고객이 운송장 {tracking}의 배송 위치를 문의함",
            (
                Outcome(
                    "해결",
                    ("배송 조회",),
                    (),
                    ("상담원이 조회 결과 {date}에 배달될 예정이라고 안내함",),
                    extra_slots=(("date", "날짜"),),
                ),
            ),
            needs_verification=False,
        ),
        Scenario(
            "배송 지연",
            (("tracking", "운송장번호"), ("date", "날짜")),
            "고객이 {date}에 온다던 운송장 {tracking} 택배가 아직 안 왔다고 문의함",
            (
                Outcome(
                    "부분 해결",
                    ("배송 조회", "지점 확인 요청"),
                    ("콜백 예약",),
                    ("상담원이 담당 지점에 확인을 요청하고 {time}에 다시 연락하기로 함",),
                    extra_slots=(("time", "시간"),),
                    weight=2,
                ),
                Outcome(
                    "해결",
                    ("배송 조회",),
                    (),
                    ("물량이 몰려 하루 늦어졌고 {new_date}에 배달된다고 안내함",),
                    extra_slots=(("new_date", "날짜"),),
                ),
            ),
            needs_verification=False,
        ),
        Scenario(
            "파손 신고",
            (("tracking", "운송장번호"), ("item", "물품명")),
            "고객이 운송장 {tracking}으로 받은 {item}이 파손되었다고 신고함",
            (
                Outcome(
                    "부분 해결",
                    ("파손 접수",),
                    ("보상 검토", "서류 제출 대기"),
                    (
                        "상담원이 파손을 접수함",
                        "물품 가격 {amount}에 대한 영수증과 사진을 보내 달라고 안내함",
                    ),
                    extra_slots=(("amount", "금액"),),
                    weight=2,
                ),
                Outcome(
                    "이관",
                    ("파손 접수", TRANSFER),
                    ("보상 검토",),
                    ("고가 물품이라 보상 전담 부서로 이관함",),
                ),
            ),
        ),
        Scenario(
            "분실 신고",
            (("tracking", "운송장번호"), ("item", "물품명")),
            "고객이 배송 완료로 나오는 운송장 {tracking}의 {item}을 받지 못했다고 신고함",
            (
                Outcome(
                    "이관",
                    ("분실 접수", TRANSFER),
                    ("보상 검토",),
                    ("상담원이 분실을 접수하고 사고 처리 부서로 이관함",),
                    weight=2,
                ),
                Outcome(
                    "해결",
                    ("배송 조회",),
                    (),
                    ("조회해 보니 경비실에 맡겨져 있다고 안내함",),
                ),
            ),
        ),
        Scenario(
            "반품 수거 요청",
            (("tracking", "운송장번호"),),
            "고객이 운송장 {tracking} 물건을 반품하려고 수거를 요청함",
            (
                Outcome(
                    "해결",
                    ("수거 예약",),
                    ("수거 예정",),
                    ("상담원이 {date}에 수거하도록 예약함",),
                    extra_slots=(("date", "날짜"),),
                ),
            ),
            needs_verification=False,
        ),
        Scenario(
            "배송 일정 변경",
            (("tracking", "운송장번호"), ("date", "날짜")),
            "고객이 운송장 {tracking} 택배를 {date}에 받고 싶다고 요청함",
            (
                Outcome(
                    "해결",
                    ("배송 일정 변경",),
                    (),
                    ("상담원이 배송일을 요청한 날로 바꿔 줌",),
                    weight=2,
                ),
                Outcome(
                    "미해결",
                    ("배송 조회",),
                    (),
                    ("이미 배송 출발해 일정을 바꿀 수 없다고 안내함",),
                ),
            ),
            needs_verification=False,
        ),
        Scenario(
            "택배 보내기 예약",
            (("item", "물품명"),),
            "고객이 {item}을 보내려고 방문 접수를 원함",
            (
                Outcome(
                    "해결",
                    ("방문 접수 예약",),
                    ("수거 예정",),
                    ("상담원이 {date}에 방문 접수를 예약하고 요금이 {amount}이라고 안내함",),
                    extra_slots=(("date", "날짜"), ("amount", "금액")),
                ),
            ),
            needs_verification=False,
        ),
        Scenario(
            "오배송",
            (("tracking", "운송장번호"), ("item", "물품명")),
            "고객이 운송장 {tracking}으로 주문하지 않은 {item}이 왔다고 알림",
            (
                Outcome(
                    "부분 해결",
                    ("오배송 접수", "회수 요청"),
                    ("재배송 예정",),
                    ("상담원이 오배송을 접수하고 잘못 온 물건의 회수와 원래 물건 배송을 요청함",),
                ),
            ),
            needs_verification=False,
        ),
    ),
    extra_questions=("편의점 택배 접수 가능 여부", "일요일 배송 여부", "포장 박스 구매 방법"),
)

# ---------------------------------------------------------------------------------------------------------
# 4. Credit card (held out of training)

_MERCHANTS = ("솔바람 카페", "한결 마트", "푸른숲 주유소", "모래시계 서점", "달빛 호텔", "해오름 온라인몰")

CARD = Domain(
    key="card",
    label="카드사",
    company="구름카드",
    entity_types=(
        EntityType("카드 끝자리", "id", _last4),
        EntityType("가맹점명", "text", _pick(_MERCHANTS)),
        EntityType("금액", "amount", _amount(5_000, 3_000_000, 100)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
    ),
    scenarios=(
        Scenario(
            "카드 분실",
            (("last4", "카드 끝자리"),),
            "고객이 끝자리 {last4} 카드를 잃어버렸다고 신고함",
            (
                Outcome(
                    "해결",
                    ("분실 신고", "카드 정지", "재발급 신청"),
                    ("재발급 카드 배송",),
                    ("상담원이 카드를 정지하고 재발급을 신청함",),
                ),
            ),
        ),
        Scenario(
            "결제 취소 확인",
            (("merchant", "가맹점명"), ("amount", "금액"), ("date", "날짜")),
            "고객이 {date} {merchant}에서 취소한 {amount} 결제가 아직 청구된다고 문의함",
            (
                Outcome(
                    "해결",
                    ("결제 내역 확인",),
                    (),
                    ("상담원이 가맹점 취소가 접수되어 다음 청구서에서 빠진다고 안내함",),
                    weight=2,
                ),
                Outcome(
                    "부분 해결",
                    ("결제 내역 확인", "가맹점 확인 요청"),
                    ("콜백 예약",),
                    ("취소 기록이 없어 가맹점 확인을 요청하고 {time}에 다시 연락하기로 함",),
                    extra_slots=(("time", "시간"),),
                ),
            ),
        ),
        Scenario(
            "부정 사용 의심",
            (("merchant", "가맹점명"), ("amount", "금액")),
            "고객이 쓴 적 없는 {merchant} {amount} 결제 문자를 받았다고 신고함",
            (
                Outcome(
                    "이관",
                    ("카드 정지", TRANSFER),
                    ("보상 검토",),
                    ("상담원이 카드를 바로 정지하고 부정 사용 조사 부서로 이관함",),
                ),
            ),
        ),
        Scenario(
            "한도 상향",
            (("amount", "금액"),),
            "고객이 카드 한도를 {amount}으로 올리고 싶다고 요청함",
            (
                Outcome(
                    "부분 해결",
                    ("한도 심사 접수",),
                    ("서류 제출 대기",),
                    ("상담원이 한도 심사를 접수하고 소득 증빙 서류를 내 달라고 안내함",),
                    weight=2,
                ),
                Outcome(
                    "해결",
                    ("한도 상향",),
                    (),
                    ("상담원이 바로 한도를 올려 줌",),
                ),
            ),
        ),
        Scenario(
            "청구 금액 문의",
            (("amount", "금액"),),
            "고객이 이번 달 카드 청구 금액 {amount}을 확인하고 싶어 함",
            (
                Outcome(
                    "해결",
                    ("청구 내역 조회",),
                    (),
                    ("상담원이 청구 내역을 확인해 주고 결제일이 {date}이라고 안내함",),
                    extra_slots=(("date", "날짜"),),
                ),
            ),
        ),
        Scenario(
            "결제일 변경",
            (("date", "날짜"),),
            "고객이 카드 결제일을 바꾸고 싶다고 요청함",
            (
                Outcome(
                    "해결",
                    ("결제일 변경",),
                    (),
                    ("상담원이 다음 달부터 결제일을 바꾸고 이번 결제일은 {date}이라고 안내함",),
                ),
            ),
        ),
        Scenario(
            "연회비 문의",
            (("amount", "금액"),),
            "고객이 연회비 {amount}이 청구된 이유를 문의함",
            (
                Outcome(
                    "해결",
                    ("연회비 조회",),
                    (),
                    ("상담원이 카드 발급 기념일에 연회비가 청구된다고 설명함",),
                ),
                Outcome(
                    "부분 해결",
                    ("연회비 조회", "연회비 면제 신청"),
                    ("콜백 예약",),
                    ("면제 조건 심사를 접수하고 {time}에 결과를 알려 주기로 함",),
                    extra_slots=(("time", "시간"),),
                ),
            ),
        ),
        Scenario(
            "할부 전환",
            (("merchant", "가맹점명"), ("amount", "금액")),
            "고객이 {merchant}에서 일시불로 결제한 {amount}을 할부로 바꾸고 싶어 함",
            (
                Outcome(
                    "해결",
                    ("할부 전환",),
                    (),
                    ("상담원이 결제를 할부로 전환함",),
                ),
            ),
        ),
    ),
    held_out=True,
    extra_questions=("포인트 사용처", "해외 결제 수수료", "카드 앱 비밀번호 변경"),
)

DOMAINS: dict[str, Domain] = {d.key: d for d in (SHOP, TELECOM, PARCEL, CARD)}
TRAIN_DOMAINS: tuple[str, ...] = tuple(k for k, d in DOMAINS.items() if not d.held_out)
HELD_OUT_DOMAINS: tuple[str, ...] = tuple(k for k, d in DOMAINS.items() if d.held_out)
