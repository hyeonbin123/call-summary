"""가상의 콜센터 업종: 모델이 고르는 라벨 목록과 명세의 시나리오 틀.

여기 있는 것은 모두 지어낸 것이다. 회사 이름, 상품, 요금제는 이 프로젝트를 위해 만들었다.
카드 업종은 학습에서 뺀다(test-C): 기록 스키마는 같고 라벨 목록만 다르다.
"""

# [파일 역할] 가상 업종 네 개(쇼핑몰, 통신사, 택배사, 카드사)의 정의를 담은 데이터 모듈이다. 업종마다 회사
# 이름, 핵심 값 유형(과 값 생성기), 문의 유형(시나리오)별 결과 후보(처리 결과, 처리 목록, 후속 조치, 요약에
# 들어갈 사실)를 정한다.
# [왜 필요한가] 이 프로젝트는 정답을 먼저 만들고 대화를 나중에 만든다. specs.py가 이 표에서 시나리오·결과·값을
# 뽑아 명세를 만들고, 구조 필드 정답은 명세에서 바로 나온다. 같은 표가 프롬프트(prompts.py)에 넣는 라벨 목록,
# 채점의 값 종류, 서비스의 번호 형식 검사(verify.py)에도 쓰여, 라벨 정의가 한곳에만 있다.
# [설계 이유] 실제 회사 이름·상품·약관을 쓰지 않으려고 모두 지어냈다(README, 모듈 docstring). 카드 업종은
# 학습에서 빼(held_out) test-c로 새 업종 일반화를 잰다: 스키마는 같고 라벨 목록만 다르다.
# [주요 구성] 값 생성기 → 자료 구조(EntityType, Outcome, Scenario, Domain) → 업종 넷 → DOMAINS 표, catalog,
# TRAIN/HELD_OUT_DOMAINS.

from __future__ import annotations

# 값 생성기는 모두 seed를 받은 random.Random을 인자로 받아, 명세 생성이 결정적이게 한다. Kind: 값 종류(id,
# amount, date, time, text). 채점의 정규화 방식이 종류마다 다르다.
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .values import Kind

# ---------------------------------------------------------------------------------------------------------
# 값 생성기: (rng) -> 통화에서 말하는 그대로의 겉꼴. 정답 값은 이 겉꼴이며, 채점은 양쪽을 정규화하므로
# "32,000원"과 "3만 2천 원"은 같은 금액이다.


# 금액 생성기를 만든다: [lo, hi] 범위에서 step 단위로 뽑고, 절반은 "32,000원", 절반은 "3만 2천 원"처럼 읽는
# 꼴로 적는다(1만 원 미만은 늘 숫자 꼴). 같은 금액의 여러 말투를 섞어, 모델이 표기가 달라도 같은 값으로
# 옮기는지 시험한다(README "핵심 값과 그 말투"). 채점은 양쪽을 정규화해 같은 금액으로 본다.
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


# 날짜: 9~11월 중 1~30일, "10월 5일" 꼴. 해는 넣지 않는다. 9~11월만 뽑아 해를 넘기지 않으므로 specs.py가
# 정규화한 "MM-DD" 문자열 비교로 날짜 순서를 맞출 수 있다.
def _date(rng: random.Random) -> str:
    month = rng.choice([9, 10, 11])
    day = rng.randint(1, 30)
    return f"{month}월 {day}일"


# 시각: 9~17시(12시 제외) 중, 분은 0분이 30분보다 두 배 자주 나온다. "오후 2시 30분" 꼴로 오전·오후를
# 붙인다(채점은 오전·오후까지 맞아야 같은 시각으로 본다).
def _time(rng: random.Random) -> str:
    hour = rng.choice([9, 10, 11, 13, 14, 15, 16, 17])
    minute = rng.choice([0, 0, 30])
    period = "오전" if hour < 12 else "오후"
    h12 = hour if hour <= 12 else hour - 12
    return f"{period} {h12}시" + (f" {minute}분" if minute else "")


# 주문번호: D + 7자리. 첫 파일럿에서 "915-4821-33" 꼴로 했더니 교사가 번호를 줄여 말해 9건 중 6건이 거절돼 이
# 꼴로 바꿨다(docs/experiments.md 1단계 파일럿).
def _order_id(rng: random.Random) -> str:
    return f"D{rng.randint(1_000_000, 9_999_999)}"


# 운송장번호: 4-4-4 하이픈 꼴의 12자리. 음성 조건 v2는 이 꼴을 숫자 하나씩 읽는다.
def _tracking(rng: random.Random) -> str:
    return f"{rng.randint(5000, 6999)}-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


# 접수번호(통신사): R + 6자리. 카드 끝자리(카드사): 0을 앞에 채운 4자리.
def _receipt(rng: random.Random) -> str:
    return f"R{rng.randint(100000, 999999)}"


def _last4(rng: random.Random) -> str:
    return f"{rng.randint(0, 9999):04d}"


# 닫힌 목록(카탈로그)에서 고르는 값 생성기. 함수(클로저) 대신 dataclass로 둔 것은 목록(options)을 밖에서 읽을
# 수 있게 하려는 것이다: catalog()가 이것을 읽어 음성 조건 v3의 업종 맥락(hotwords, 프롬프트)을
# 만든다(docstring).
@dataclass(frozen=True)
class _Pick:
    """닫힌 목록(카탈로그)에서 고르는 값 생성기. 음성 조건 v3는 이 목록을 맥락으로 넘긴다."""

    options: tuple[str, ...]

    def __call__(self, rng: random.Random) -> str:
        return rng.choice(self.options)


# _Pick을 만드는 도우미(다른 생성기와 같은 "rng → 문자열" 꼴로 쓴다).
def _pick(options: tuple[str, ...]) -> Callable[[random.Random], str]:
    return _Pick(options)


# 핵심 값 유형 하나: 라벨("주문번호" 등, 기록 JSON의 entities[].type), 값 종류(채점 정규화 방식), 값 생성기,
# 번호 유형만의 배포 형식(정규식).
# id_format은 서비스가 "확인이 필요한 번호"를 고를 때 정규화한 값이 이 형식에 맞는지 본다(verify.py, README
# 5단계). 실제 회사마다 번호 체계가 다르므로 배포 설정에 해당한다(주석).
@dataclass(frozen=True)
class EntityType:
    label: str
    kind: Kind
    make: Callable[[random.Random], str]
    # 번호 유형에만 쓴다: 배포 형식. 정규화한 값(values.normalize) 전체가 맞아야 하는 정규식이다. 형식 밖
    # 값은 확인이 필요하다(verify.py). 실제 번호 체계는 회사마다 달라 배포 설정에 해당한다.
    id_format: str | None = None


# 한 시나리오의 결과 하나: 처리 결과(해결/부분 해결/미해결/이관), 상담원이 한 처리 목록, 후속 조치 코드,
# 요약에 들어가야 할 사실(슬롯 이름을 채우는 형식 문자열), 이 결과에만 필요한 추가 슬롯, 교사에게 줄 끝맺음
# 지시(note), 뽑힐 가중치(weight). 구조 필드 정답 가운데 resolution·actions_taken·follow_up이 여기서 나온다.
@dataclass(frozen=True)
class Outcome:
    resolution: str
    actions: tuple[str, ...]
    follow_up: tuple[str, ...]
    facts: tuple[str, ...]  # 슬롯 이름을 채우는 형식 문자열들
    extra_slots: tuple[tuple[str, str], ...] = ()  # 이 결과에만 필요한 (슬롯, 값 유형 라벨)
    note: str = ""  # 대화를 쓰는 모델에게 주는, 통화를 어떻게 끝낼지에 대한 지시
    weight: float = 1.0


# 문의 유형 하나(= 기록의 category): 필수 슬롯(이름, 값 유형), 고객 요청 사실, 결과 후보들.
# needs_verification이 참이면 계정 관련 요청이라 상담원이 본인 확인을 할 수 있다(specs.py가 70% 확률로 본인
# 확인 사건과 '본인 확인' 처리를 넣는다).
@dataclass(frozen=True)
class Scenario:
    category: str
    slots: tuple[tuple[str, str], ...]  # (슬롯 이름, 값 유형 라벨). 모든 슬롯 값이 통화에 나온다
    request_fact: str  # 고객이 원한 것. 슬롯 이름을 채우는 형식 문자열
    outcomes: tuple[Outcome, ...]
    needs_verification: bool = True  # 계정 관련 요청: 상담원이 먼저 본인 확인을 할 수 있다


# 업종 하나. held_out이면 학습 분할에서 뺀다. extra_questions는 고객이 끝에 덧붙이는 작은 질문(대화 사건),
# slot_makers는 슬롯 이름별로 값 범위를 따로 정한 생성기다.
@dataclass(frozen=True)
class Domain:
    key: str
    label: str
    company: str
    entity_types: tuple[EntityType, ...]
    scenarios: tuple[Scenario, ...]
    held_out: bool = False
    extra_questions: tuple[str, ...] = field(default=())  # 고객이 덧붙일 수 있는 상관없는 작은 질문
    # slot_makers가 생긴 이유: 파일럿에서 택배비·데이터 충전·부가서비스 월 요금·적립금·카드 한도·연회비가 그
    # 유형의 넓은 금액 범위에서 뽑혀 "택배비 128만 원" 같은 어색한 값이 나왔다(docs/experiments.md 1단계
    # 파일럿, 커밋 56e691a). 그래서 그럴듯한 범위가 값 유형의 범위와 다른 슬롯 이름마다 값 생성기를 따로
    # 둔다(택배비는 상품 가격이 아니고, 데이터 충전은 월 청구액이 아니다).
    slot_makers: dict[str, Callable[[random.Random], str]] = field(default_factory=dict)

    # 라벨로 값 유형을 찾는다. 없는 라벨이면 KeyError(목록 밖 유형).
    def entity_type(self, label: str) -> EntityType:
        for et in self.entity_types:
            if et.label == label:
                return et
        raise KeyError(label)

    # 프롬프트에 넣는 라벨 목록들. categories: 시나리오 순서대로의 문의 유형. actions: 공통 처리(본인 확인)를
    # 먼저 두고 모든 결과의 처리를 처음 나온 순서대로 중복 없이 모은다(dict를 순서 있는 집합으로 쓴다).
    # follow_up_codes: 모든 결과의 후속 조치 코드. 모델이 고를 수 있는 라벨이 곧 이 목록이므로, 목록 밖 라벨을
    # 쓴 건은 off_list 지표로 센다.
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


# VERIFY: 모든 업종에 있는 공통 처리 라벨. 명세의 본인 확인 사건이 있으면 정답 처리 목록에 들어간다. TRANSFER:
# 다른 부서로 넘길 때의 처리 라벨(이관 결과에 쓰인다).
VERIFY = "본인 확인"
TRANSFER = "전문 부서 이관"
COMMON_ACTIONS = (VERIFY,)

# ---------------------------------------------------------------------------------------------------------
# 1. 온라인 쇼핑몰

# 쇼핑몰 상품 카탈로그 12개(지어낸 이름). 음성 조건에서 인식기가 잘 깨뜨리는 값이다(상품명 보존율 55.6%,
# docs/experiments.md 음성 조건 v3 단계 A).
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

# 1. 온라인 쇼핑몰 "도토리마켓". 값 유형: 주문번호, 상품명, 금액, 날짜, 시간. 학습·dev·test-a·test-b에 쓰인다.
# 결과의 weight=2는 그 결과가 두 배 자주 뽑힌다는 뜻이다(specs._pick_outcome이 가중치로 고른다).
SHOP = Domain(
    key="shop",
    label="온라인 쇼핑몰",
    company="도토리마켓",
    entity_types=(
        EntityType("주문번호", "id", _order_id, id_format=r"D\d{7}"),
        EntityType("상품명", "text", _pick(_SHOP_PRODUCTS)),
        EntityType("금액", "amount", _amount(8_000, 250_000, 100)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
    ),
    scenarios=(
        # 배송 문의: 해결(배송 조회, 도착 예정일 안내) / 부분 해결(택배사 확인 요청 + 정한 시각에 콜백).
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
        # 주문 취소: 해결(취소 + 환불 접수 → 환불 처리 대기) / 미해결(이미 출고되어 취소 불가, 반품 안내).
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
        # 반품 신청: 해결(반품 접수 + 수거 예약 → 수거 예정) / 미해결(반품 기한 지남).
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
        # 교환 신청: 해결(교환 접수 + 수거 예약 → 재배송 예정. 새 상품을 보내는 경우라 후속 조치는 재배송
        # 예정만 적는다, p2 기록 기준과 같다) / 부분 해결(재고 없음 → 재입고 알림).
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
        # 환불 문의: 해결(환불 상태 조회, 환불 예정일 안내) / 이관(환불 기록이 없어 결제 담당 부서로, 콜백
        # 예약).
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
        # 상품 불량: 해결(불량 접수 + 수거 → 재배송 예정) / 이관(다친 곳이 있어 보상 담당 부서로 → 보상 검토).
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
        # 결제 오류: 해결(중복 결제 한 건 취소 → 환불 처리 대기) / 부분 해결(결제사 확인 필요, 콜백 예약).
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
        # 적립금·쿠폰 문의: 해결 두 가지(유효기간 소멸 안내 + 쿠폰 발급 / 적립 예정일 안내). 적립금 금액은
        # slot_makers로 1천~5만 원 범위에서 뽑는다.
        Scenario(
            "적립금·쿠폰 문의",
            (("points", "금액"),),
            "고객이 적립금 {points}이 사라졌다고 문의함",
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
    # 업종의 덧붙임 질문과 슬롯별 값 범위(적립금 1천~5만 원).
    extra_questions=("앱에서 주문 내역 보는 법", "회원 등급 기준", "앱 알림 끄는 법"),
    slot_makers={"points": _amount(1_000, 50_000, 100)},
)

# ---------------------------------------------------------------------------------------------------------
# 2. 이동통신사

# 통신사 카탈로그: 요금제 6개, 기기 5개, 부가서비스 5개(모두 지어낸 이름). "은하 탭 9"는 음성 조건에서 학생이
# "은하택구"를 바로잡은 예로 README에 나온다.
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

# 2. 이동통신사 "별빛텔레콤". 값 유형: 요금제, 기기명, 부가서비스, 금액, 날짜, 시간, 접수번호.
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
        EntityType("접수번호", "id", _receipt, id_format=r"R\d{6}"),
    ),
    scenarios=(
        # 요금 문의: 해결(요금 내역 조회, 소액결제 설명) / 부분 해결(이의 신청 + 콜백).
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
        # 요금제 변경: 해결(바로 변경) / 부분 해결(이번 달 변경 제한, 적용일 예약). 두 요금제 슬롯은 같은
        # 유형이라 specs._draw_slot이 서로 다른 값으로 뽑는다.
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
        # 데이터 소진: 해결 두 가지(데이터 충전 금액 / 다른 요금제로 변경).
        Scenario(
            "데이터 소진",
            (("plan", "요금제"),),
            "고객이 {plan} 요금제의 데이터가 다 떨어져 속도가 느리다고 문의함",
            (
                Outcome(
                    "해결",
                    ("데이터 충전",),
                    (),
                    ("상담원이 {charge}짜리 데이터 충전을 해 줌",),
                    extra_slots=(("charge", "금액"),),
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
        # 해지 문의: 미해결(위약금 안내, 고객이 더 생각하기로 해 콜백 예약) / 이관(해지 전담 부서). "더 생각해
        # 보기로 하고 끝나면 미해결"은 p2 기록 기준에 적힌 관례다.
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
        # 통신 장애: 부분 해결(장애 접수 + 기사 방문 예약 → 기사 방문 예정, 방문일은 신고일 뒤로 맞춤) /
        # 해결(원격 점검으로 정상화).
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
        # 분실·정지: 해결(분실 정지) / 부분 해결(분실 정지 + 임대폰 신청 → 임대폰 배송).
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
        # 부가서비스 해지: 해결 두 가지(해지 + 요금 환급 접수 → 환불 처리 대기 / 가입 기록이 있어 해지만). 월
        # 요금은 slot_makers로 1천~1만 5천 원.
        Scenario(
            "부가서비스 해지",
            (("addon", "부가서비스"), ("addon_fee", "금액")),
            "고객이 가입한 적 없는 {addon}이 월 {addon_fee}씩 청구된다고 문의함",
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
        # 기기 할부 문의: 해결 두 가지(남은 할부금과 종료일 안내 / 완납 접수).
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
    # 슬롯별 값 범위: 데이터 충전 3천~3만 원, 부가서비스 월 요금 1천~1만 5천 원.
    extra_questions=("가까운 대리점 위치", "멤버십 포인트 사용처", "와이파이 비밀번호 바꾸는 법"),
    slot_makers={"charge": _amount(3_000, 30_000, 100), "addon_fee": _amount(1_000, 15_000, 100)},
)

# ---------------------------------------------------------------------------------------------------------
# 3. 택배사

# 택배 물품 카탈로그 8개.
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

# 3. 택배사 "새싹택배". 값 유형: 운송장번호, 물품명, 금액, 날짜, 시간. 음성 조건 v1의 측정 인공물(운송장번호를
#    범위로 읽음)과 v2의 수정(숫자 하나씩 읽기)이 이 업종의 운송장번호에 관한 것이다.
# 계정 수준 요청이 아닌 시나리오(배송 조회, 배송 지연, 반품 수거 요청, 배송 일정 변경, 택배 보내기 예약,
# 오배송)는 needs_verification=False로 두어 본인 확인 사건이 들어가지 않는다.
PARCEL = Domain(
    key="parcel",
    label="택배사",
    company="새싹택배",
    entity_types=(
        EntityType("운송장번호", "id", _tracking, id_format=r"\d{12}"),
        EntityType("물품명", "text", _pick(_PARCEL_ITEMS)),
        EntityType("금액", "amount", _amount(3_000, 1_500_000, 1000)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
    ),
    scenarios=(
        # 배송 조회: 해결 하나(배달 예정일 안내).
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
        # 배송 지연: 부분 해결(지점 확인 요청 + 콜백) / 해결(물량 때문에 늦어짐, 새 도착일 안내. 새 도착일은
        # 처음 날짜 뒤로 맞춘다, 커밋 56e691a).
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
                    ("물량이 몰려 배송이 늦어졌고 {new_date}에 배달된다고 안내함",),
                    extra_slots=(("new_date", "날짜"),),
                ),
            ),
            needs_verification=False,
        ),
        # 파손 신고: 부분 해결(파손 접수, 영수증·사진 요청 → 보상 검토 + 서류 제출 대기) / 이관(고가 물품이라
        # 보상 전담 부서로).
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
        # 분실 신고: 이관(사고 처리 부서, 보상 검토) / 해결(경비실 보관 안내).
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
        # 반품 수거 요청: 해결 하나(수거 예약 → 수거 예정).
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
        # 배송 일정 변경: 해결(요청한 날로 변경) / 미해결(이미 배송 출발).
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
        # 택배 보내기 예약: 해결 하나(방문 접수 예약과 요금 안내 → 수거 예정). 요금은 slot_makers로 3천~3만
        # 원.
        Scenario(
            "택배 보내기 예약",
            (("item", "물품명"),),
            "고객이 {item}을 보내려고 방문 접수를 원함",
            (
                Outcome(
                    "해결",
                    ("방문 접수 예약",),
                    ("수거 예정",),
                    ("상담원이 {date}에 방문 접수를 예약하고 요금이 {fee}이라고 안내함",),
                    extra_slots=(("date", "날짜"), ("fee", "금액")),
                ),
            ),
            needs_verification=False,
        ),
        # 오배송: 부분 해결 하나(오배송 접수 + 회수 요청 → 재배송 예정).
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
    # 택배비 범위(3천~3만 원, 500원 단위). 파일럿에서 "택배비 128만 원"이 나와 따로 정한 범위다.
    extra_questions=("편의점 택배 접수 가능 여부", "일요일 배송 여부", "포장 박스 구매 방법"),
    slot_makers={"fee": _amount(3_000, 30_000, 500)},
)

# ---------------------------------------------------------------------------------------------------------
# 4. 카드사(학습에서 뺌)

# 카드 가맹점 카탈로그 6개.
_MERCHANTS = ("솔바람 카페", "한결 마트", "푸른숲 주유소", "모래시계 서점", "달빛 호텔", "해오름 온라인몰")

# 4. 카드사 "구름카드" (학습에서 뺀 업종, test-c 전용). 값 유형: 카드 끝자리, 가맹점명, 금액, 날짜, 시간.
# 분실 신고·카드 정지·재발급 신청·한도 심사 접수처럼 이 업종에만 있는 처리 라벨이 많다. 새 업종에서 학습한
# 모델이 틀린 것도 대부분 이런 처리 라벨이었다(README 3단계). 학생은 프롬프트의 목록만 보고 고른다.
CARD = Domain(
    key="card",
    label="카드사",
    company="구름카드",
    entity_types=(
        EntityType("카드 끝자리", "id", _last4, id_format=r"\d{4}"),
        EntityType("가맹점명", "text", _pick(_MERCHANTS)),
        EntityType("금액", "amount", _amount(5_000, 3_000_000, 100)),
        EntityType("날짜", "date", _date),
        EntityType("시간", "time", _time),
    ),
    scenarios=(
        # 카드 분실: 해결 하나(분실 신고 + 정지 + 재발급 신청 → 재발급 카드 배송).
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
        # 결제 취소 확인: 해결(가맹점 취소 접수 확인) / 부분 해결(취소 기록 없음, 가맹점 확인 요청 + 콜백).
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
        # 부정 사용 의심: 이관 하나(카드 즉시 정지 + 조사 부서, 보상 검토).
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
        # 한도 상향: 부분 해결(심사 접수 → 서류 제출 대기) / 해결(바로 상향). 원하는 한도는 slot_makers로
        # 200만~2,000만 원.
        Scenario(
            "한도 상향",
            (("limit", "금액"),),
            "고객이 카드 한도를 {limit}으로 올리고 싶다고 요청함",
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
        # 청구 금액 문의: 해결 하나(청구 내역과 결제일 안내).
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
        # 결제일 변경: 해결 하나(다음 달부터 변경, 이번 결제일 안내). 날짜 슬롯은 결과 사실에만 쓰인다.
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
        # 연회비 문의: 해결(청구 이유 설명) / 부분 해결(면제 심사 + 콜백). 연회비는 slot_makers로 1만~30만 원.
        Scenario(
            "연회비 문의",
            (("annual_fee", "금액"),),
            "고객이 연회비 {annual_fee}이 청구된 이유를 문의함",
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
        # 할부 전환: 해결 하나.
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
    # held_out=True: specs.SPLIT_DOMAINS가 이 업종을 test-c에만 넣는다.
    held_out=True,
    extra_questions=("포인트 사용처", "해외 결제 수수료", "카드 앱 비밀번호 변경"),
    slot_makers={
        "annual_fee": _amount(10_000, 300_000, 1000),
        "limit": _amount(2_000_000, 20_000_000, 500_000),
    },
)

# 업종 키 → 정의. 프롬프트, 명세, 채점, 서비스가 모두 이 표로 업종을 찾는다.
DOMAINS: dict[str, Domain] = {d.key: d for d in (SHOP, TELECOM, PARCEL, CARD)}


# 업종의 닫힌 목록 값(상품명, 요금제, 기기명, 부가서비스, 물품명, 가맹점명)을 값 유형 순서대로 중복 없이
# 모은다. 음성 조건 v3의 업종 맥락 글(asr_arms.context_prompt)이 쓴다.
def catalog(domain: Domain) -> tuple[str, ...]:
    """업종의 닫힌 목록 값 유형이 가질 수 있는 모든 값. 값 유형 순서대로."""
    out: dict[str, None] = {}
    for et in domain.entity_types:
        if isinstance(et.make, _Pick):
            out.update(dict.fromkeys(et.make.options))
    return tuple(out)


# 학습 업종(쇼핑몰, 통신, 택배)과 학습에서 뺀 업종(카드). specs.SPLIT_DOMAINS가 분할마다 어느 쪽을 쓸지
# 정한다.
TRAIN_DOMAINS: tuple[str, ...] = tuple(k for k, d in DOMAINS.items() if not d.held_out)
HELD_OUT_DOMAINS: tuple[str, ...] = tuple(k for k, d in DOMAINS.items() if d.held_out)
