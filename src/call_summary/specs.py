"""시나리오 명세: 통화 한 건의 정답. 대화가 생기기 전에 seed를 고정한 난수로 뽑는다.

명세는 문의 유형, 핵심 값(과 그 값을 말하는 말투), 상담원이 한 처리, 통화가 끝난 방식, 대화 사건, 요약에
꼭 들어가야 할 사실을 정한다. 정답 상담 후처리 기록은 명세만으로 정해지는 순수 함수이므로, LLM이 구조화
라벨을 만드는 일은 없다.
"""

# [파일 역할] 통화 한 건의 정답(시나리오 명세)을 seed 고정 난수로 뽑는다. 명세에서 정답 기록(gold)이 함수로
# 바로 나오고, 교사 모델은 이 명세를 보고 대화만 쓴다.
# [왜 이 방식인가] 라이선스가 분명한 공개 한국어 상담 요약 데이터를 찾지 못해 "정답을 먼저 만들고 대화를
# 나중에 만든다"로 정했다(docs/plan.md "데이터"). LLM이 구조화 정답을 만들지 않으므로 정답에 모델 오류가
# 섞이지 않는다(docs/experiments.md "공통": 정답은 명세 생성기에서만 나온다).
# [결정성] 모듈 수준 random이나 시각을 쓰지 않고 (분할, 업종, 번호, 유형)으로 seed를 정한 random.Random만
# 쓴다. 같은 명령으로 언제든 같은 명세가 다시 나온다.
# [흐름] make_spec: 시나리오 고르기 → 결과(outcome) 고르기 → 슬롯 값 뽑기 → 날짜 순서 맞추기 → 본인 확인·대화
# 사건 뽑기 → 말실수 값(distractor) → 요약 사실 문장. make_specs는 유형을 고르게 돌려 가며 여러 건을 만든다.
# [연결] build_specs.py가 make_specs로 data/specs/*.jsonl을 만들고, gen.py가 명세로 대화를 쓰게 하고 검사한다.
# dataset.Item.gold()가 Spec.gold로 정답 기록을 만든다.

from __future__ import annotations

# random: seed를 받은 random.Random 인스턴스만 쓴다. dataclass: 명세를 바뀌지 않는 값 객체로 두고 asdict로
# JSONL에 쓴다.
import random
from dataclasses import asdict, dataclass

# domains: 업종별 시나리오 틀, 값 생성기, 라벨 목록. schema: 정답 기록 형식. values.normalize: 날짜 순서를
# 비교할 때 "10월 5일" 같은 말투를 "10-05"로 읽는다.
from .domains import DOMAINS, HELD_OUT_DOMAINS, TRAIN_DOMAINS, VERIFY, Domain, Outcome, Scenario
from .schema import AfterCallRecord, Entity, FollowUp
from .values import normalize

# 분할 이름. 분할 이름이 seed 문자열에 들어가므로 같은 번호라도 분할이 다르면 다른 명세가 나온다(train과
# test의 명세 seed가 겹치지 않게, docs/plan.md "데이터 나누기").
SPLITS = ("train", "dev", "test-a", "test-b", "test-c")
# 분할마다 만들 업종. test-b는 학습 업종에 대해 자기 분할(test-b)의 명세를 쓰지만 대화는 다른 계열의 모델이
# 쓴다. test-c는 학습에서 빼 둔 업종이다.
SPLIT_DOMAINS: dict[str, tuple[str, ...]] = {
    "train": TRAIN_DOMAINS,
    "dev": TRAIN_DOMAINS,
    "test-a": TRAIN_DOMAINS,
    "test-b": TRAIN_DOMAINS,
    "test-c": HELD_OUT_DOMAINS,
}

# 대화 사건: 대화를 다양하게 만들고, 모델이 헷갈릴 만한 상황(말실수 뒤 정정, 대기, 잡담 등)을 넣는다. 대부분은
# 교사에게 주는 지시일 뿐이고 정답을 바꾸는 것은 verify(처리 목록에 '본인 확인'이 더해짐)와 correction(틀린
# 번호가 대화에 섞임)이다. 사건 하나하나는 대화를 쓰는 모델에게 주는 지시다.
EV_VERIFY = "verify"  # 상담원이 본인 확인(이름 + 생년월일)을 하고, 처리에 '본인 확인'이 더해진다
EV_CORRECTION = "correction"  # 고객이 틀린 값을 먼저 말했다가 고친다(대화에 말실수 값이 남는다)
EV_COMPLAINT = "complaint"  # 고객이 화가 나 있고, 상담원이 사과한다
EV_HOLD = "hold"  # 상담원이 확인하는 동안 기다려 달라고 한다
EV_REPEAT = "repeat"  # 고객이 같은 것을 두 번 묻는다
EV_SMALLTALK = "smalltalk"  # 고객이 용건 전에 잠깐 잡담을 한다
EV_EXTRA_QUESTION = "extra_question"  # 고객이 끝에 상관없는 작은 질문을 덧붙인다

# 사건마다 독립적으로 일어날 확률. 한 통화에 여러 사건이 함께 들어갈 수 있다.
EVENT_PROBS: dict[str, float] = {
    EV_CORRECTION: 0.25,
    EV_COMPLAINT: 0.25,
    EV_HOLD: 0.3,
    EV_REPEAT: 0.15,
    EV_SMALLTALK: 0.15,
    EV_EXTRA_QUESTION: 0.2,
}
# 계정 정보가 필요한 시나리오(needs_verification)에서 본인 확인을 하는 비율. 하지 않는 통화도 생기므로 정답
# 처리 목록의 '본인 확인'은 통화마다 다르다. gen.py가 대화의 본인 확인 여부가 명세와 같은지 검사한다.
VERIFY_PROB = 0.7


# 슬롯 값 하나: 시나리오 틀의 자리 이름(slot), 핵심 값 유형(type), 대화에 글자 그대로 나와야 하는 말투(value).
# gen.py의 자동 검사가 이 value가 대화에 그대로 있는지 확인한다.
@dataclass(frozen=True)
class SlotValue:
    slot: str
    type: str
    value: str  # 말투(겉 형태). 대화에 글자 그대로 나와야 한다


# 통화 한 건의 명세(정답). 구조 필드의 정답(유형, 결과, 핵심 값, 처리, 후속 조치)과 교사에게 줄 지시(사건,
# 말실수 값, 덧붙일 질문, 결과 메모), 요약에 꼭 들어가야 할 사실(facts)을 모두 담는다. facts는 요약
# 판정(judge.py)과 값 확인(value_facts.py)의 기준이 된다.
@dataclass(frozen=True)
class Spec:
    spec_id: str
    split: str
    domain: str
    category: str
    outcome_index: int
    resolution: str
    slots: tuple[SlotValue, ...]
    actions: tuple[str, ...]
    follow_up: tuple[str, ...]
    events: tuple[str, ...]
    distractor: SlotValue | None  # 고객이 먼저 말하는 틀린 값(EV_CORRECTION)
    extra_question: str | None
    facts: tuple[str, ...]
    note: str

    # 명세에서 정답 기록을 만든다. 구조 필드는 명세에서 그대로 옮기고, summary만 밖에서
    # 받는다(dataset.Item.gold가 교사가 쓴 참고 요약을 넘긴다. 채점은 요약을 보지 않는다).
    # 같은 (유형, 값)은 한 번만 넣고, 후속 조치 코드가 하나라도 있으면 required=True다.
    def gold(self, summary: str = "") -> AfterCallRecord:
        entities: list[Entity] = []
        seen: set[tuple[str, str]] = set()
        for sv in self.slots:
            key = (sv.type, sv.value)
            if key not in seen:
                seen.add(key)
                entities.append(Entity(type=sv.type, value=sv.value))
        return AfterCallRecord(
            category=self.category,
            resolution=self.resolution,  # type: ignore[arg-type]
            entities=entities,
            actions_taken=list(self.actions),
            follow_up=FollowUp(required=bool(self.follow_up), codes=list(self.follow_up)),
            summary=summary,
        )

    # JSONL에 쓰는 dict. 튜플은 asdict를 거치며 그대로 남고 json이 목록으로 쓴다.
    def to_dict(self) -> dict:
        return asdict(self)

    # 파일에서 읽은 dict를 Spec으로 되살린다. 목록을 튜플로, 슬롯 dict를 SlotValue로 바꾼다.
    # extra_question·note는 없을 수 있어 .get으로 읽는다. 호출 위치: dataset.Item.from_dict, gen.py,
    # finalize.py.
    @classmethod
    def from_dict(cls, d: dict) -> Spec:
        slots = tuple(SlotValue(**s) for s in d["slots"])
        distractor = SlotValue(**d["distractor"]) if d.get("distractor") else None
        return cls(
            spec_id=d["spec_id"],
            split=d["split"],
            domain=d["domain"],
            category=d["category"],
            outcome_index=d["outcome_index"],
            resolution=d["resolution"],
            slots=slots,
            actions=tuple(d["actions"]),
            follow_up=tuple(d["follow_up"]),
            events=tuple(d["events"]),
            distractor=distractor,
            extra_question=d.get("extra_question"),
            facts=tuple(d["facts"]),
            note=d.get("note", ""),
        )


# 슬롯 하나의 값을 뽑는다. 슬롯 이름에 따로 정한 생성기(slot_makers: 택배비처럼 그 유형의 넓은 범위가 어색한
# 슬롯)가 있으면 그것을, 없으면 핵심 값 유형의 생성기를 쓴다. 이 범위는 파일럿에서 "택배비 128만 원" 같은 값이
# 나와 더했다(docs/experiments.md 1단계 파일럿).
# 같은 유형 안에서 값이 겹치지 않게 최대 50번 다시 뽑고, 그래도 안 되면 예외를 낸다.
def _draw_slot(
    rng: random.Random, domain: Domain, slot: str, label: str, taken: dict[str, set[str]]
) -> SlotValue:
    make = domain.slot_makers.get(slot, domain.entity_type(label).make)
    used = taken.setdefault(label, set())
    for _ in range(50):
        value = make(rng)
        if value not in used:
            used.add(value)
            return SlotValue(slot=slot, type=label, value=value)
    raise RuntimeError(f"could not draw a distinct {label}")


# 파일럿에서 배송 지연의 새 도착일과 기사 방문일이 처음 날짜보다 앞서는 명세가 나와 더한
# 규칙이다(docs/experiments.md 1단계 파일럿, 커밋 56e691a).
# 날짜가 이 순서여야 하는 슬롯 쌍: 배송 지연의 새 날짜와 방문일은 처음 날짜보다 뒤에 온다.
DATE_ORDER = (("date", "new_date"), ("date", "visit_date"))


# DATE_ORDER의 두 날짜가 거꾸로면 값만 맞바꾼다. 새로 뽑지 않고 맞바꾸므로 난수 소비가 늘지 않는다.
# normalize("date")가 "MM-DD" 두 자리 문자열이라 문자열 비교가 날짜 비교가 된다(생성기는 9~11월만 뽑아 해를
# 넘기지 않는다).
def _order_dates(slots: list[SlotValue]) -> list[SlotValue]:
    by = {sv.slot: i for i, sv in enumerate(slots)}
    for first, later in DATE_ORDER:
        if first in by and later in by:
            a, b = slots[by[first]], slots[by[later]]
            if (normalize("date", a.value) or "") > (normalize("date", b.value) or ""):
                slots[by[first]] = SlotValue(slot=a.slot, type=a.type, value=b.value)
                slots[by[later]] = SlotValue(slot=b.slot, type=b.type, value=a.value)
    return slots


# 말실수 값을 만든다: 번호의 숫자 하나를 다른 숫자로 바꾼다. 고객이 처음에 이 값을 말했다가 고치는
# 장면(EV_CORRECTION)에 쓰고, 정답에는 넣지 않는다. 모델이 고치기 전 값을 기록하면 오답이 되므로 정정 이해를
# 시험한다.
# 번호(id)만 다루고 그 밖의 종류는 예외를 낸다. 호출하는 곳(make_spec)은 번호로만 부른다.
def _mutate(rng: random.Random, sv: SlotValue, kind: str) -> SlotValue:
    """그럴듯한 틀린 값: 번호의 숫자 하나를 바꾼다. 다른 종류는 ValueError를 낸다(번호만 쓴다)."""
    if kind == "id":
        digits = [i for i, ch in enumerate(sv.value) if ch.isdigit()]
        i = rng.choice(digits)
        new_digit = rng.choice([d for d in "0123456789" if d != sv.value[i]])
        return SlotValue(slot=sv.slot, type=sv.type, value=sv.value[:i] + new_digit + sv.value[i + 1 :])
    raise ValueError(kind)


# 시나리오의 결과 후보 중 하나를 weight 비율로 고른다. 번호(outcome_index)도 돌려 명세에 남긴다.
def _pick_outcome(rng: random.Random, scenario: Scenario) -> tuple[int, Outcome]:
    weights = [o.weight for o in scenario.outcomes]
    idx = rng.choices(range(len(scenario.outcomes)), weights=weights, k=1)[0]
    return idx, scenario.outcomes[idx]


# 명세 한 건을 만든다. 입력: 분할, 업종 키, 업종 안의 번호, (선택) 유형. 같은 입력이면 늘 같은 명세가 나온다.
# 주의: 아래 단계의 순서가 난수 소비 순서다. 단계를 바꾸거나 값 범위를 바꾸면 같은 seed에서도 다른 명세가
# 나온다(test-b의 한 건이 이 때문에 바뀌어 다시 썼다, docs/experiments.md 1단계 진행 기록).
# 호출 위치: make_specs, 테스트(tests/test_specs.py, tests/test_scoring.py 등).
def make_spec(split: str, domain_key: str, index: int, category: str | None = None) -> Spec:
    """(분할, 업종, 번호, 유형)이 같으면 늘 같은 명세가 나온다(결정적)."""
    if split not in SPLITS:
        raise ValueError(split)
    domain = DOMAINS[domain_key]
    # 문자열 seed: 분할·업종·번호·유형이 하나라도 다르면 다른 난수열이 된다. 문자열 seed는 실행마다 바뀌는
    # 해시 무작위화의 영향을 받지 않는다.
    rng = random.Random(f"call-summary:{split}:{domain_key}:{index}:{category or ''}")
    # 유형을 지정하면 그 유형의 시나리오를, 아니면 난수로 하나 고른다.
    scenario = (
        next(s for s in domain.scenarios if s.category == category)
        if category is not None
        else rng.choice(domain.scenarios)
    )
    outcome_index, outcome = _pick_outcome(rng, scenario)

    # 시나리오의 슬롯과 고른 결과에만 필요한 슬롯(extra_slots, 예: 새 도착일)을 차례로 뽑는다.
    taken: dict[str, set[str]] = {}
    slots = [
        _draw_slot(rng, domain, slot, label, taken) for slot, label in scenario.slots + outcome.extra_slots
    ]
    slots = _order_dates(slots)
    by_slot = {sv.slot: sv.value for sv in slots}

    events: list[str] = []
    actions: list[str] = []
    # 본인 확인은 사건이자 처리다: 일어나면 처리 목록 맨 앞에 '본인 확인'이 들어간다. 그 뒤 결과의 처리들,
    # 그다음 나머지 사건을 확률대로 뽑는다.
    if scenario.needs_verification and rng.random() < VERIFY_PROB:
        events.append(EV_VERIFY)
        actions.append(VERIFY)
    actions.extend(outcome.actions)
    for ev, p in EVENT_PROBS.items():
        if rng.random() < p:
            events.append(ev)

    # 말실수 사건은 바꿀 번호가 있어야 하므로, 번호 슬롯이 없는 시나리오에서는 사건을 없던 것으로 한다.
    distractor = None
    if EV_CORRECTION in events:
        id_slots = [sv for sv in slots if domain.entity_type(sv.type).kind == "id"]
        if id_slots:
            distractor = _mutate(rng, rng.choice(id_slots), "id")
        else:
            events.remove(EV_CORRECTION)
    extra_question = rng.choice(domain.extra_questions) if EV_EXTRA_QUESTION in events else None

    # 요약에 꼭 들어가야 할 사실: 고객의 요청 한 문장 + 결과의 사실 문장들. 틀의 {슬롯} 자리에 뽑은 값을
    # 넣는다. value_facts.fact_slots가 이 틀을 거꾸로 읽어 어느 값이 들어갔는지 찾는다.
    facts = (scenario.request_fact.format(**by_slot),) + tuple(f.format(**by_slot) for f in outcome.facts)
    return Spec(
        # 건 id: 예) dev-shop-00012. 평가 기록과 짝지은 비교에서 건을 맞추는 열쇠가 된다.
        spec_id=f"{split}-{domain_key}-{index:05d}",
        split=split,
        domain=domain_key,
        category=scenario.category,
        outcome_index=outcome_index,
        resolution=outcome.resolution,
        slots=tuple(slots),
        actions=tuple(actions),
        follow_up=outcome.follow_up,
        events=tuple(events),
        distractor=distractor,
        extra_question=extra_question,
        facts=facts,
        note=outcome.note,
    )


# 업종마다 n_per_domain건을 만든다. i번째 건에 유형 i mod (유형 수)를 지정해 모든 유형이 거의 같은 수로
# 나온다. 호출 위치: build_specs.py.
def make_specs(split: str, domain_keys: tuple[str, ...], n_per_domain: int) -> list[Spec]:
    """유형을 고르게 나눈다: 업종의 i번째 건은 유형 i mod (유형 수)를 받는다."""
    out: list[Spec] = []
    for key in domain_keys:
        cats = DOMAINS[key].categories
        for i in range(n_per_domain):
            out.append(make_spec(split, key, i, category=cats[i % len(cats)]))
    return out
