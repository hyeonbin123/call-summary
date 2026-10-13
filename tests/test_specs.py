# [테스트 묶음] 업종 카탈로그(domains.py)와 명세 생성기(specs.py)를 검증한다: 학습·제외 업종 나누기, 카탈로그
# 정합성(라벨·슬롯·사실 틀), 결정성, 직렬화 왕복, 모든 결과에 닿는지와 정답 기록의 유효성, 유형 균형, 분할
# 사이 id 분리, 사건 비율, 날짜 순서와 금액 범위.
# [왜 중요한가] 구조 필드의 정답은 명세에서만 나온다(docs/experiments.md "공통"). 카탈로그에 빈틈이
# 있으면(목록에 없는 라벨, 채울 수 없는 사실 틀, 읽을 수 없는 값) 정답 자체가 틀리고, 생성기가 결정적이지
# 않으면 데이터를 다시 만들 수 없다.

import string
from collections import Counter

import pytest

from call_summary.domains import DOMAINS, HELD_OUT_DOMAINS, TRAIN_DOMAINS, VERIFY
from call_summary.schema import RESOLUTIONS, AfterCallRecord
from call_summary.specs import EV_CORRECTION, EV_EXTRA_QUESTION, EV_VERIFY, Spec, make_spec, make_specs
from call_summary.values import normalize


# 학습 업종은 쇼핑몰·통신·택배 셋이고, 카드 하나만 학습에서 뺀다(test-c: 학습에 없던 업종으로 일반화를 보는
# 분할).
def test_domain_split():
    assert set(TRAIN_DOMAINS) == {"shop", "telecom", "parcel"}
    assert HELD_OUT_DOMAINS == ("card",)


# 업종마다 카탈로그 정합성: 핵심 값 유형 이름이 겹치지 않고, 문의 유형이 7개 이상이며 겹치지 않고, 처리 목록에
# '본인 확인'이 있고, 덧붙일 질문이 있다. 시나리오의 모든 슬롯이 업종의 핵심 값 유형을 쓰고, 결과마다 처리
# 결과가 허용 값이며, 슬롯 이름이 겹치지 않고, 사실 틀이 쓰는 {슬롯}이 모두 그 시나리오·결과에 있다(없으면
# 사실 문장을 만들 때 KeyError).
@pytest.mark.parametrize("key", list(DOMAINS))
def test_domain_catalog_is_consistent(key):
    d = DOMAINS[key]
    labels = [et.label for et in d.entity_types]
    assert len(labels) == len(set(labels))
    assert len(d.categories) == len(set(d.categories)) >= 7
    assert VERIFY in d.actions
    assert d.extra_questions
    for sc in d.scenarios:
        for slot, label in sc.slots:
            assert label in labels, (sc.category, slot)
        assert sc.outcomes
        for o in sc.outcomes:
            assert o.resolution in RESOLUTIONS
            slot_names = {s for s, _ in sc.slots + o.extra_slots}
            assert len(slot_names) == len(sc.slots) + len(o.extra_slots), "duplicate slot name"
            for tmpl in (sc.request_fact, *o.facts):
                used = {f for _, f, _, _ in string.Formatter().parse(tmpl) if f}
                assert used <= slot_names, (sc.category, tmpl)
            for _, label in o.extra_slots:
                assert label in labels


# 같은 (분할, 업종, 번호)면 같은 명세, 분할이 다르면 다른 명세가 나온다(seed 문자열에 분할이 들어간다).
def test_make_spec_is_deterministic():
    a = make_spec("train", "shop", 7)
    b = make_spec("train", "shop", 7)
    assert a == b
    assert make_spec("dev", "shop", 7) != a


# to_dict → from_dict로 되살린 명세가 원래와 같다. 데이터 파일에 명세를 JSON으로 저장하고 다시 읽기 때문에
# 필요하다.
def test_spec_roundtrip():
    for i in range(30):
        s = make_spec("dev", "telecom", i)
        assert Spec.from_dict(s.to_dict()) == s


# 업종마다 400건을 만들어: 모든 시나리오의 모든 결과가 한 번 이상 나오고(죽은 결과 없음), 정답 기록이 스키마를
# 통과하며 라벨이 모두 업종 목록 안이고, 모든 핵심 값을 그 종류로 정규화할 수 있는지 본다. 같은 유형의
# 슬롯끼리 값이 겹치지 않고, '본인 확인' 처리와 본인 확인 사건이 늘 함께이며, 말실수 값은 정답 값과 다르고,
# 덧붙일 질문은 그 사건이 있을 때만, 사실 문장에 채우지 못한 {슬롯}이 남지 않는다.
@pytest.mark.parametrize("key", list(DOMAINS))
def test_every_outcome_is_reachable_and_valid(key):
    d = DOMAINS[key]
    specs = make_specs("train", (key,), 400)
    reached = {(s.category, s.outcome_index) for s in specs}
    for sc in d.scenarios:
        for i in range(len(sc.outcomes)):
            assert (sc.category, i) in reached
    for s in specs:
        gold = s.gold("요약")
        AfterCallRecord.model_validate(gold.model_dump())
        assert gold.category in d.categories
        assert set(gold.actions_taken) <= set(d.actions)
        assert set(gold.follow_up.codes) <= set(d.follow_up_codes)
        for e in gold.entities:
            kind = d.entity_type(e.type).kind
            assert normalize(kind, e.value) is not None, (e.type, e.value)
        # same-type slots never share a value
        per_type = Counter((sv.type, sv.value) for sv in s.slots)
        assert max(per_type.values()) == 1
        assert (VERIFY in s.actions) == (EV_VERIFY in s.events)
        if s.distractor is not None:
            assert EV_CORRECTION in s.events
            assert s.distractor.value not in {sv.value for sv in s.slots}
        assert (s.extra_question is not None) == (EV_EXTRA_QUESTION in s.events)
        assert all("{" not in f for f in s.facts)


# 택배 80건이면 모든 유형이 나오고 유형별 건수 차이가 1 이하이며 건 id가 모두 다르다(i mod 유형 수로 돌려
# 지정).
def test_make_specs_balances_categories():
    specs = make_specs("dev", ("parcel",), 80)
    counts = Counter(s.category for s in specs)
    assert len(counts) == len(DOMAINS["parcel"].categories)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({s.spec_id for s in specs}) == 80


# train과 dev의 명세 id가 겹치지 않는다(id에 분할 이름이 들어간다). 분할 사이 핵심 값 겹침은 finalize가 따로
# 거른다.
def test_splits_do_not_share_ids():
    train = {s.spec_id for s in make_specs("train", TRAIN_DOMAINS, 50)}
    dev = {s.spec_id for s in make_specs("dev", TRAIN_DOMAINS, 50)}
    assert not train & dev


# 말실수 사건 비율이 설정(0.25)보다 조금 낮은 0.1~0.3 사이인지 본다. 번호 슬롯이 없는 시나리오에서는 이 사건을
# 없애기 때문이다.
def test_event_rates_roughly_match():
    specs = make_specs("train", TRAIN_DOMAINS, 400)
    rate = sum(EV_CORRECTION in s.events for s in specs) / len(specs)
    # only specs with an id slot keep the correction event
    assert 0.1 < rate < 0.3


# 파일럿에서 찾은 어색한 값이 다시 나오지 않는지 확인한다(docs/experiments.md 1단계 파일럿, 커밋 56e691a): 새
# 도착일·방문일은 처음 날짜보다 뒤이고, 택배 요금·데이터 충전·부가서비스 요금 같은 슬롯은 정해 둔 상한
# 안이다("택배비 128만 원" 같은 값 방지).
def test_date_pairs_are_ordered_and_fees_are_plausible():
    from call_summary.values import normalize as norm

    for key in ("telecom", "parcel"):
        for s in make_specs("train", (key,), 300):
            by = {sv.slot: sv.value for sv in s.slots}
            for first, later in (("date", "new_date"), ("date", "visit_date")):
                if first in by and later in by:
                    assert norm("date", by[first]) < norm("date", by[later])
            for slot, hi in (("fee", 30_000), ("charge", 30_000), ("addon_fee", 15_000)):
                if slot in by:
                    assert int(norm("amount", by[slot])) <= hi
