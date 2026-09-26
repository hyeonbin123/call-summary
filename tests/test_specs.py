import string
from collections import Counter

import pytest

from call_summary.domains import DOMAINS, HELD_OUT_DOMAINS, TRAIN_DOMAINS, VERIFY
from call_summary.schema import RESOLUTIONS, AfterCallRecord
from call_summary.specs import EV_CORRECTION, EV_EXTRA_QUESTION, EV_VERIFY, Spec, make_spec, make_specs
from call_summary.values import normalize


def test_domain_split():
    assert set(TRAIN_DOMAINS) == {"shop", "telecom", "parcel"}
    assert HELD_OUT_DOMAINS == ("card",)


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


def test_make_spec_is_deterministic():
    a = make_spec("train", "shop", 7)
    b = make_spec("train", "shop", 7)
    assert a == b
    assert make_spec("dev", "shop", 7) != a


def test_spec_roundtrip():
    for i in range(30):
        s = make_spec("dev", "telecom", i)
        assert Spec.from_dict(s.to_dict()) == s


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


def test_make_specs_balances_categories():
    specs = make_specs("dev", ("parcel",), 80)
    counts = Counter(s.category for s in specs)
    assert len(counts) == len(DOMAINS["parcel"].categories)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({s.spec_id for s in specs}) == 80


def test_splits_do_not_share_ids():
    train = {s.spec_id for s in make_specs("train", TRAIN_DOMAINS, 50)}
    dev = {s.spec_id for s in make_specs("dev", TRAIN_DOMAINS, 50)}
    assert not train & dev


def test_event_rates_roughly_match():
    specs = make_specs("train", TRAIN_DOMAINS, 400)
    rate = sum(EV_CORRECTION in s.events for s in specs) / len(specs)
    # only specs with an id slot keep the correction event
    assert 0.1 < rate < 0.3


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
