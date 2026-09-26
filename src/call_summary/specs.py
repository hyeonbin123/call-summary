"""Scenario specs: the ground truth of one call, drawn with a seeded RNG before any dialogue exists.

A spec fixes the category, the entity values (and how they are said), what the agent did, how the call
ended, the conversational events, and the facts a summary must contain. The gold after-call record is a
pure function of the spec, so no LLM ever produces a structured label.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .domains import DOMAINS, HELD_OUT_DOMAINS, TRAIN_DOMAINS, VERIFY, Domain, Outcome, Scenario
from .schema import AfterCallRecord, Entity, FollowUp
from .values import normalize

SPLITS = ("train", "dev", "test-a", "test-b", "test-c")
# Which domains each generated split covers. test-b reuses test-b specs over the training domains but its
# dialogues are written by a different model family; test-c is the held-out domain.
SPLIT_DOMAINS: dict[str, tuple[str, ...]] = {
    "train": TRAIN_DOMAINS,
    "dev": TRAIN_DOMAINS,
    "test-a": TRAIN_DOMAINS,
    "test-b": TRAIN_DOMAINS,
    "test-c": HELD_OUT_DOMAINS,
}

# Conversational events. Each is an instruction to the dialogue writer; some also change the gold.
EV_VERIFY = "verify"  # agent verifies identity (name + birth date), adds the 본인 확인 action
EV_CORRECTION = "correction"  # caller first says a wrong value, then corrects it (distractor in the text)
EV_COMPLAINT = "complaint"  # caller is upset, agent apologizes
EV_HOLD = "hold"  # agent asks the caller to wait while checking
EV_REPEAT = "repeat"  # caller asks the same thing twice
EV_SMALLTALK = "smalltalk"  # caller chats briefly before the request
EV_EXTRA_QUESTION = "extra_question"  # caller adds a small unrelated question at the end

EVENT_PROBS: dict[str, float] = {
    EV_CORRECTION: 0.25,
    EV_COMPLAINT: 0.25,
    EV_HOLD: 0.3,
    EV_REPEAT: 0.15,
    EV_SMALLTALK: 0.15,
    EV_EXTRA_QUESTION: 0.2,
}
VERIFY_PROB = 0.7  # for scenarios that need account access


@dataclass(frozen=True)
class SlotValue:
    slot: str
    type: str
    value: str  # surface form, exactly as it must appear in the dialogue


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
    distractor: SlotValue | None  # wrong value the caller says first (EV_CORRECTION)
    extra_question: str | None
    facts: tuple[str, ...]
    note: str

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

    def to_dict(self) -> dict:
        return asdict(self)

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


# Slot pairs whose dates must be in this order: a delay's new date and a visit come after the first date.
DATE_ORDER = (("date", "new_date"), ("date", "visit_date"))


def _order_dates(slots: list[SlotValue]) -> list[SlotValue]:
    by = {sv.slot: i for i, sv in enumerate(slots)}
    for first, later in DATE_ORDER:
        if first in by and later in by:
            a, b = slots[by[first]], slots[by[later]]
            if (normalize("date", a.value) or "") > (normalize("date", b.value) or ""):
                slots[by[first]] = SlotValue(slot=a.slot, type=a.type, value=b.value)
                slots[by[later]] = SlotValue(slot=b.slot, type=b.type, value=a.value)
    return slots


def _mutate(rng: random.Random, sv: SlotValue, kind: str) -> SlotValue:
    """A plausible wrong value: one digit changed (ids) or a different round amount."""
    if kind == "id":
        digits = [i for i, ch in enumerate(sv.value) if ch.isdigit()]
        i = rng.choice(digits)
        new_digit = rng.choice([d for d in "0123456789" if d != sv.value[i]])
        return SlotValue(slot=sv.slot, type=sv.type, value=sv.value[:i] + new_digit + sv.value[i + 1 :])
    raise ValueError(kind)


def _pick_outcome(rng: random.Random, scenario: Scenario) -> tuple[int, Outcome]:
    weights = [o.weight for o in scenario.outcomes]
    idx = rng.choices(range(len(scenario.outcomes)), weights=weights, k=1)[0]
    return idx, scenario.outcomes[idx]


def make_spec(split: str, domain_key: str, index: int, category: str | None = None) -> Spec:
    """Deterministic in (split, domain, index, category)."""
    if split not in SPLITS:
        raise ValueError(split)
    domain = DOMAINS[domain_key]
    rng = random.Random(f"call-summary:{split}:{domain_key}:{index}:{category or ''}")
    scenario = (
        next(s for s in domain.scenarios if s.category == category)
        if category is not None
        else rng.choice(domain.scenarios)
    )
    outcome_index, outcome = _pick_outcome(rng, scenario)

    taken: dict[str, set[str]] = {}
    slots = [
        _draw_slot(rng, domain, slot, label, taken) for slot, label in scenario.slots + outcome.extra_slots
    ]
    slots = _order_dates(slots)
    by_slot = {sv.slot: sv.value for sv in slots}

    events: list[str] = []
    actions: list[str] = []
    if scenario.needs_verification and rng.random() < VERIFY_PROB:
        events.append(EV_VERIFY)
        actions.append(VERIFY)
    actions.extend(outcome.actions)
    for ev, p in EVENT_PROBS.items():
        if rng.random() < p:
            events.append(ev)

    distractor = None
    if EV_CORRECTION in events:
        id_slots = [sv for sv in slots if domain.entity_type(sv.type).kind == "id"]
        if id_slots:
            distractor = _mutate(rng, rng.choice(id_slots), "id")
        else:
            events.remove(EV_CORRECTION)
    extra_question = rng.choice(domain.extra_questions) if EV_EXTRA_QUESTION in events else None

    facts = (scenario.request_fact.format(**by_slot),) + tuple(f.format(**by_slot) for f in outcome.facts)
    return Spec(
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


def make_specs(split: str, domain_keys: tuple[str, ...], n_per_domain: int) -> list[Spec]:
    """Balanced over categories: index i of a domain gets category i mod (number of categories)."""
    out: list[Spec] = []
    for key in domain_keys:
        cats = DOMAINS[key].categories
        for i in range(n_per_domain):
            out.append(make_spec(split, key, i, category=cats[i % len(cats)]))
    return out
