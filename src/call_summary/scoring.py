"""Scoring of structured fields. Code only: no judge model is involved here (summaries are scored elsewhere).

A reply that does not validate as an AfterCallRecord scores zero on every field; `schema_ok`-only
aggregates are reported separately so format failures and content errors can be told apart.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from .domains import DOMAINS
from .schema import AfterCallRecord, ParseResult
from .values import Kind, normalize, occurs_in_transcript


@dataclass(frozen=True)
class ItemScore:
    item_id: str
    domain: str
    gold_category: str
    json_ok: bool
    schema_ok: bool
    pred_category: str | None
    category_ok: bool
    resolution_ok: bool
    entity_tp: int
    entity_fp: int
    entity_fn: int
    entity_pred: int
    entity_hallucinated: int  # predicted values not found anywhere in the transcript
    action_tp: int
    action_fp: int
    action_fn: int
    follow_required_ok: bool
    follow_codes_ok: bool
    off_list: (
        int  # labels outside the domain's allowed lists (category, actions, follow-up codes, entity types)
    )
    exact: bool  # every structured field right (summary not included)

    def to_dict(self) -> dict:
        return asdict(self)


def _entity_keys(domain_key: str, record: AfterCallRecord) -> tuple[Counter, int]:
    """Normalized (type, value) multiset; returns also how many entities were unreadable/unknown type."""
    d = DOMAINS[domain_key]
    kinds: dict[str, Kind] = {et.label: et.kind for et in d.entity_types}
    keys: Counter = Counter()
    bad = 0
    for e in record.entities:
        kind = kinds.get(e.type)
        norm = normalize(kind, e.value) if kind else None
        if norm is None:
            bad += 1
        else:
            keys[(e.type, norm)] += 1
    return keys, bad


def score_item(
    item_id: str,
    domain_key: str,
    gold: AfterCallRecord,
    parsed: ParseResult,
    transcript: str,
    spoken: bool = False,
) -> ItemScore:
    """spoken=True for recognised-speech transcripts (stage 4): see values.occurs_in_transcript."""
    d = DOMAINS[domain_key]
    gold_keys, _ = _entity_keys(domain_key, gold)
    gold_actions = set(gold.actions_taken)
    n_gold_entities = sum(gold_keys.values())
    pred = parsed.record
    if not parsed.schema_ok or pred is None:
        return ItemScore(
            item_id=item_id,
            domain=domain_key,
            gold_category=gold.category,
            json_ok=parsed.json_ok,
            schema_ok=False,
            pred_category=None,
            category_ok=False,
            resolution_ok=False,
            entity_tp=0,
            entity_fp=0,
            entity_fn=n_gold_entities,
            entity_pred=0,
            entity_hallucinated=0,
            action_tp=0,
            action_fp=0,
            action_fn=len(gold_actions),
            follow_required_ok=False,
            follow_codes_ok=False,
            off_list=0,
            exact=False,
        )

    pred_keys, unreadable = _entity_keys(domain_key, pred)
    tp = sum((pred_keys & gold_keys).values())
    fp = sum((pred_keys - gold_keys).values()) + unreadable
    fn = sum((gold_keys - pred_keys).values())

    kinds: dict[str, Kind] = {et.label: et.kind for et in d.entity_types}
    hallucinated = sum(
        0 if occurs_in_transcript(kinds.get(e.type, "text"), e.value, transcript, spoken) else 1
        for e in pred.entities
    )

    pred_actions = set(pred.actions_taken)
    a_tp = len(pred_actions & gold_actions)
    a_fp = len(pred_actions - gold_actions)
    a_fn = len(gold_actions - pred_actions)

    follow_required_ok = pred.follow_up.required == gold.follow_up.required
    follow_codes_ok = set(pred.follow_up.codes) == set(gold.follow_up.codes)

    off_list = (
        (pred.category not in d.categories)
        + sum(a not in d.actions for a in pred_actions)
        + sum(c not in d.follow_up_codes for c in set(pred.follow_up.codes))
        + sum(e.type not in kinds for e in pred.entities)
    )
    category_ok = pred.category == gold.category
    resolution_ok = pred.resolution == gold.resolution
    exact = (
        category_ok
        and resolution_ok
        and fp == 0
        and fn == 0
        and a_fp == 0
        and a_fn == 0
        and follow_required_ok
        and follow_codes_ok
    )
    return ItemScore(
        item_id=item_id,
        domain=domain_key,
        gold_category=gold.category,
        json_ok=True,
        schema_ok=True,
        pred_category=pred.category,
        category_ok=category_ok,
        resolution_ok=resolution_ok,
        entity_tp=tp,
        entity_fp=fp,
        entity_fn=fn,
        entity_pred=len(pred.entities),
        entity_hallucinated=hallucinated,
        action_tp=a_tp,
        action_fp=a_fp,
        action_fn=a_fn,
        follow_required_ok=follow_required_ok,
        follow_codes_ok=follow_codes_ok,
        off_list=off_list,
        exact=exact,
    )


# ---------------------------------------------------------------------------------------------------------
# Aggregates. Each takes a list of ItemScore and returns a float (nan when undefined).

Metric = Callable[[Sequence[ItemScore]], float]
NAN = float("nan")


def _rate(items: Sequence[ItemScore], pred: Callable[[ItemScore], bool]) -> float:
    return sum(pred(s) for s in items) / len(items) if items else NAN


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else NAN


def json_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.json_ok)


def schema_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.schema_ok)


def category_acc(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.category_ok)


def category_macro_f1(items: Sequence[ItemScore]) -> float:
    """Macro-F1 over (domain, gold category) labels that occur in the gold set."""
    labels = {(s.domain, s.gold_category) for s in items}
    if not labels:
        return NAN
    f1s = []
    for dom, cat in labels:
        tp = sum(1 for s in items if s.domain == dom and s.gold_category == cat and s.category_ok)
        fn = sum(1 for s in items if s.domain == dom and s.gold_category == cat and not s.category_ok)
        fp = sum(1 for s in items if s.domain == dom and s.pred_category == cat and s.gold_category != cat)
        f1s.append(_f1(tp, fp, fn))
    return sum(f1s) / len(f1s)


def resolution_acc(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.resolution_ok)


def entity_precision(items: Sequence[ItemScore]) -> float:
    tp = sum(s.entity_tp for s in items)
    fp = sum(s.entity_fp for s in items)
    return tp / (tp + fp) if tp + fp else NAN


def entity_recall(items: Sequence[ItemScore]) -> float:
    tp = sum(s.entity_tp for s in items)
    fn = sum(s.entity_fn for s in items)
    return tp / (tp + fn) if tp + fn else NAN


def entity_f1(items: Sequence[ItemScore]) -> float:
    return _f1(
        sum(s.entity_tp for s in items), sum(s.entity_fp for s in items), sum(s.entity_fn for s in items)
    )


def hallucination_rate(items: Sequence[ItemScore]) -> float:
    """Share of predicted entity values that appear nowhere in the transcript."""
    n = sum(s.entity_pred for s in items)
    return sum(s.entity_hallucinated for s in items) / n if n else NAN


def action_f1(items: Sequence[ItemScore]) -> float:
    return _f1(
        sum(s.action_tp for s in items), sum(s.action_fp for s in items), sum(s.action_fn for s in items)
    )


def follow_up_acc(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.follow_required_ok and s.follow_codes_ok)


def exact_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.exact)


def off_list_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.off_list > 0)


METRICS: dict[str, Metric] = {
    "json": json_rate,
    "schema": schema_rate,
    "exact": exact_rate,
    "category_acc": category_acc,
    "category_macro_f1": category_macro_f1,
    "resolution_acc": resolution_acc,
    "entity_p": entity_precision,
    "entity_r": entity_recall,
    "entity_f1": entity_f1,
    "hallucination": hallucination_rate,
    "action_f1": action_f1,
    "follow_up_acc": follow_up_acc,
    "off_list": off_list_rate,
}


def summarize(items: Sequence[ItemScore]) -> dict[str, float]:
    return {name: fn(items) for name, fn in METRICS.items()}


def bootstrap_ci(
    items: Sequence[ItemScore], metric: Metric, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(point, low, high) with a percentile bootstrap over items."""
    point = metric(items)
    if not items:
        return point, NAN, NAN
    rng = random.Random(seed)
    k = len(items)
    stats = []
    for _ in range(n_boot):
        v = metric([items[rng.randrange(k)] for _ in range(k)])
        if v == v:  # skip nan
            stats.append(v)
    if not stats:
        return point, NAN, NAN
    stats.sort()
    lo = stats[int(alpha / 2 * (len(stats) - 1))]
    hi = stats[int((1 - alpha / 2) * (len(stats) - 1))]
    return point, lo, hi


def paired_bootstrap_diff(
    a: Sequence[ItemScore],
    b: Sequence[ItemScore],
    metric: Metric,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """metric(b) - metric(a) with a CI from resampling the same item ids for both sides."""
    by_a = {s.item_id: s for s in a}
    by_b = {s.item_id: s for s in b}
    ids = sorted(by_a.keys() & by_b.keys())
    if len(ids) != len(by_a) or len(ids) != len(by_b):
        raise ValueError("paired bootstrap needs the same item ids on both sides")
    xa = [by_a[i] for i in ids]
    xb = [by_b[i] for i in ids]
    point = metric(xb) - metric(xa)
    rng = random.Random(seed)
    k = len(ids)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        v = metric([xb[i] for i in idx]) - metric([xa[i] for i in idx])
        if v == v:
            stats.append(v)
    if not stats:
        return point, NAN, NAN
    stats.sort()
    return point, stats[int(alpha / 2 * (len(stats) - 1))], stats[int((1 - alpha / 2) * (len(stats) - 1))]


def value_survival(
    domain_key: str, gold: AfterCallRecord, transcript: str, spoken: bool = True
) -> dict[str, list[bool]]:
    """Per entity type, whether each gold value can still be found in the transcript."""
    kinds: dict[str, Kind] = {et.label: et.kind for et in DOMAINS[domain_key].entity_types}
    out: dict[str, list[bool]] = {}
    for e in gold.entities:
        out.setdefault(e.type, []).append(occurs_in_transcript(kinds[e.type], e.value, transcript, spoken))
    return out


def unfound_split(
    domain_key: str, gold: AfterCallRecord, pred: AfterCallRecord, transcript: str, spoken: bool = True
) -> tuple[Counter, Counter]:
    """Predicted values not found in the transcript (what `entity_hallucinated` counts), per type: restored
    (equal to a gold value after normalization, e.g. a misheard product name put right) or invented."""
    kinds: dict[str, Kind] = {et.label: et.kind for et in DOMAINS[domain_key].entity_types}
    gold_keys, _ = _entity_keys(domain_key, gold)
    restored: Counter = Counter()
    invented: Counter = Counter()
    for e in pred.entities:
        if occurs_in_transcript(kinds.get(e.type, "text"), e.value, transcript, spoken):
            continue
        norm = normalize(kinds[e.type], e.value) if e.type in kinds else None
        (restored if norm is not None and (e.type, norm) in gold_keys else invented)[e.type] += 1
    return restored, invented
