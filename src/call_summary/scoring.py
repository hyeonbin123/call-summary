"""Scoring of structured fields. Code only: no judge model is involved here (summaries are scored elsewhere).

A reply that does not validate as an AfterCallRecord scores zero on every field; `schema_ok`-only
aggregates are reported separately so format failures and content errors can be told apart.
"""

# [파일 역할] 모델이 낸 기록을 정답 기록(명세에서 나온 것)과 필드별로 비교해 건별 점수(ItemScore)를 만들고,
# 여러 건을 모아 지표(exact, 핵심 값 F1, 환각 등)와 부트스트랩 구간을 낸다.
# [왜 필요한가] 이 프로젝트의 결론은 모두 구조 필드 지표, 특히 exact로 낸다(docs/experiments.md "공통"). 판정
# 모델 없이 코드로만 채점해야 결과가 결정적이고 다시 계산할 수 있다. 요약 문장 채점은
# judge.py·value_facts.py가 따로 맡는다.
# [흐름] score_item(건 하나 채점) → 지표 함수들(METRICS) → summarize(지표 한 벌) → bootstrap_ci(한 실행의
# 구간)·paired_bootstrap_diff(두 실행의 짝지은 차이) → value_survival·unfound_split(4단계 음성 조건 분석).
# [연결] evaluate.py·service_eval.py·bench.py가 건마다 score_item을 부르고, compare.py가 METRICS와
# paired_bootstrap_diff로 실행끼리 비교한다. asr_report.py·asr_arms.py가 value_survival·unfound_split을 쓴다.

from __future__ import annotations

# random: 부트스트랩 재표본을 seed를 받은 random.Random으로 뽑아 같은 입력이면 같은 구간이 나오게 한다.
# Counter: 핵심 값을 (유형, 정규화 값)의 다중 집합으로 세어 같은 값이 두 번 나오는 경우까지 맞춘다.
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

# 업종별 허용 라벨 목록(유형·처리·후속 조치 코드·핵심 값 유형과 그 종류), 기록 형식, 값 정규화 규칙을
# 가져온다.
from .domains import DOMAINS
from .schema import AfterCallRecord, ParseResult
from .values import Kind, normalize, occurs_in_transcript


# 건 하나의 채점 결과. 지표를 바로 담지 않고 tp/fp/fn 같은 개수를 담아 두어, 여러 건을 합쳐 F1을
# 계산(micro)하거나 부트스트랩으로 다시 뽑아 계산할 수 있게 한다. 평가 기록(items.jsonl)의 "score"로 저장되고
# compare.py가 ItemScore(**d)로 다시 읽는다. frozen=True로 만든 뒤 바뀌지 않게 한다.
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


# 기록의 핵심 값을 (유형, 정규화 값) 다중 집합으로 바꾼다. 유형이 업종 목록에 없거나 값을 그 종류로 읽을 수
# 없으면(정규화가 None) 집합에 넣지 않고 bad로 센다. 예측 쪽의 bad는 score_item이 오답(fp)으로 더한다.
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


# 답 하나를 정답과 비교해 ItemScore를 만든다. 입력: 건 id, 업종, 정답 기록, parse_reply 결과, 전사(환각
# 검사용), spoken(음성 인식 전사면 True: 번호를 숫자 덩어리로도 찾음).
# 호출 위치: evaluate.run_items, service_eval.py, bench.py. evaluate·service_eval은 분할 이름이 "-asr"로
# 끝나면 spoken=True로 부른다.
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
    # 정답 쪽 값은 명세에서 나와 늘 읽을 수 있으므로 bad 수는 버린다.
    gold_keys, _ = _entity_keys(domain_key, gold)
    gold_actions = set(gold.actions_taken)
    n_gold_entities = sum(gold_keys.values())
    pred = parsed.record
    # 스키마를 통과하지 못한 답은 모든 필드를 오답으로 친다(docs/experiments.md "공통" schema 항목). 정답
    # 값·처리는 모두 fn으로 넣어 F1에 반영하고, 예측한 값이 없으므로 환각 분모(entity_pred)에는 아무것도
    # 더하지 않는다.
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

    # 핵심 값: 다중 집합의 교집합이 tp, 예측에만 있는 것과 읽을 수 없는 값이 fp, 정답에만 있는 것이 fn이다.
    # 정규화한 값으로 비교하므로 "32,000원"과 "3만 2천 원"은 같은 값이다(tests/test_scoring.py).
    pred_keys, unreadable = _entity_keys(domain_key, pred)
    tp = sum((pred_keys & gold_keys).values())
    fp = sum((pred_keys - gold_keys).values()) + unreadable
    fn = sum((gold_keys - pred_keys).values())

    # 환각: 예측한 값 하나하나가 전사 어디에도 없으면 1로 센다(판정 모델을 쓰지 않는 코드 검사, docs/plan.md
    # "평가 방식"). 정답과 같은지는 보지 않으므로, 인식 오류를 바로잡아 정답과 같아진 값도 전사에 없으면
    # 들어간다. 그래서 4단계부터 unfound_split으로 복원과 지어낸 값을 따로 센다.
    # 목록에 없는 유형은 text로 보고 부분 문자열로 찾는다. 빈 값("")은 정규화가 None이라 늘 환각으로
    # 센다(docs/experiments.md 3단계: 빈 값 채우기도 환각에 포함).
    kinds: dict[str, Kind] = {et.label: et.kind for et in d.entity_types}
    hallucinated = sum(
        0 if occurs_in_transcript(kinds.get(e.type, "text"), e.value, transcript, spoken) else 1
        for e in pred.entities
    )

    # 처리 목록과 후속 조치 코드는 순서와 중복을 무시한 집합으로 비교한다.
    pred_actions = set(pred.actions_taken)
    a_tp = len(pred_actions & gold_actions)
    a_fp = len(pred_actions - gold_actions)
    a_fn = len(gold_actions - pred_actions)

    follow_required_ok = pred.follow_up.required == gold.follow_up.required
    follow_codes_ok = set(pred.follow_up.codes) == set(gold.follow_up.codes)

    # 업종 허용 목록 밖 라벨 수: 유형 1개, 처리·후속 조치 코드(중복 제거)·핵심 값 유형마다 1개씩. 지표
    # off_list는 이것이 1 이상인 건의 비율이다(tests/test_scoring.py가 3개로 세는지 확인).
    off_list = (
        (pred.category not in d.categories)
        + sum(a not in d.actions for a in pred_actions)
        + sum(c not in d.follow_up_codes for c in set(pred.follow_up.codes))
        + sum(e.type not in kinds for e in pred.entities)
    )
    category_ok = pred.category == gold.category
    resolution_ok = pred.resolution == gold.resolution
    # exact(주 지표): 요약을 뺀 구조 필드가 모두 맞아야 참이다. 핵심 값은 빠진 것도(fn) 더한 것도(fp) 없어야
    # 한다.
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

# 지표 함수의 꼴: 건 점수 목록 → 실수. 분모가 0이면 0이 아니라 NaN을 돌려 "정의되지 않음"을 구분한다(예: 값을
# 하나도 예측하지 않으면 환각률은 NaN).
Metric = Callable[[Sequence[ItemScore]], float]
NAN = float("nan")


# 조건을 만족하는 건의 비율(건이 없으면 NaN).
def _rate(items: Sequence[ItemScore], pred: Callable[[ItemScore], bool]) -> float:
    return sum(pred(s) for s in items) / len(items) if items else NAN


# F1 = 2tp / (2tp + fp + fn). 정밀도·재현율을 따로 구하지 않고 한 번에 계산한다.
def _f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else NAN


def json_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.json_ok)


def schema_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.schema_ok)


def category_acc(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.category_ok)


# 문의 유형 macro-F1. 업종마다 유형 이름이 겹칠 수 있어 (업종, 유형)을 한 라벨로 보고, 정답에 나온 라벨마다
# F1을 구해 평균한다. 드문 유형도 같은 무게로 보려는 지표다(계획의 "정확도, macro-F1").
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


# 핵심 값 정밀도·재현율·F1은 건별 평균이 아니라 모든 건의 tp/fp/fn을 합쳐 계산한다(docs/experiments.md "공통":
# 여러 건을 합쳐 계산).
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


# 환각률 = 전사에 없는 예측 값 수 / 예측한 값 수. 분모가 예측 값이라 값을 적게 낼수록 분모가 작아진다.
def hallucination_rate(items: Sequence[ItemScore]) -> float:
    """Share of predicted entity values that appear nowhere in the transcript."""
    n = sum(s.entity_pred for s in items)
    return sum(s.entity_hallucinated for s in items) / n if n else NAN


def action_f1(items: Sequence[ItemScore]) -> float:
    return _f1(
        sum(s.action_tp for s in items), sum(s.action_fp for s in items), sum(s.action_fn for s in items)
    )


# 후속 조치는 필요 여부와 코드 집합이 둘 다 맞아야 맞은 것으로 본다.
def follow_up_acc(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.follow_required_ok and s.follow_codes_ok)


def exact_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.exact)


def off_list_rate(items: Sequence[ItemScore]) -> float:
    return _rate(items, lambda s: s.off_list > 0)


# 보고하는 지표 이름 → 함수. 이 이름이 summary.json과 compare.py 표의 열 이름이 된다(docs/experiments.md
# "공통"의 지표 목록과 같다).
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


# 한 실행의 지표에 대한 백분위 부트스트랩 95% 구간. 건을 복원 추출로 n_boot번(기본 2,000번, 문서의 규칙) 다시
# 뽑아 지표를 계산하고, 정렬한 값의 2.5%·97.5% 위치를 고른다.
# seed를 고정해 같은 실행이면 같은 구간이 나온다. 다시 뽑은 표본에서 지표가 NaN이면(예: 예측 값이 하나도 없음)
# 그 표본은 뺀다.
# 호출 위치: evaluate.summary_table.
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


# 두 실행(a, b)의 지표 차이 metric(b) − metric(a)와 그 구간. 같은 건 id를 양쪽에서 함께 다시 뽑는 짝지은
# 부트스트랩이라, 같은 건에서의 차이만 보고 건마다 다른 어려움은 양쪽에 같이 들어간다(docs/experiments.md
# "공통": 두 모델 비교는 짝지은 부트스트랩).
# 판정 규칙("하한 > 0이면 넘음" 등)이 이 구간으로 정해진다(docs/experiments.md 3단계 판정 문장). 양쪽 건 id가
# 다르면 짝을 지을 수 없으므로 예외를 낸다.
# 호출 위치: compare.py(--paired), asr_arms.py(값 보존율 비교).
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


# 정답 핵심 값 중 전사에서 찾을 수 있는 것을 유형별 참/거짓 목록으로 돌려준다(4단계 "값 보존율"). 음성 인식
# 전사에 값이 온전히 남았는지 보는 것으로, 모델이 전사를 그대로 옮겨 맞힐 수 있는 상한에
# 가깝다(docs/experiments.md 4단계 채점). 기본 spoken=True.
# 호출 위치: asr_report.py survival, asr_arms.py.
def value_survival(
    domain_key: str, gold: AfterCallRecord, transcript: str, spoken: bool = True
) -> dict[str, list[bool]]:
    """Per entity type, whether each gold value can still be found in the transcript."""
    kinds: dict[str, Kind] = {et.label: et.kind for et in DOMAINS[domain_key].entity_types}
    out: dict[str, list[bool]] = {}
    for e in gold.entities:
        out.setdefault(e.type, []).append(occurs_in_transcript(kinds[e.type], e.value, transcript, spoken))
    return out


# 환각으로 센 예측 값(전사에 없는 값)을 정답과 같은 값(복원: 학습에서 본 이름으로 인식 오류를 바로잡은 경우)과
# 정답에도 없는 값(지어낸 값)으로 나눠 유형별로 센다. 4단계 규칙에 모델을 재기 전에 더한
# 구분이다(docs/experiments.md 4단계 "복원과 환각을 나눠 센다").
# 호출 위치: asr_report.py unfound.
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
        # 읽을 수 없는 값이나 목록 밖 유형은 정답과 같을 수 없으므로 지어낸 값으로 센다.
        norm = normalize(kinds[e.type], e.value) if e.type in kinds else None
        (restored if norm is not None and (e.type, norm) in gold_keys else invented)[e.type] += 1
    return restored, invented
