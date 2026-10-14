"""요약의 결정적 값 확인(docs/experiments.md "요약 판정 재보정").

명세의 필수 사실은 저마다 시나리오나 결과의 틀에 슬롯 이름으로 값을 채워 만든 문장이다. 사실 안의 숫자형
값(번호, 금액, 날짜, 시각: 종류 id, amount, date, time)을 채점 규칙(`values.occurs_in_transcript`, 쓴 글
규칙)으로 요약 글에서 찾는다. 이름(종류 text)은 요약이 줄이거나 바꿔 쓸 수 있으므로 판정 모델에 맡긴다.

  python -m call_summary.value_facts reports/<run>... [--judge-file judge-<name>.jsonl] [--json out.json]

--judge-file을 주면 그 실행의 판정 결과도 값 확인과 합친다: 숫자형 값이 요약에 빠진 사실은, 판정 모델이
이미 틀림이라고 하지 않았다면 누락으로 센다(`hybrid_verdicts`).
"""

# [파일 역할] 요약 판정 재보정(2026-10-08)의 값 확인 V. 명세의 요약 사실에 적힌 번호·금액·날짜·시각이 모델
# 요약에 들어 있는지를 판정 모델 없이 채점과 같은 정규화로 센다. 판정 모델 결과 파일이 있으면 V를 판정에 더한
# 형태(V+J)의 사실 재현율도 낸다.
# [왜 필요한가] 2단계 판정 모델(qwen2.5 14B)은 kappa가 약 0.32로 낮고 값이 빠진 사실도 포함으로
# 봤다(docs/experiments.md "요약 판정 재보정" 왜). 사실 목록은 명세에서 나오므로 그 안의 숫자 값은 코드로
# 결정적으로 확인할 수 있다. 이름류(text)는 요약이 줄여 쓰거나 바꿔 쓸 수 있어 판정 모델에 맡긴다.
# [흐름] fact_slots(사실마다 들어간 슬롯을 틀에서 거꾸로 읽음) → value_mentions(숫자형 값만) →
# mention_found·fact_value_ok(요약에서 찾기) → hybrid_verdicts(V+J) → run_value_table·_judge_numbers(실행
# 하나의 지표와 부트스트랩 구간) → main(표 출력).
# [연결] judge_calib.py가 fact_value_ok·hybrid_verdicts로 보정 표본의 V와 V+J를 만든다. 결과 수치는 README
# 5단계 요약 품질 줄과 docs/experiments.md "값 확인 결과"에 있다.

from __future__ import annotations

# string.Formatter: 사실 틀("고객이 운송장 {tracking}의 배송 위치를 문의함" 같은 format 문자열)에서 자리
# 이름을 읽는다. random: 부트스트랩을 seed 고정으로 돌린다.
import argparse
import json
import math
import random
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

# judge.VERDICTS: 판정 라벨(포함, 누락, 틀림)을 판정 모델 쪽과 같은 문자열로 쓴다. specs.Spec: 명세의
# 슬롯·사실·말실수 값. values: 채점과 같은 정규화와 찾기.
from .dataset import load_items, read_jsonl
from .domains import DOMAINS
from .judge import VERDICTS
from .specs import Spec
from .values import Kind, normalize, occurs_in_transcript

# 값 확인 대상 종류: 번호·금액·날짜·시각. 이름류(text)는 뺀다(docs/experiments.md 요약 판정 재보정 1).
NUMERIC_KINDS: tuple[Kind, ...] = ("id", "amount", "date", "time")
INCLUDED = VERDICTS[0]  # 포함
MISSING = VERDICTS[1]  # 누락
WRONG = VERDICTS[2]  # 틀림


# 사실 하나에 적힌 숫자형 정답 값 하나: 몇 번째 사실인지, 어느 슬롯·유형·종류인지, 명세의 말투 그대로의 값.
@dataclass(frozen=True)
class ValueMention:
    fact: int  # spec.facts 안의 순번
    slot: str
    type: str  # 핵심 값 유형 라벨
    kind: Kind
    value: str  # 명세의 말투 그대로의 정답 값


# 사실마다 어느 슬롯 값이 들어갔는지를 틀 순서대로 돌려준다. 명세의 시나리오·결과 번호로 원래 틀을 찾아 슬롯
# 값으로 다시 채워 보고, 명세의 사실과 글자까지 같을 때만 믿는다. 다르면(손으로 고친 명세나 다른 생성기의
# 명세) 틀린 대응을 쓰지 않도록 멈춘다(규칙 1 "다시 채운 문장이 같은지 확인하고 다르면 멈춘다").
# 같은 슬롯이 한 틀에 두 번 나와도 dict.fromkeys로 한 번만 센다.
def fact_slots(spec: Spec) -> list[tuple[str, ...]]:
    """사실마다 채워 넣은 슬롯 이름(틀에 나온 순서). 명세의 사실이 자기 슬롯으로 채운 틀과 다르면(손으로
    고친 명세나 다른 생성기의 명세) 예외를 낸다."""
    domain = DOMAINS[spec.domain]
    scenario = next(s for s in domain.scenarios if s.category == spec.category)
    outcome = scenario.outcomes[spec.outcome_index]
    templates = (scenario.request_fact, *outcome.facts)
    by_slot = {sv.slot: sv.value for sv in spec.slots}
    if tuple(t.format(**by_slot) for t in templates) != spec.facts:
        raise ValueError(f"{spec.spec_id}: the facts are not the templates filled with the slots")
    return [tuple(dict.fromkeys(name for _, name, _, _ in Formatter().parse(t) if name)) for t in templates]


# 사실에 적힌 값 중 숫자형 종류만 (사실, 슬롯)마다 하나씩 모은다.
def value_mentions(spec: Spec) -> list[ValueMention]:
    """사실에 적힌 숫자형 정답 값들. (사실, 슬롯)마다 하나씩."""
    domain = DOMAINS[spec.domain]
    by_slot = {sv.slot: sv for sv in spec.slots}
    out = []
    for i, names in enumerate(fact_slots(spec)):
        for name in names:
            sv = by_slot[name]
            kind = domain.entity_type(sv.type).kind
            if kind in NUMERIC_KINDS:
                out.append(ValueMention(i, name, sv.type, kind, sv.value))
    return out


# 값마다 요약에서 찾았는지. 쓴 글 규칙(spoken=False)이라 "3만 2천 원"과 "32,000원", 하이픈·공백만 다른 번호는
# 같은 값으로 찾는다. 번호는 부분 문자열로 찾으므로 앞뒤에 글자가 붙은 번호("OD1902657")도 찾은 것으로 치는
# 약점이 있다(docs/experiments.md 보정 세트 절).
def mention_found(spec: Spec, summary: str) -> list[tuple[ValueMention, bool]]:
    return [(m, occurs_in_transcript(m.kind, m.value, summary)) for m in value_mentions(spec)]


# 사실마다: 숫자형 값이 없으면 None(판정 모델에만 맡김), 있으면 모든 값을 찾았을 때만 True. 값 하나라도 못
# 찾으면 False로 남는다.
def fact_value_ok(spec: Spec, summary: str) -> list[bool | None]:
    """사실마다: 숫자형 값이 없으면 None, 숫자형 값이 모두 요약에 있으면 True."""
    out: list[bool | None] = [None] * len(spec.facts)
    for m, found in mention_found(spec, summary):
        out[m.fact] = found and out[m.fact] is not False
    return out


# V+J 형태: 판정 모델의 사실별 판정에 값 확인을 덮는다. 값이 빠진 사실(False)의 '포함'만 '누락'으로 바꾸고
# '틀림'은 그대로 둔다(규칙 4). 판정 프롬프트를 바꾸지 않고 검토 의견("값 사실은 결정적으로 확인")을 넣는
# 방법이다.
# 판정이 비어 있으면(판정 실패, 또는 V 혼자를 만들 때 빈 튜플을 넘김) 모두 '포함'에서 시작해 값 확인만
# 반영한다. judge_calib.py가 V 혼자의 라벨을 이렇게 만든다.
def hybrid_verdicts(verdicts: Sequence[str], value_ok: Sequence[bool | None]) -> tuple[str, ...]:
    """값 확인을 덮은 판정: 값이 빠지면 포함을 누락으로 바꾼다(틀림은 그대로 둔다).
    판정이 비어 있으면(판정 실패) 포함에서 시작하므로 값 확인 결과만 반영된다."""
    base = tuple(verdicts) if verdicts else (INCLUDED,) * len(value_ok)
    if len(base) != len(value_ok):
        raise ValueError("one verdict per fact")
    return tuple(
        MISSING if ok is False and v == INCLUDED else v for v, ok in zip(base, value_ok, strict=True)
    )


# 고객이 처음에 잘못 말한 번호(말실수 사건의 교란 값)가 요약에 들어갔는지. 정정을 놓치고 고치기 전 값을 쓴
# 요약을 센다(값 확인 결과 표의 "교란 번호가 요약에" 열).
def distractor_written(spec: Spec, summary: str) -> bool:
    """고객이 처음에 잘못 말한 값(EV_CORRECTION)이 요약에 들어 있는지."""
    d = spec.distractor
    if d is None:
        return False
    kind = DOMAINS[spec.domain].entity_type(d.type).kind
    return normalize(kind, d.value) is not None and occurs_in_transcript(kind, d.value, summary)


# 건 단위 백분위 부트스트랩 95% 구간(seed 0). 행은 건 하나이고 stat이 비율을 다시 계산한다. NaN이 나온 표본은
# 뺀다.
def _boot(
    rows: Sequence, stat: Callable[[Sequence], float], n_boot: int, seed: int = 0
) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    rng = random.Random(seed)
    k = len(rows)
    vals = []
    for _ in range(n_boot):
        v = stat([rows[rng.randrange(k)] for _ in range(k)])
        if v == v:
            vals.append(v)
    if not vals:
        return math.nan, math.nan
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


# 건들을 합친 비율(분자 합 / 분모 합). 분모가 0이면 NaN.
def _ratio(rows: Sequence, num: Callable, den: Callable) -> float:
    d = sum(den(r) for r in rows)
    return sum(num(r) for r in rows) / d if d else math.nan


# 기록된 평가 실행 하나에 값 확인을 대 보고 지표를 계산한다(새 추론 없음). 데이터는 manifest의 data 경로로
# 읽는다.
# 요약이 비었거나 답이 스키마를 통과하지 못한 건(pred 없음)은 빈 요약으로 세어 값을 모두 못 찾은 것으로
# 친다(규칙 1).
# 지표: value_recall(사실·값 쌍 단위 값 보존율), value_fact_recall(값이 있는 사실 중 모든 값을 찾은 비율),
# items_all_values(값이 있는 건 중 모든 값을 찾은 건 비율), by_kind([찾은 수, 전체]), 교란 번호가 요약에
# 들어간 건 수. judge_file을 주면 _judge_numbers의 판정 지표를 붙인다.
def run_value_table(run: str | Path, judge_file: str | None = None, n_boot: int = 2000) -> dict:
    """기록된 실행 하나의 값 확인 수치(judge_file을 주면 판정 그대로와 V+J의 사실 재현율도)."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(manifest["data"].replace("\\", "/"))}
    rows = []
    for r in read_jsonl(run / "items.jsonl"):
        it = items[r["item_id"]]
        summary = (r.get("pred") or {}).get("summary", "") or ""
        found = mention_found(it.spec, summary)
        ok = fact_value_ok(it.spec, summary)
        rows.append(
            {
                "item_id": it.item_id,
                "empty": not summary.strip(),
                "found": found,
                "value_ok": ok,
                "distractor": it.spec.distractor is not None,
                "distractor_written": distractor_written(it.spec, summary),
            }
        )

    # 부트스트랩에 넘길 건별 분자·분모: 찾은 값 수, 값 수, 모든 값을 찾은 사실 수, 값이 있는 사실 수.
    def n_found(r):
        return sum(f for _, f in r["found"])

    def n_ment(r):
        return len(r["found"])

    def vf_ok(r):
        return sum(1 for x in r["value_ok"] if x is True)

    def vf_all(r):
        return sum(1 for x in r["value_ok"] if x is not None)

    by_kind: dict[str, list[int]] = {}
    for r in rows:
        for m, f in r["found"]:
            by_kind.setdefault(m.kind, [0, 0])
            by_kind[m.kind][0] += int(f)
            by_kind[m.kind][1] += 1
    with_values = [r for r in rows if r["found"]]
    table: dict = {
        "run": str(run),
        "n_items": len(rows),
        "n_empty": sum(r["empty"] for r in rows),
        "n_mentions": sum(n_ment(r) for r in rows),
        "value_recall": _ratio(rows, n_found, n_ment),
        "value_recall_ci": _boot(rows, lambda xs: _ratio(xs, n_found, n_ment), n_boot),
        "value_fact_recall": _ratio(rows, vf_ok, vf_all),
        "value_fact_recall_ci": _boot(rows, lambda xs: _ratio(xs, vf_ok, vf_all), n_boot),
        "items_all_values": (
            sum(1 for r in with_values if all(f for _, f in r["found"])) / len(with_values)
            if with_values
            else math.nan
        ),
        "by_kind": {k: by_kind[k] for k in NUMERIC_KINDS if k in by_kind},
        "distractor_items": sum(r["distractor"] for r in rows),
        "distractor_in_summary": sum(r["distractor_written"] for r in rows),
    }
    if judge_file:
        table["judge"] = _judge_numbers(run / judge_file, rows, n_boot)
    return table


# 실행 폴더 안의 판정 파일(judge.jsonl 또는 judge-<이름>.jsonl)을 읽어 판정 그대로(raw)와 V+J(hybrid)의 사실
# 재현율, wrong_rate를 낸다. 판정이 빠진 건이 있으면 다른 파일을 잘못 준 것이므로 멈춘다.
# 판정 실패(ok False) 건은 재현율 계산에서 빼고 그 수를 n_judge_failed로 남긴다. wrong_rate는 어긋난 문장이
# 하나 이상이거나 '틀림' 사실이 있는 건의 비율이다(docs/experiments.md "채택한 판정으로 서빙 학생 판정"의
# 2026-10-09 바로잡음: judge.wrong_rate와 같은 정의).
def _judge_numbers(path: Path, rows: list[dict], n_boot: int) -> dict:
    judged = {d["item_id"]: d for d in read_jsonl(path)}
    missing = [r["item_id"] for r in rows if r["item_id"] not in judged]
    if missing:
        raise ValueError(f"{path} has no judgement for {len(missing)} items, e.g. {missing[:3]}")
    usable = []
    for r in rows:
        d = judged[r["item_id"]]
        if not d.get("ok", True):
            continue
        raw = tuple(d["verdicts"])
        usable.append(
            {
                "raw": raw,
                "hybrid": hybrid_verdicts(raw, r["value_ok"]),
                "wrong": int(d.get("wrong_statements", 0)) > 0 or WRONG in raw,
            }
        )

    def inc(key):
        return lambda x: x[key].count(INCLUDED)

    def total(x):
        return len(x["raw"])

    return {
        "file": path.name,
        "n_judge_failed": len(rows) - len(usable),
        "raw_fact_recall": _ratio(usable, inc("raw"), total),
        "raw_fact_recall_ci": _boot(usable, lambda xs: _ratio(xs, inc("raw"), total), n_boot),
        "hybrid_fact_recall": _ratio(usable, inc("hybrid"), total),
        "hybrid_fact_recall_ci": _boot(usable, lambda xs: _ratio(xs, inc("hybrid"), total), n_boot),
        "wrong_rate": sum(x["wrong"] for x in usable) / len(usable) if usable else math.nan,
    }


# 백분율 문자열(NaN이면 "-").
def _pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.1f}"


# 명령줄 입구: 실행 폴더 여럿의 값 확인 표를 한 줄씩 찍고, --judge-file을 주면 판정 지표 줄을 덧붙인다.
# --json이면 수치 전체를 파일로 남긴다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="기록된 실행 폴더들(reports/<run_id>)")
    ap.add_argument("--judge-file", help="각 실행 폴더 안의 판정 결과 파일 이름. 예: judge-<name>.jsonl")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--json", help="모든 수치를 이 파일에 JSON으로 쓴다")
    args = ap.parse_args(argv)
    # Windows 콘솔 기본 인코딩(cp949)으로 내보내다 깨지지 않게 표준 출력을 UTF-8로 바꾼다.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    tables = [run_value_table(r, args.judge_file, args.n_boot) for r in args.runs]
    print("run | n | value_recall [95%] | value_fact_recall | items_all_values | by kind | distractor")
    for t in tables:
        lo, hi = t["value_recall_ci"]
        kinds = ", ".join(f"{k} {f}/{n}" for k, (f, n) in t["by_kind"].items())
        print(
            f"{Path(t['run']).name} | {t['n_items']} | {_pct(t['value_recall'])} [{_pct(lo)}, {_pct(hi)}] | "
            f"{_pct(t['value_fact_recall'])} | {_pct(t['items_all_values'])} | {kinds} | "
            f"{t['distractor_in_summary']}/{t['distractor_items']}"
        )
        if "judge" in t:
            j = t["judge"]
            print(
                f"  judge {j['file']}: raw fact recall {_pct(j['raw_fact_recall'])}, "
                f"with value check {_pct(j['hybrid_fact_recall'])}, wrong_rate {_pct(j['wrong_rate'])}, "
                f"failed {j['n_judge_failed']}"
            )
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
