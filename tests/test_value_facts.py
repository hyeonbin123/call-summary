import json
import math
from pathlib import Path

import pytest

from call_summary import value_facts
from call_summary.dataset import Item, load_items, write_jsonl
from call_summary.specs import SPLIT_DOMAINS, Spec, make_spec
from call_summary.value_facts import (
    fact_slots,
    fact_value_ok,
    hybrid_verdicts,
    run_value_table,
    value_mentions,
)
from call_summary.values import normalize


def _spec(domain: str, category: str, want_resolution: str | None = None, distractor: bool = False) -> Spec:
    for i in range(400):
        spec = make_spec("dev", domain, i, category=category)
        if want_resolution is not None and spec.resolution != want_resolution:
            continue
        if distractor and spec.distractor is None:
            continue
        return spec
    raise AssertionError("no such spec")


def _slots(spec: Spec) -> dict[str, str]:
    return {sv.slot: sv.value for sv in spec.slots}


def _item(spec: Spec) -> Item:
    turns = [{"speaker": "고객", "text": " ".join(sv.value for sv in spec.slots)}]
    return Item(item_id=spec.spec_id, split="dev", domain=spec.domain, spec=spec, turns=turns)


def _run(root: Path, preds: list[tuple[Spec, str | None]], judge: list[dict] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data.jsonl"
    write_jsonl(data, [_item(spec).to_dict() for spec, _ in preds])
    run = root / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    write_jsonl(
        run / "items.jsonl",
        [{"item_id": spec.spec_id, "pred": None if s is None else {"summary": s}} for spec, s in preds],
    )
    if judge is not None:
        write_jsonl(run / "judge-x.jsonl", judge)
    return run


def test_fact_slots_rebuild_every_generated_fact():
    for split in ("dev", "test-c"):
        for domain in SPLIT_DOMAINS[split]:
            for i in range(30):
                spec = make_spec(split, domain, i)
                slots = fact_slots(spec)
                assert len(slots) == len(spec.facts)
                assert all(set(s) <= set(_slots(spec)) for s in slots)


def test_fact_slots_match_the_committed_datasets():
    root = Path(__file__).resolve().parents[1] / "datasets"
    for name in ("dev.jsonl", "dev-asr-v2.jsonl", "test-a.jsonl", "test-c.jsonl", "test-d-v2.jsonl"):
        path = root / name
        if not path.exists():
            pytest.skip(f"{name} not in this checkout")
        for it in load_items(path):
            assert len(fact_slots(it.spec)) == len(it.spec.facts), it.item_id


def test_fact_slots_refuse_a_spec_whose_facts_were_edited():
    spec = _spec("shop", "배송 문의")
    edited = Spec.from_dict({**spec.to_dict(), "facts": ["다른 사실", *spec.facts[1:]]})
    with pytest.raises(ValueError):
        fact_slots(edited)


def test_value_mentions_keep_numeric_kinds_only():
    spec = _spec("shop", "배송 문의", "해결")  # request: order (id) + product (text); outcome: date
    mentions = value_mentions(spec)
    assert [(m.fact, m.slot, m.kind) for m in mentions] == [(0, "order", "id"), (1, "date", "date")]
    assert mentions[0].value == _slots(spec)["order"]


def test_values_are_found_in_their_normalized_forms():
    spec = _spec("parcel", "배송 지연", "부분 해결")  # request: date + tracking; outcome: time
    by = _slots(spec)
    month, day = by["date"].removesuffix("일").split("월 ")
    tracking = by["tracking"].replace("-", " ")
    summary = (
        f"고객은 운송장 {tracking} 택배가 {month}/{day}에 오지 않았다고 함. {by['time']}에 다시 연락하기로 함"
    )
    assert fact_value_ok(spec, summary) == [True, True]
    assert fact_value_ok(spec, "고객이 택배가 안 왔다고 함") == [False, False]


def test_an_afternoon_time_without_its_period_is_not_the_same_time():
    spec = next(
        s
        for i in range(400)
        if (s := make_spec("dev", "parcel", i, category="배송 지연")).resolution == "부분 해결"
        and "오후" in _slots(s)["time"]
    )
    by = _slots(spec)
    bare = by["time"].removeprefix("오후 ")
    assert fact_value_ok(spec, f"운송장 {by['tracking']}, {by['date']}. {bare}에 연락")[1] is False


def test_amounts_compare_as_numbers():
    spec = _spec("shop", "주문 취소", "해결")  # the outcome fact has the refund amount
    won = int(normalize("amount", _slots(spec)["amount"]) or 0)
    assert fact_value_ok(spec, f"주문 취소, {won:,}원 환불 접수")[1] is True
    assert fact_value_ok(spec, f"주문 취소, {won + 100:,}원 환불 접수")[1] is False


def test_facts_without_numeric_values_are_not_checked():
    spec = _spec("shop", "교환 신청", "해결")  # request has the order; the outcome fact has no slot
    assert fact_value_ok(spec, "")[1] is None


def test_hybrid_turns_value_misses_into_omissions_but_keeps_wrong():
    assert hybrid_verdicts(("포함", "포함", "포함"), [False, True, None]) == ("누락", "포함", "포함")
    assert hybrid_verdicts(("틀림", "누락"), [False, False]) == ("틀림", "누락")
    # a failed judgement has no verdicts: the hybrid starts from 포함 so only the value check speaks
    assert hybrid_verdicts((), [False, None]) == ("누락", "포함")
    with pytest.raises(ValueError):
        hybrid_verdicts(("포함",), [True, True])


def test_run_table_counts_values_facts_and_items(tmp_path):
    a = _spec("shop", "배송 문의", "해결")  # mentions: order | date
    b = _spec("parcel", "배송 지연", "부분 해결")  # mentions: date, tracking | time
    s_a = f"주문 {_slots(a)['order']} 배송 문의, {_slots(a)['date']} 도착 예정"  # all found
    s_b = f"운송장 {_slots(b)['tracking']} 지연"  # 1 of 3 found
    t = run_value_table(_run(tmp_path, [(a, s_a), (b, s_b)]), n_boot=200)
    assert t["n_items"] == 2 and t["n_mentions"] == 5
    assert math.isclose(t["value_recall"], 3 / 5)
    assert math.isclose(t["value_fact_recall"], 2 / 4)  # a: 2 of 2 value facts, b: 0 of 2
    assert math.isclose(t["items_all_values"], 1 / 2)
    assert t["by_kind"]["id"] == [2, 2] and t["by_kind"]["date"] == [1, 2] and t["by_kind"]["time"] == [0, 1]
    lo, hi = t["value_recall_ci"]
    assert lo <= t["value_recall"] <= hi


def test_run_table_reads_a_missing_prediction_as_an_empty_summary(tmp_path):
    a = _spec("shop", "배송 문의", "해결")
    t = run_value_table(_run(tmp_path, [(a, None)]), n_boot=50)
    assert t["value_recall"] == 0.0 and t["n_empty"] == 1


def test_run_table_counts_a_distractor_written_into_the_summary(tmp_path):
    spec = _spec("shop", "배송 문의", distractor=True)
    assert spec.distractor is not None
    good = run_value_table(_run(tmp_path / "g", [(spec, f"주문 {_slots(spec)['order']} 문의")]), n_boot=10)
    bad = run_value_table(_run(tmp_path / "b", [(spec, f"주문 {spec.distractor.value} 문의")]), n_boot=10)
    assert good["distractor_items"] == 1 and good["distractor_in_summary"] == 0
    assert bad["distractor_in_summary"] == 1


def test_run_table_adds_raw_and_hybrid_judge_numbers(tmp_path):
    a = _spec("shop", "배송 문의", "해결")
    summary = f"주문 {_slots(a)['order']} 배송 문의"  # the date is missing: fact 2 fails the value check
    judge = [{"item_id": a.spec_id, "verdicts": ["포함", "포함"], "wrong_statements": 0, "ok": True}]
    t = run_value_table(_run(tmp_path, [(a, summary)], judge=judge), judge_file="judge-x.jsonl", n_boot=50)
    assert t["judge"]["raw_fact_recall"] == 1.0
    assert t["judge"]["hybrid_fact_recall"] == 0.5
    assert t["judge"]["n_judge_failed"] == 0 and t["judge"]["wrong_rate"] == 0.0


def test_run_table_refuses_a_judge_file_that_misses_items(tmp_path):
    a = _spec("shop", "배송 문의", "해결")
    run = _run(tmp_path, [(a, "요약")], judge=[])
    with pytest.raises(ValueError):
        run_value_table(run, judge_file="judge-x.jsonl", n_boot=10)


def test_cli_writes_one_row_per_run(tmp_path, capsys):
    a = _spec("shop", "배송 문의", "해결")
    run = _run(tmp_path, [(a, "요약")])
    out = tmp_path / "t.json"
    assert value_facts.main([str(run), "--n-boot", "20", "--json", str(out)]) == 0
    assert "value_recall" in capsys.readouterr().out
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["run"] == str(run) and rows[0]["value_recall"] == 0.0
