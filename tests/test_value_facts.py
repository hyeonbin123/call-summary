# [테스트 묶음] 요약 값 확인 V(call_summary.value_facts)를 검증한다: 사실 틀에서 슬롯을 거꾸로 읽기(생성한
# 명세·커밋된 데이터 모두), 고친 사실의 거절, 숫자형 종류만 고르기, 정규화한 꼴로 찾기(날짜·번호·금액·시각),
# 숫자 값이 없는 사실은 확인하지 않기, V+J 합치기, 실행 하나의 표(값 보존율·사실·건·종류별·교란 번호·판정
# 지표), 명령줄.
# [왜 중요한가] V는 판정 모델 없이 결정적으로 도는 보조 지표이고, 채택한 판정 형태(V+J1)의 대부분이 V에서
# 왔다(README 5단계 요약 품질). 사실과 슬롯의 대응이 틀리면 엉뚱한 값을 찾게 되므로 대응을 확인하고, 다르면
# 멈춰야 한다.

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


# 원하는 업종·유형(그리고 처리 결과, 말실수 사건 여부)의 명세를 dev 번호 순서로 찾는다. 명세가 결정적이라 매번
# 같은 명세가 나온다.
def _spec(domain: str, category: str, want_resolution: str | None = None, distractor: bool = False) -> Spec:
    for i in range(400):
        spec = make_spec("dev", domain, i, category=category)
        if want_resolution is not None and spec.resolution != want_resolution:
            continue
        if distractor and spec.distractor is None:
            continue
        return spec
    raise AssertionError("no such spec")


# 슬롯 이름 → 명세의 값(말투 그대로).
def _slots(spec: Spec) -> dict[str, str]:
    return {sv.slot: sv.value for sv in spec.slots}


# 슬롯 값을 모두 한 발화로 말하는 가짜 건.
def _item(spec: Spec) -> Item:
    turns = [{"speaker": "고객", "text": " ".join(sv.value for sv in spec.slots)}]
    return Item(item_id=spec.spec_id, split="dev", domain=spec.domain, spec=spec, turns=turns)


# 기록된 평가 실행 꼴(데이터 파일, manifest, items.jsonl)을 만든다. 요약이 None이면 pred가 없는 건(스키마
# 실패)이고, judge를 주면 판정 파일 judge-x.jsonl도 쓴다.
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


# dev(학습 업종)와 test-c(카드) 명세에서 모든 사실의 슬롯을 틀로 되살릴 수 있고, 찾은 슬롯이 모두 그 명세에
# 있는지 본다.
def test_fact_slots_rebuild_every_generated_fact():
    for split in ("dev", "test-c"):
        for domain in SPLIT_DOMAINS[split]:
            for i in range(30):
                spec = make_spec(split, domain, i)
                slots = fact_slots(spec)
                assert len(slots) == len(spec.facts)
                assert all(set(s) <= set(_slots(spec)) for s in slots)


# 커밋된 데이터(dev, dev-asr-v2, test-a, test-c, test-d-v2)의 모든 명세에서도 사실과 틀이 맞아야 한다.
# 데이터를 만든 뒤 틀이 바뀌었다면 여기서 드러난다. 파일이 없는 체크아웃이면 건너뛴다.
def test_fact_slots_match_the_committed_datasets():
    root = Path(__file__).resolve().parents[1] / "datasets"
    for name in ("dev.jsonl", "dev-asr-v2.jsonl", "test-a.jsonl", "test-c.jsonl", "test-d-v2.jsonl"):
        path = root / name
        if not path.exists():
            pytest.skip(f"{name} not in this checkout")
        for it in load_items(path):
            assert len(fact_slots(it.spec)) == len(it.spec.facts), it.item_id


# 사실 하나를 손으로 바꾼 명세는 틀을 다시 채운 문장과 달라 ValueError로 멈춘다(규칙: 다르면 멈춘다).
def test_fact_slots_refuse_a_spec_whose_facts_were_edited():
    spec = _spec("shop", "배송 문의")
    edited = Spec.from_dict({**spec.to_dict(), "facts": ["다른 사실", *spec.facts[1:]]})
    with pytest.raises(ValueError):
        fact_slots(edited)


# 쇼핑몰 배송 문의(해결): 요청 사실의 주문번호(id)와 결과 사실의 날짜(date)만 값 사실로 고르고, 상품명(text)은
# 판정 모델에 맡겨 고르지 않는다.
def test_value_mentions_keep_numeric_kinds_only():
    spec = _spec("shop", "배송 문의", "해결")  # request: order (id) + product (text); outcome: date
    mentions = value_mentions(spec)
    assert [(m.fact, m.slot, m.kind) for m in mentions] == [(0, "order", "id"), (1, "date", "date")]
    assert mentions[0].value == _slots(spec)["order"]


# 요약이 날짜를 "10/5"처럼, 운송장번호를 하이픈 대신 공백으로 적어도 채점과 같은 정규화로 찾는다. 값이 하나도
# 없는 요약은 두 사실 모두 False다.
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


# "오후 2시"를 "2시"로만 적으면 같은 시각이 아니다(시간대 낱말이 없으면 12를 더하지 않아 다른 시각으로
# 정규화된다). 규칙 1: 시각은 오전·오후까지 맞아야 한다.
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


# 금액은 말투와 상관없이 숫자로 비교한다: 정답 금액을 "32,000원" 꼴로 다시 적어도 찾고, 100원만 달라도 찾지
# 못한다.
def test_amounts_compare_as_numbers():
    spec = _spec("shop", "주문 취소", "해결")  # the outcome fact has the refund amount
    won = int(normalize("amount", _slots(spec)["amount"]) or 0)
    assert fact_value_ok(spec, f"주문 취소, {won:,}원 환불 접수")[1] is True
    assert fact_value_ok(spec, f"주문 취소, {won + 100:,}원 환불 접수")[1] is False


# 숫자 값이 없는 사실은 None(확인하지 않음)이다. 빈 요약이어도 False가 아니라 None이라, V+J에서는 판정 모델의
# 판정이 그대로 남는다.
def test_facts_without_numeric_values_are_not_checked():
    spec = _spec("shop", "교환 신청", "해결")  # request has the order; the outcome fact has no slot
    assert fact_value_ok(spec, "")[1] is None


# V+J: 값이 빠진 사실의 '포함'만 '누락'으로 바뀌고, 판정 모델의 '틀림'·'누락'은 그대로다. 판정이 없으면(실패)
# '포함'에서 시작해 V만 반영하고, 판정 수와 사실 수가 다르면 예외다.
def test_hybrid_turns_value_misses_into_omissions_but_keeps_wrong():
    assert hybrid_verdicts(("포함", "포함", "포함"), [False, True, None]) == ("누락", "포함", "포함")
    assert hybrid_verdicts(("틀림", "누락"), [False, False]) == ("틀림", "누락")
    # a failed judgement has no verdicts: the hybrid starts from 포함 so only the value check speaks
    assert hybrid_verdicts((), [False, None]) == ("누락", "포함")
    with pytest.raises(ValueError):
        hybrid_verdicts(("포함",), [True, True])


# 두 건의 실행에서 지표를 손 계산과 맞춘다: 값 5개 중 3개(값 보존율 3/5), 값 사실 4개 중 2개, 모든 값을 찾은
# 건 1/2, 종류별 [찾음, 전체](번호 2/2, 날짜 1/2, 시각 0/1). 부트스트랩 구간이 점 추정을 감싸는지도 본다.
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


# 답이 스키마를 통과하지 못해 pred가 없는 건은 빈 요약으로 세어 값을 하나도 찾지 못한 것으로 친다(규칙 1).
def test_run_table_reads_a_missing_prediction_as_an_empty_summary(tmp_path):
    a = _spec("shop", "배송 문의", "해결")
    t = run_value_table(_run(tmp_path, [(a, None)]), n_boot=50)
    assert t["value_recall"] == 0.0 and t["n_empty"] == 1


# 말실수 사건이 있는 건에서 요약이 정정한 번호를 쓰면 교란 번호 0, 고치기 전 번호를 쓰면 1로 센다.
def test_run_table_counts_a_distractor_written_into_the_summary(tmp_path):
    spec = _spec("shop", "배송 문의", distractor=True)
    assert spec.distractor is not None
    good = run_value_table(_run(tmp_path / "g", [(spec, f"주문 {_slots(spec)['order']} 문의")]), n_boot=10)
    bad = run_value_table(_run(tmp_path / "b", [(spec, f"주문 {spec.distractor.value} 문의")]), n_boot=10)
    assert good["distractor_items"] == 1 and good["distractor_in_summary"] == 0
    assert bad["distractor_in_summary"] == 1


# 판정 모델이 두 사실 모두 '포함'으로 봐도 요약에 날짜가 없으면 V+J에서는 둘째 사실이 '누락'이 되어 재현율이
# 1.0 → 0.5로 내려간다. 판정 모델이 값이 빠진 사실도 포함으로 보던 문제(docs/experiments.md "값 확인 결과")를
# V가 바로잡는 방식이다.
def test_run_table_adds_raw_and_hybrid_judge_numbers(tmp_path):
    a = _spec("shop", "배송 문의", "해결")
    summary = f"주문 {_slots(a)['order']} 배송 문의"  # the date is missing: fact 2 fails the value check
    judge = [{"item_id": a.spec_id, "verdicts": ["포함", "포함"], "wrong_statements": 0, "ok": True}]
    t = run_value_table(_run(tmp_path, [(a, summary)], judge=judge), judge_file="judge-x.jsonl", n_boot=50)
    assert t["judge"]["raw_fact_recall"] == 1.0
    assert t["judge"]["hybrid_fact_recall"] == 0.5
    assert t["judge"]["n_judge_failed"] == 0 and t["judge"]["wrong_rate"] == 0.0


# 판정 파일에 실행의 건이 빠져 있으면(다른 실행의 파일을 준 경우 등) 일부만으로 계산하지 않고 멈춘다.
def test_run_table_refuses_a_judge_file_that_misses_items(tmp_path):
    a = _spec("shop", "배송 문의", "해결")
    run = _run(tmp_path, [(a, "요약")], judge=[])
    with pytest.raises(ValueError):
        run_value_table(run, judge_file="judge-x.jsonl", n_boot=10)


# 명령줄이 표 머리줄을 찍고, --json 파일에 실행마다 한 행(실행 경로, 값 보존율)을 쓰는지 본다.
def test_cli_writes_one_row_per_run(tmp_path, capsys):
    a = _spec("shop", "배송 문의", "해결")
    run = _run(tmp_path, [(a, "요약")])
    out = tmp_path / "t.json"
    assert value_facts.main([str(run), "--n-boot", "20", "--json", str(out)]) == 0
    assert "value_recall" in capsys.readouterr().out
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["run"] == str(run) and rows[0]["value_recall"] == 0.0
