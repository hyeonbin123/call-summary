import json
import math

from call_summary.dataset import Item
from call_summary.evaluate import pick_shots, run_items
from call_summary.prompts import build_messages, record_to_json, system_prompt
from call_summary.providers import ScriptedProvider
from call_summary.schema import parse_reply
from call_summary.scoring import (
    METRICS,
    bootstrap_ci,
    exact_rate,
    paired_bootstrap_diff,
    score_item,
    summarize,
)
from call_summary.specs import make_spec


def _item(i=0, domain="shop", category: str | None = "주문 취소"):
    spec = make_spec("dev", domain, i, category=category)
    values = " ".join(f"{sv.type} {sv.value}" for sv in spec.slots)
    turns = [
        {"speaker": "상담원", "text": "네 도토리마켓입니다."},
        {"speaker": "고객", "text": f"문의드려요. {values} 입니다."},
    ]
    return Item(
        item_id=spec.spec_id, split="dev", domain=domain, spec=spec, turns=turns, summary="요약입니다."
    )


def test_parse_reply_variants():
    it = _item()
    good = record_to_json(it.gold())
    assert parse_reply(good).schema_ok
    assert parse_reply(f"```json\n{good}\n```").schema_ok
    assert parse_reply("설명: " + good + " 끝").schema_ok
    bad = parse_reply("그런 건 모르겠어요")
    assert not bad.json_ok and not bad.schema_ok
    extra = json.loads(good)
    extra["confidence"] = 0.9
    r = parse_reply(json.dumps(extra, ensure_ascii=False))
    assert r.json_ok and not r.schema_ok
    wrong_enum = json.loads(good)
    wrong_enum["resolution"] = "완료"
    assert not parse_reply(json.dumps(wrong_enum, ensure_ascii=False)).schema_ok
    brace_in_string = json.loads(good)
    brace_in_string["summary"] = "고객이 {괄호}를 말함"
    assert parse_reply(json.dumps(brace_in_string, ensure_ascii=False)).schema_ok


def test_perfect_prediction_is_exact():
    it = _item()
    s = score_item(it.item_id, it.domain, it.gold(), parse_reply(record_to_json(it.gold())), it.transcript)
    assert (
        s.exact and s.entity_fp == 0 and s.entity_fn == 0 and s.entity_hallucinated == 0 and s.off_list == 0
    )


def test_format_failure_scores_zero():
    it = _item()
    s = score_item(it.item_id, it.domain, it.gold(), parse_reply("nope"), it.transcript)
    assert not s.schema_ok and not s.exact
    assert s.entity_fn == len(it.gold().entities)
    assert s.action_fn == len(it.gold().actions_taken)


def test_equivalent_amount_surface_matches():
    it = _item()
    gold = it.gold()
    pred = gold.model_copy(deep=True)
    for e in pred.entities:
        if e.type == "금액":
            # rewrite in the other surface style
            from call_summary.values import normalize

            e.value = f"{int(normalize('amount', e.value)):,}원"
    s = score_item(it.item_id, it.domain, gold, parse_reply(record_to_json(pred)), it.transcript)
    assert s.entity_fp == 0 and s.entity_fn == 0


def test_wrong_and_invented_entities():
    it = _item()
    gold = it.gold()
    pred = gold.model_copy(deep=True)
    pred.entities[0].value = "999-9999-99"  # wrong order id, not in the transcript
    pred.entities.append(type(pred.entities[0])(type="날짜", value="11월 30일"))
    s = score_item(it.item_id, it.domain, gold, parse_reply(record_to_json(pred)), it.transcript)
    assert s.entity_tp == len(gold.entities) - 1
    assert s.entity_fp == 2 and s.entity_fn == 1
    assert s.entity_hallucinated == 2
    assert not s.exact


def test_off_list_labels_are_counted():
    it = _item()
    pred = it.gold().model_copy(deep=True)
    pred.category = "기타"
    pred.actions_taken.append("마음대로 처리")
    pred.entities.append(type(pred.entities[0])(type="고객명", value="홍길동"))
    s = score_item(it.item_id, it.domain, it.gold(), parse_reply(record_to_json(pred)), it.transcript)
    assert s.off_list == 3
    assert not s.category_ok


def test_bootstrap_and_paired_diff():
    items = [_item(i) for i in range(20)]
    good = [
        score_item(it.item_id, it.domain, it.gold(), parse_reply(record_to_json(it.gold())), it.transcript)
        for it in items
    ]
    bad = [score_item(it.item_id, it.domain, it.gold(), parse_reply("x"), it.transcript) for it in items]
    point, lo, hi = bootstrap_ci(good, exact_rate, n_boot=200)
    assert point == lo == hi == 1.0
    d, dlo, dhi = paired_bootstrap_diff(bad, good, exact_rate, n_boot=200)
    assert d == 1.0 and dlo == 1.0
    mixed = good[:10] + bad[10:]
    p, lo, hi = bootstrap_ci(mixed, exact_rate, n_boot=500)
    assert lo < p < hi
    summary = summarize(bad)
    assert set(summary) == set(METRICS)
    assert math.isnan(summary["hallucination"])  # nothing predicted


def test_system_prompt_lists_domain_labels():
    p = system_prompt("card")
    assert "구름카드" in p and "카드 분실" in p and "가맹점명" in p
    assert "도토리마켓" not in p


def test_run_items_with_scripted_provider():
    items = [_item(i, category=None) for i in range(6)]
    pool = [_item(100 + i, category=None) for i in range(12)]
    by_transcript = {it.transcript: record_to_json(it.gold()) for it in items}

    def reply(messages):
        text = messages[-1]["content"]
        body = text.split("상담 대화:\n", 1)[1].rsplit("\n\n상담 기록 JSON:", 1)[0]
        return by_transcript[body]

    provider = ScriptedProvider(reply)
    rows, scores = run_items(provider, items, lambda d: pick_shots(pool, d, 2))
    assert all(s.exact for s in scores)
    assert len(rows) == 6
    # 1 system + 2 shots * 2 + 1 user
    assert len(provider.calls[0]) == 6
    assert provider.calls[0] == build_messages("shop", items[0].transcript, pick_shots(pool, "shop", 2))
