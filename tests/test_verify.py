import json
import random
import re

from call_summary import verify
from call_summary.dataset import Item, write_jsonl
from call_summary.domains import DOMAINS
from call_summary.schema import AfterCallRecord, Entity, FollowUp
from call_summary.specs import make_spec
from call_summary.values import normalize


def _record(*entities: tuple[str, str]) -> AfterCallRecord:
    return AfterCallRecord(
        category="주문 취소",
        resolution="해결",
        entities=[Entity(type=t, value=v) for t, v in entities],
        actions_taken=[],
        follow_up=FollowUp(required=False, codes=[]),
        summary="요약입니다.",
    )


def _reasons(domain, record, transcript):
    return [(f.type, f.value, f.reasons) for f in verify.verify_entities(domain, record, transcript)]


def test_every_identifier_type_has_a_format_its_values_pass():
    rng = random.Random(0)
    for d in DOMAINS.values():
        for et in d.entity_types:
            if et.kind != "id":
                assert et.id_format is None, et.label
                continue
            assert et.id_format, et.label
            for _ in range(200):
                assert re.fullmatch(et.id_format, normalize("id", et.make(rng))), et.label


def test_format_and_transcript_checks():
    transcript = "고객: 주문번호 D2031905요. 아니 D795268이요.\n상담원: 3만 2천 원 환불됩니다."
    rec = _record(
        ("주문번호", "D2031905"),  # right format, said in the call
        ("주문번호", "D795268"),  # copied from the transcript, one digit short
        ("주문번호", "D1111111"),  # right format, never said
        ("주문번호", "B7952681"),  # D heard as B and not in the call
        ("금액", "9만 원"),  # not an identifier: never checked
    )
    before = rec.model_dump()
    assert _reasons("shop", rec, transcript) == [
        ("주문번호", "D795268", ["format"]),
        ("주문번호", "D1111111", ["not_in_transcript"]),
        ("주문번호", "B7952681", ["format", "not_in_transcript"]),
    ]
    assert rec.model_dump() == before  # the record is never changed


def test_format_is_checked_on_the_normalized_value():
    # Written with spaces or in lower case, the number is the same one (scoring compares normalized values).
    transcript = "고객: 운송장 번호는 6759, 9785, 4536이에요."
    rec = _record(("운송장번호", "6759 9785 4536"), ("운송장번호", "6759-9785-4536"))
    assert _reasons("parcel", rec, transcript) == []
    assert _reasons("shop", _record(("주문번호", "d2031905")), "주문번호 D2031905") == []
    # Twelve digits found in the transcript but with a digit too many is still a format problem.
    got = _reasons("parcel", _record(("운송장번호", "6936-8635-34496")), "6936 8635 34496")
    assert got == [("운송장번호", "6936-8635-34496", ["format"])]


def test_other_identifier_types():
    assert _reasons("telecom", _record(("접수번호", "R12345")), "접수번호 R12345") == [
        ("접수번호", "R12345", ["format"])
    ]
    assert _reasons("telecom", _record(("접수번호", "R123456")), "접수번호 알 123456") == []
    assert _reasons("card", _record(("카드 끝자리", "12345")), "끝자리 12345") == [
        ("카드 끝자리", "12345", ["format"])
    ]


def _item(domain: str, index: int, category: str | None, turns: list[dict]) -> Item:
    spec = make_spec("dev", domain, index, category=category)
    return Item(spec.spec_id, "dev-asr", domain, spec, turns, summary="요약입니다.")


def test_guard_report_counts(tmp_path):
    shop = _item(
        "shop",
        3,
        "주문 취소",
        [{"speaker": "고객", "text": "주문번호 D2031905, 아니 D2031906이요. D203190 맞나?"}],
    )
    gold_order = next(e.value for e in shop.gold().entities if e.type == "주문번호")
    assert gold_order == "D2031905"
    parcel = _item("parcel", 3, None, [{"speaker": "고객", "text": "운송장 5213 7079 1436이요."}])
    gold_tracking = next(e.value for e in parcel.gold().entities if e.type == "운송장번호")
    data = tmp_path / "dev-asr.jsonl"
    write_jsonl(data, [shop.to_dict(), parcel.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    shop_pred = shop.gold().model_copy(
        update={
            "entities": [
                Entity(type="주문번호", value="D2031905"),  # correct
                Entity(type="주문번호", value="D2031906"),  # wrong, in the transcript: silent
                Entity(type="주문번호", value="D203190"),  # wrong, format: flagged
                Entity(type="상품명", value="아무거나"),  # not an identifier
            ]
        }
    )
    parcel_pred = parcel.gold().model_copy(
        update={"entities": [Entity(type="운송장번호", value="5213707914366")]}  # wrong, both reasons
    )
    rows = [
        {"item_id": shop.item_id, "pred": shop_pred.model_dump()},
        {"item_id": parcel.item_id, "pred": parcel_pred.model_dump()},
        {"item_id": "dev-shop-99999", "pred": None},  # rejected reply: nothing goes out
    ]
    write_jsonl(run / "items.jsonl", rows)

    got = verify.guard_report(run)
    assert (got["predicted"], got["correct"], got["wrong"]) == (4, 1, 3)
    assert (got["flagged_correct"], got["flagged_wrong"]) == (0, 2)
    assert got["flagged_wrong_reasons"] == {"format": 1, "format+not_in_transcript": 1}
    assert got["silent_wrong"] == [[shop.item_id, "주문번호", "D2031906", ["D2031905"], "one_char"]]
    assert got["silent_wrong_shapes"] == {"one_char": 1}
    assert got["gold_ids_not_predicted"] == 1 and gold_tracking == "5213-7079-1436"
    assert got["rows_without_record"] == 1
    assert got["by_type"]["운송장번호"]["flagged_wrong"] == 1
    table = verify.guard_table({"run": got})
    assert "4 | 1 | 3 | 0 (0.0%) | 2/3 (66.7%) | 3/4 (75.0%) → 1/4 (25.0%)" in table


def test_guard_report_false_flags_and_written_rule(tmp_path):
    # A correct number the recogniser lost (the model restored it) is flagged: a false flag by the metric.
    parcel = _item("parcel", 3, None, [{"speaker": "고객", "text": "운송장 오이일삼 칠공칠구 1436이요."}])
    data = tmp_path / "d.jsonl"
    write_jsonl(data, [parcel.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    pred = parcel.gold().model_copy(update={"entities": [Entity(type="운송장번호", value="521370791436")]})
    write_jsonl(run / "items.jsonl", [{"item_id": parcel.item_id, "pred": pred.model_dump()}])
    got = verify.guard_report(run, data=data)
    assert (got["correct"], got["flagged_correct"]) == (1, 1)
    assert got["false_flags"] == [[parcel.item_id, "운송장번호", "521370791436", ["not_in_transcript"]]]
    # Written without hyphens: the brief's raw-string rule would also have flagged its format.
    assert got["written_format_rule_would_also_flag"] == {"correct": 1, "wrong": 0}
