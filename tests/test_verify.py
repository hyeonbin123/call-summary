# [테스트 묶음] 확인이 필요한 번호 표시(call_summary.verify)를 검증한다: 번호 유형마다 배포 형식이 있고
# 생성기가 만든 값이 그 형식을 통과하는지, 형식 검사와 전사 포함 검사의 사유, 형식을 정규화한 값에서 보는지,
# 다른 번호 유형, 기록된 실행에 대 보는 집계(guard_report)와 표.
# [왜 중요한가] 이 표시를 서비스에 넣은 기준은 맞은 번호를 거의 표시하지 않는 것(오검출 2%
# 이하)이었다(docs/experiments.md "확인이 필요한 번호 표시" 채택 기준). 맞은 번호를 표시하거나 기록을 바꾸면
# 안 되고, 결과 표의 수치가 이 집계에서 나온다.

import json
import random
import re

from call_summary import verify
from call_summary.dataset import Item, write_jsonl
from call_summary.domains import DOMAINS
from call_summary.schema import AfterCallRecord, Entity, FollowUp
from call_summary.specs import make_spec
from call_summary.values import normalize


# 핵심 값만 바꿔 끼운 쇼핑몰 기록.
def _record(*entities: tuple[str, str]) -> AfterCallRecord:
    return AfterCallRecord(
        category="주문 취소",
        resolution="해결",
        entities=[Entity(type=t, value=v) for t, v in entities],
        actions_taken=[],
        follow_up=FollowUp(required=False, codes=[]),
        summary="요약입니다.",
    )


# verify_entities 결과를 (유형, 값, 사유) 목록으로 바꿔 비교하기 쉽게 한다.
def _reasons(domain, record, transcript):
    return [(f.type, f.value, f.reasons) for f in verify.verify_entities(domain, record, transcript)]


# 번호 종류(id)의 유형에만 배포 형식이 있고, 생성기가 만든 값 200개가 모두 정규화한 꼴로 형식을 통과한다.
# 생성기의 맞은 값이 형식 오류로 표시되는 일을 막는다.
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


# 사유 확인: 형식이 맞고 전사에 있는 번호는 통과, 전사의 한 자리 모자란 번호를 옮긴 것은 format, 형식은 맞지만
# 말한 적 없는 번호는 not_in_transcript, D를 B로 잘못 들은 번호는 둘 다. 금액 같은 다른 유형은 검사하지 않고,
# 기록은 바뀌지 않는다.
def test_format_and_transcript_checks():
    transcript = "고객: 주문번호 D2031905요. 아니 D795268이요.\n상담원: 3만 2천 원 환불됩니다."
    rec = _record(
        ("주문번호", "D2031905"),  # 형식이 맞고 통화에서 말함
        ("주문번호", "D795268"),  # 전사에서 옮겼지만 한 자리 모자람
        ("주문번호", "D1111111"),  # 형식은 맞지만 말한 적 없음
        ("주문번호", "B7952681"),  # D를 B로 잘못 들었고 통화에도 없음
        ("금액", "9만 원"),  # 번호가 아니라 검사하지 않음
    )
    before = rec.model_dump()
    assert _reasons("shop", rec, transcript) == [
        ("주문번호", "D795268", ["format"]),
        ("주문번호", "D1111111", ["not_in_transcript"]),
        ("주문번호", "B7952681", ["format", "not_in_transcript"]),
    ]
    assert rec.model_dump() == before  # 기록은 절대 바뀌지 않는다


# 띄어 쓰거나 하이픈으로 적거나 소문자로 적은 맞은 번호는 표시하지 않는다(채점과 같은 정규화). 쉼표로 나뉘어
# 들린 전사에서도 넓은 번호 규칙으로 찾는다. 숫자가 하나 많은 운송장번호는 전사에 있어도 형식 오류로
# 표시한다(docs/experiments.md "브리프와 다른 점": 형식은 정규화한 값에서 본다).
def test_format_is_checked_on_the_normalized_value():
    transcript = "고객: 운송장 번호는 6759, 9785, 4536이에요."
    rec = _record(("운송장번호", "6759 9785 4536"), ("운송장번호", "6759-9785-4536"))
    assert _reasons("parcel", rec, transcript) == []
    assert _reasons("shop", _record(("주문번호", "d2031905")), "주문번호 D2031905") == []
    got = _reasons("parcel", _record(("운송장번호", "6936-8635-34496")), "6936 8635 34496")
    assert got == [("운송장번호", "6936-8635-34496", ["format"])]


# 접수번호(R + 6자리)와 카드 끝자리(4자리)의 형식 검사. "알 123456"처럼 글자 이름과 떨어져 들린 접수번호도
# 숫자 덩어리 규칙으로 전사에서 찾는다(서비스는 전사가 음성 인식 결과인지 모르므로 늘 넓은 규칙을 쓴다).
def test_other_identifier_types():
    assert _reasons("telecom", _record(("접수번호", "R12345")), "접수번호 R12345") == [
        ("접수번호", "R12345", ["format"])
    ]
    assert _reasons("telecom", _record(("접수번호", "R123456")), "접수번호 알 123456") == []
    assert _reasons("card", _record(("카드 끝자리", "12345")), "끝자리 12345") == [
        ("카드 끝자리", "12345", ["format"])
    ]


# 음성 조건(dev-asr) 분할의 가짜 건. 정답은 명세에서 나오고 발화는 직접 준다.
def _item(domain: str, index: int, category: str | None, turns: list[dict]) -> Item:
    spec = make_spec("dev", domain, index, category=category)
    return Item(spec.spec_id, "dev-asr", domain, spec, turns, summary="요약입니다.")


# guard_report의 집계를 손으로 만든 실행과 맞춘다: 예측한 번호 4개 중 맞음 1·틀림 3, 표시된 틀린 번호 2(형식
# 1, 형식+전사 없음 1), 표시 안 된 틀린 번호 1(정답과 한 글자 다름, 5→6), 답에 없는 정답 번호 1(운송장번호),
# 기록 없는 행 1(서비스가 거절한 답은 번호가 나가지 않음). 표 한 줄의 오검출률·표시율·조용히 나가는 틀린
# 번호(표시 전 → 후)도 문서의 표 꼴 그대로인지 본다.
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
                Entity(type="주문번호", value="D2031905"),  # 맞음
                Entity(type="주문번호", value="D2031906"),  # 틀렸지만 전사에 있음: 표시 없이 나감
                Entity(type="주문번호", value="D203190"),  # 틀림, 형식 오류: 표시됨
                Entity(type="상품명", value="아무거나"),  # 번호가 아님
            ]
        }
    )
    parcel_pred = parcel.gold().model_copy(
        update={"entities": [Entity(type="운송장번호", value="5213707914366")]}  # 틀림, 두 사유 모두
    )
    rows = [
        {"item_id": shop.item_id, "pred": shop_pred.model_dump()},
        {"item_id": parcel.item_id, "pred": parcel_pred.model_dump()},
        {"item_id": "dev-shop-99999", "pred": None},  # 거절된 답: 아무것도 나가지 않음
    ]
    write_jsonl(run / "items.jsonl", rows)

    got = verify.guard_report(run)
    assert (got["predicted"], got["correct"], got["wrong"]) == (4, 1, 3)
    assert (got["flagged_correct"], got["flagged_wrong"]) == (0, 2)
    assert got["flagged_wrong_reasons"] == {"format": 1, "format+not_in_transcript": 1}
    assert got["silent_wrong"] == [[shop.item_id, "주문번호", "D2031906", ["D2031905"], "one_char", "5>6"]]
    assert got["silent_wrong_shapes"] == {"one_char": 1}
    assert got["silent_wrong_one_char_changes"] == {"5>6": 1}
    assert got["gold_ids_not_predicted"] == 1 and gold_tracking == "5213-7079-1436"
    assert got["rows_without_record"] == 1
    assert got["by_type"]["운송장번호"]["flagged_wrong"] == 1
    table = verify.guard_table({"run": got})
    assert "4 | 1 | 3 | 0 (0.0%) | 2/3 (66.7%) | 3/4 (75.0%) → 1/4 (25.0%)" in table


# 인식기가 번호 일부를 한글로 적어 전사에서 찾을 수 없지만 모델이 맞게 복원한 번호는 표시된다. 지표로는
# 오검출(맞은 번호를 표시)로 세고 목록에 남긴다. 하이픈 없이 적은 이 번호는 값 그대로 형식 규칙이었다면 형식
# 오류로도 걸렸을 것이라 그 수(맞음 1)도 센다.
def test_guard_report_false_flags_and_written_rule(tmp_path):
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
    assert got["written_format_rule_would_also_flag"] == {"correct": 1, "wrong": 0}
