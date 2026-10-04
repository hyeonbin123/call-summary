import json

from call_summary import asr_report
from call_summary.dataset import write_jsonl
from call_summary.schema import Entity, ParseResult
from call_summary.scoring import score_item, unfound_split
from tests.test_scoring import _item


def _pred(it, extra=()):
    gold = it.gold()
    return gold.model_copy(update={"entities": [*gold.entities, *extra]})


def test_unfound_values_split_into_restored_and_invented():
    it = _item(0)  # shop: every gold value is said in the transcript
    gold = it.gold()
    invented = Entity(type="금액", value="999,900원")
    unknown = Entity(type="좌석번호", value="12A")  # not a type of this domain
    pred = _pred(it, [invented, unknown])

    restored, made_up = unfound_split(it.domain, gold, pred, it.transcript)
    assert not restored and made_up == {"금액": 1, "좌석번호": 1}

    silent = "상담원: 네 도토리마켓입니다."  # the values did not survive: copying gold back is a restoration
    restored, made_up = unfound_split(it.domain, gold, pred, silent)
    assert sum(restored.values()) == len(gold.entities)
    assert made_up == {"금액": 1, "좌석번호": 1}
    parsed = ParseResult(json_ok=True, schema_ok=True, record=pred)
    score = score_item(it.item_id, it.domain, gold, parsed, silent, spoken=True)
    assert score.entity_hallucinated == sum(restored.values()) + sum(made_up.values())


def test_survival_table_per_file(tmp_path):
    it = _item(0)
    heard = _item(0)
    heard.turns = [{"speaker": "상담원", "text": "네 도토리마켓입니다."}]
    write_jsonl(tmp_path / "v1.jsonl", [it.to_dict()])
    write_jsonl(tmp_path / "v2.jsonl", [heard.to_dict()])
    got = asr_report.survival([tmp_path / "v1.jsonl", tmp_path / "v2.jsonl"])
    types = {e.type for e in it.gold().entities}
    v1, v2 = got[str(tmp_path / "v1.jsonl")], got[str(tmp_path / "v2.jsonl")]
    assert v1["items"] == 1 and v1["all_found"] == 1
    assert v2["all_found"] == 0
    assert set(v1["types"]) == types and all(found == n for found, n in v1["types"].values())
    assert all(found == 0 for found, _ in v2["types"].values())
    table = asr_report.survival_table(got)
    assert "모든 값이 남은 건" in table and "100.0" in table and "0.0" in table


def test_unfound_report_of_a_run(tmp_path):
    it = _item(0)
    it.split = "dev-asr"
    it.turns = [{"speaker": "상담원", "text": "네 도토리마켓입니다."}]
    data = tmp_path / "dev-asr.jsonl"
    write_jsonl(data, [it.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    pred = _pred(it, [Entity(type="금액", value="999,900원")])
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "pred": pred.model_dump()}])

    got = asr_report.unfound(run)
    n_gold = len(it.gold().entities)
    assert got["predicted"] == n_gold + 1
    assert sum(got["restored"].values()) == n_gold and got["invented"] == {"금액": 1}
    assert "복원" in asr_report.unfound_table({"run": got})


def test_unfound_report_counts_empty_values(tmp_path):
    # Stage 3 reported hallucination with and without empty values ("" filled into a type the call never
    # mentioned): empty values are unfound and invented, and the table gives the rate without them too.
    it = _item(0)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    extra = [
        Entity(type="금액", value=""),
        Entity(type="금액", value=" "),
        Entity(type="금액", value="999,900원"),
    ]
    pred = _pred(it, extra)
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "pred": pred.model_dump()}])

    got = asr_report.unfound(run)
    n = len(it.gold().entities) + 3
    assert got["predicted"] == n and got["invented"] == {"금액": 3} and got["empty"] == 2
    table = asr_report.unfound_table({"run": got})
    assert "빈 값" in table and f"{100 * 1 / n:.1f}% (1)" in table


def test_unfound_report_counts_placeholder_values(tmp_path):
    # The 2026 comparators wrote a word for "no value" ("없음", "알 수 없음") where stage 3's 14B wrote "":
    # the same failure in text form. Placeholders are counted on their own so `empty` stays as stage 3
    # counted it, and the table gives the rate without both.
    it = _item(0)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    extra = [
        Entity(type="금액", value="없음"),
        Entity(type="주문번호", value="알 수 없음"),
        Entity(type="금액", value=" 미정 "),
        Entity(type="금액", value=""),
        Entity(type="금액", value="999,900원"),
        Entity(type="금액", value="없음 아님"),  # not a placeholder word: an invented value
    ]
    pred = _pred(it, extra)
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "pred": pred.model_dump()}])

    got = asr_report.unfound(run)
    n = len(it.gold().entities) + 6
    assert got["predicted"] == n and got["invented"] == {"금액": 5, "주문번호": 1}
    assert got["empty"] == 1 and got["placeholder"] == 3
    assert asr_report.is_placeholder("알수 없음") and not asr_report.is_placeholder("")
    table = asr_report.unfound_table({"run": got})
    assert "자리 표시 값" in table
    assert f"{100 * 5 / n:.1f}% (5)" in table  # without empty values only, as before
    assert f"{100 * 2 / n:.1f}% (2)" in table  # without empty and placeholder values


WRITTEN = "운송장 번호 6291-7877-5168로 받은 건이에요."


def test_tracking_form_classes():
    def form(heard):
        return asr_report.tracking_form("6291-7877-5168", WRITTEN, heard, WRITTEN.index("6291"))

    assert form(WRITTEN) == ("as_written", "6291-7877-5168")
    assert form("운송장 번호 6291, 7877, 5168로 받은 건이에요.")[0] == "split_found"
    assert form("운송장 번호 육이구일 칠팔칠칠 오일육팔로 받은 건이에요.")[0] == "hangul_digit_names"
    # one digit heard as a Hangul syllable inside the number
    assert form("운송장 번호 629일 7877 5168로 받은 건이에요.") == ("lost_hangul", "629일 7877 5168")
    assert form("운송장 번호 629-7877-5168로 받은 건이에요.")[0] == "lost_fewer"
    # an inserted digit at the edge stays in the span; a date elsewhere in the utterance does not count
    assert form("운송장 번호 6291-7877-52685로 받은 건이에요.") == ("lost_more", "6291-7877-52685")
    dated = "운송장 번호 6291-7877-5268로 10월 20일에 받은 건이에요."
    assert form(dated) == ("lost_changed", "6291-7877-5268")
    # the v1 range reading has its own row: "에서" alone does not make the hangul class
    assert form("운송장 번호 6291 78717에서 5168로 받은 건이에요.") == ("lost_more", "6291 78717에서 5168")


def test_tracking_forms_per_file(tmp_path):
    it = _item(0, domain="parcel", category=None)
    it.split = "dev-asr"
    it.meta = {"asr": {"written": [WRITTEN, "네 확인했습니다.", WRITTEN]}}
    it.turns = [
        {"speaker": "고객", "text": "운송장 번호 6291-7877-5168로 받은 건이에요."},
        {"speaker": "상담원", "text": "네 확인했습니다."},
        {"speaker": "고객", "text": "운송장 번호 6291 78717에서 5168로 받은 건이에요."},
    ]
    path = tmp_path / "dev-asr.jsonl"
    write_jsonl(path, [it.to_dict()])
    got = asr_report.tracking_forms([path])[str(path)]
    assert got["numbers"] == 2 and got["as_written"] == 1 and got["lost_more"] == 1
    assert got["heard_has_에서"] == 1 and got["lost_hangul"] == 0
    assert got["lost"] == [[it.item_id, "6291-7877-5168", "lost_more", "6291 78717에서 5168"]]
    table = asr_report.tracking_forms_table({str(path): got})
    assert "| 찾지 못함 | 1 |" in table and "| 번호 수 | 2 |" in table
