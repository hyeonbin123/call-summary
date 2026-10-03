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
