import json

from call_summary.compare import paired_table, run_table
from call_summary.dataset import write_jsonl
from call_summary.evaluate import run_items, summary_table
from call_summary.judge_run import load_scores
from call_summary.prompts import record_to_json
from call_summary.providers import ScriptedProvider
from tests.test_scoring import _item


def _write_run(tmp_path, name, items, reply_fn):
    rows, scores = run_items(ScriptedProvider(reply_fn), items)
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"provider": name, "data": "x"}), encoding="utf-8")
    write_jsonl(d / "items.jsonl", rows)
    t = summary_table(scores, n_boot=50)
    t["latency_s"] = {"p50": 0.0, "p95": 0.0}
    (d / "summary.json").write_text(json.dumps(t), encoding="utf-8")
    return d


def test_tables(tmp_path):
    items = [_item(i, category=None) for i in range(8)]
    good = {it.transcript: record_to_json(it.gold()) for it in items}

    def perfect(messages):
        body = messages[-1]["content"].split("상담 대화:\n", 1)[1].rsplit("\n\n상담 기록 JSON:", 1)[0]
        return good[body]

    a = _write_run(tmp_path, "bad", items, lambda m: "nope")
    b = _write_run(tmp_path, "good", items, perfect)
    table = run_table([str(a), str(b)])
    assert "| bad |" in table and "| good |" in table and "100.0 [100.0, 100.0]" in table
    paired = paired_table(str(a), str(b), n_boot=50)
    assert "| exact | 0.0 | 100.0 | 100.0 [100.0, 100.0] |" in paired


def test_load_scores_roundtrip(tmp_path):
    p = tmp_path / "judge.jsonl"
    write_jsonl(p, [{"item_id": "x", "verdicts": ["포함", "누락"], "wrong_statements": 1, "ok": True}])
    s = load_scores(p)[0]
    assert s.verdicts == ("포함", "누락") and s.wrong_statements == 1 and s.recall == 0.5
