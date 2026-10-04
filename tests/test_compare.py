import io
import json
import sys

from call_summary.compare import main, paired_table, pick_order, pick_table, run_table, think_rows
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


def test_paired_output_on_a_cp949_console(tmp_path, monkeypatch):
    # On this Windows setup a redirected stdout is cp949: the paired table must encode there
    # (the header once used U+2212 and the command crashed with UnicodeEncodeError).
    items = [_item(i, category=None) for i in range(4)]
    a = _write_run(tmp_path, "a", items, lambda m: "nope")
    b = _write_run(tmp_path, "b", items, lambda m: "nope")
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="cp949"))
    assert main(["--paired", str(a), str(b)]) == 0
    sys.stdout.flush()
    assert out.getvalue().decode("cp949").startswith("B - A: `b` - `a` (n=4)")


def test_load_scores_roundtrip(tmp_path):
    p = tmp_path / "judge.jsonl"
    write_jsonl(p, [{"item_id": "x", "verdicts": ["포함", "누락"], "wrong_statements": 1, "ok": True}])
    s = load_scores(p)[0]
    assert s.verdicts == ("포함", "누락") and s.wrong_statements == 1 and s.recall == 0.5


def test_pick_orders_by_exact_then_entity_f1_then_lower_hallucination(tmp_path):
    def run(name, exact, f1, hall):
        d = tmp_path / name
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"provider": name}), encoding="utf-8")
        point = {"exact": exact, "entity_f1": f1, "hallucination": hall}
        (d / "summary.json").write_text(json.dumps({"n": 240, "point": point}), encoding="utf-8")
        return str(d)

    a = run("a", 0.20, 0.90, 0.01)
    b = run("b", 0.25, 0.70, 0.10)
    c = run("c", 0.25, 0.75, 0.10)
    d = run("d", 0.25, 0.75, 0.04)
    e = run("e", 0.25, 0.75, float("nan"))  # no values predicted: hallucination undefined, ranks last
    f = run("f", 0.25, 0.75, 0.04)  # full tie with d: the earlier one in the given order stays first
    assert pick_order([a, b, c, d, e, f]) == [d, f, c, e, b, a]
    table = pick_table([a, d])
    assert table.splitlines()[2].startswith("| 1 | d (`d`) | 25.0 | 75.0 | 4.0 |")


def test_think_rows_count_thinking_tags_cut_replies_and_reply_length(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    manifest = {"provider": "ollama:m", "think": "off", "ollama_options": {"presence_penalty": 0}}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def row(reply, tokens, thinking=0, done="stop", schema=True):
        return {
            "reply": reply,
            "completion_tokens": tokens,
            "thinking_chars": thinking,
            "done_reason": done,
            "score": {"schema_ok": schema},
        }

    rows = [
        row("{}", 300),
        row("<think>음</think>{}", 500, schema=False),
        row("{}", 400, thinking=120),
        row("{", 1024, done="length", schema=False),
    ]
    write_jsonl(d / "items.jsonl", rows)
    got = think_rows([str(d)])[0]
    assert got["n"] == 4 and got["think"] == "off" and got["thinking_items"] == 1 and got["think_tags"] == 1
    assert got["length_stops"] == 1 and got["schema_ok"] == 2
    assert got["completion_p50"] == 500 and got["completion_max"] == 1024
