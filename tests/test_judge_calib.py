import json
import math
from pathlib import Path

import pytest

from call_summary import judge_calib, judge_run
from call_summary.dataset import Item, read_jsonl, write_jsonl
from call_summary.judge import cohen_kappa, judge_reply
from call_summary.judge_calib import Rater, decide, kappa_table
from call_summary.providers import Reply, ScriptedProvider
from call_summary.specs import make_spec


def test_kappa_known_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # 2x2 table a=8 (both yes), b=1, c=1, d=0: po 0.8, pe 0.9*0.9 + 0.1*0.1 = 0.82
    a = [True] * 8 + [True, False]
    b = [True] * 8 + [False, True]
    assert math.isclose(cohen_kappa(a, b), (0.8 - 0.82) / (1 - 0.82))
    assert math.isnan(cohen_kappa([True, True], [True, True]))  # no variation on either side
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


def _item(i: int, domain: str = "shop") -> Item:
    spec = make_spec("dev", domain, i)
    return Item(
        item_id=spec.spec_id,
        split="dev",
        domain=domain,
        spec=spec,
        turns=[{"speaker": "고객", "text": " ".join(sv.value for sv in spec.slots)}],
    )


def _runs(tmp_path: Path, n: int = 30, empty: dict[str, set[int]] | None = None) -> tuple[Path, dict]:
    items = [_item(i) for i in range(n)]
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict() for it in items])
    runs = {}
    for name in ("A", "B", "C"):
        run = tmp_path / f"run-{name}"
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
        skip = (empty or {}).get(name, set())
        write_jsonl(
            run / "items.jsonl",
            [
                {"item_id": it.item_id, "pred": {"summary": "" if k in skip else f"{name} 요약 {k}"}}
                for k, it in enumerate(items)
            ],
        )
        runs[name] = run
    return data, runs


def test_sample_draws_disjoint_strata_skips_empty_summaries_and_hides_the_source(tmp_path):
    data, runs = _runs(tmp_path, empty={"B": {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})
    hand = runs["C"] / "hand.jsonl"  # hand labels lie in the run they label
    hand_ids = [_item(i).item_id for i in (0, 1)]
    write_jsonl(hand, [{"item_id": i, "verdicts": ["포함"], "wrong_statements": 0} for i in hand_ids])
    sample = tmp_path / "out" / "sample.jsonl"
    template = tmp_path / "out" / "template.jsonl"
    args = [
        "sample",
        "--stratum",
        f"S1={runs['A']}:5",
        "--stratum",
        f"S2={runs['B']}:4",
        "--hand",
        f"H={hand}",
        "--seed",
        "0",
        "--out",
        str(sample),
        "--template",
        str(template),
    ]
    assert judge_calib.main(args) == 0
    rows = list(read_jsonl(sample))
    by = {s: [r for r in rows if r["stratum"] == s] for s in ("S1", "S2", "H")}
    assert [len(by[s]) for s in ("S1", "S2", "H")] == [5, 4, 2]
    new_ids = [r["item_id"] for r in by["S1"] + by["S2"]]
    assert len(set(new_ids)) == 9 and not set(new_ids) & set(hand_ids)
    assert all(r["summary"].strip() for r in by["S2"])  # empty summaries were skipped
    assert [r["item_id"] for r in by["H"]] == hand_ids
    tpl = list(read_jsonl(template))
    assert len(tpl) == 9  # hand rows are labelled already
    assert all(set(t) == {"key", "facts", "summary", "verdicts", "wrong_statements", "note"} for t in tpl)
    assert [t["key"] for t in tpl] == sorted(t["key"] for t in tpl)
    # the label order mixes the strata (keys are not handed out stratum by stratum)
    key_order = [next(r["stratum"] for r in rows if r["key"] == t["key"]) for t in tpl]
    assert key_order != sorted(key_order, key=["S1", "S2"].index)
    # the same call again gives the same sample
    sample2 = tmp_path / "out2" / "sample.jsonl"
    args2 = args[:-4] + ["--out", str(sample2), "--template", str(tmp_path / "out2" / "t.jsonl")]
    assert judge_calib.main(args2) == 0
    assert list(read_jsonl(sample2)) == rows


def _all_included(messages) -> str:
    text = messages[0]["content"]
    facts = text.split("[요약에 들어가야 할 사실]")[1].split("[채점할 요약]")[0].strip().splitlines()
    return json.dumps({"facts": ["포함"] * len(facts), "wrong_statements": 0}, ensure_ascii=False)


class _Counting(ScriptedProvider):
    def generate(self, messages, json_schema=None):
        r = super().generate(messages, json_schema)
        return Reply(text=r.text, latency_s=0.1, prompt_tokens=900, completion_tokens=20, done_reason="stop")


def test_judge_writes_one_row_per_key_with_token_counts_and_resumes(tmp_path, monkeypatch):
    data, runs = _runs(tmp_path, n=8)
    sample = tmp_path / "sample.jsonl"
    template = tmp_path / "template.jsonl"
    judge_calib.main(
        [
            "sample",
            "--stratum",
            f"S={runs['A']}:4",
            "--seed",
            "0",
            "--out",
            str(sample),
            "--template",
            str(template),
        ]
    )
    provider = _Counting(replies=_all_included)
    monkeypatch.setattr(judge_calib, "make_judge", lambda **kw: provider)
    monkeypatch.setattr(judge_calib, "model_fingerprint", lambda p: {"ollama_digest": "d"})
    monkeypatch.setattr(judge_calib, "loaded_models", lambda p: [])
    out = tmp_path / "judge-m.jsonl"
    argv = ["judge", "--sample", str(sample), "--model", "m", "--out", str(out), "--num-gpu", "99"]
    assert judge_calib.main(argv) == 0
    rows = list(read_jsonl(out))
    assert len(rows) == 4 and all(r["ok"] and r["prompt_tokens"] == 900 for r in rows)
    keys = {r["key"] for r in read_jsonl(sample)}
    assert {r["key"] for r in rows} == keys
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "m" and meta["num_gpu"] == 99 and meta["run_checks"]["context_full"] == 0
    assert len(provider.calls) == 4
    assert judge_calib.main(argv) == 0  # resume: nothing left to judge
    assert len(provider.calls) == 4 and len(list(read_jsonl(out))) == 4


def test_judge_reply_skips_the_model_for_an_empty_summary():
    it = _item(0)
    p = ScriptedProvider(replies=[])
    score, reply = judge_reply(p, it.item_id, it.spec, it.transcript, "  ")
    assert reply is None and score.verdicts == ("누락",) * len(it.spec.facts) and not p.calls


def _rater(name: str, items: list[tuple[str, list[str]]]) -> Rater:
    return Rater(name, {k: tuple(v) for k, v in items})


def test_kappa_table_counts_failed_judgements_as_included():
    claude = _rater("claude", [("k1", ["포함", "틀림"]), ("k2", ["포함", "포함"]), ("k3", ["누락", "포함"])])
    judge = _rater("j", [("k1", ["포함", "틀림"]), ("k2", []), ("k3", ["누락", "포함"])])
    t = kappa_table(claude, judge, ["k1", "k2", "k3"], n_boot=100)
    assert t["n_facts"] == 6 and t["n_errors"] == 2 and t["n_failed"] == 1
    assert t["error_recall"] == 1.0 and t["false_error_rate"] == 0.0 and t["kappa"] == 1.0


def _stats(kappa, lb_vs_j0, vs_h0, failed=0, full=0, recall=0.8, thinking=0):
    return {
        "kappa": kappa,
        "diff_vs_baseline": (kappa - 0.3, lb_vs_j0, kappa),
        "diff_vs_same_check": vs_h0,
        "n_failed": failed,
        "context_full": full,
        "thinking_items": thinking,
        "error_recall": recall,
    }


def test_decide_applies_every_condition_in_order():
    ok = _stats(0.7, 0.1, 0.05)
    assert decide({"H1": ok, "H2": _stats(0.65, 0.05, 0.0)})["adopted"] == "H1"
    assert decide({"H1": _stats(0.59, 0.1, 0.05)})["adopted"] is None  # kappa below 0.6
    assert decide({"H1": _stats(0.7, 0.0, 0.05)})["adopted"] is None  # lower bound not above 0
    assert decide({"H1": _stats(0.7, 0.1, -0.01)})["adopted"] is None  # worse than the old judge's hybrid
    assert decide({"H1": _stats(0.7, 0.1, 0.05, failed=2)})["adopted"] is None
    assert decide({"H1": _stats(0.7, 0.1, 0.05, full=1)})["adopted"] is None
    assert decide({"H1": _stats(0.7, 0.1, 0.05, thinking=1)})["adopted"] is None  # thinking was not off
    # tie on kappa: higher error recall, then the later-registered (other family) candidate
    tie = decide({"H1": _stats(0.7, 0.1, 0.0, recall=0.8), "H2": _stats(0.7, 0.1, 0.0, recall=0.9)})
    assert tie["adopted"] == "H2"
    tie2 = decide({"H1": _stats(0.7, 0.1, 0.0), "H2": _stats(0.7, 0.1, 0.0)})
    assert tie2["adopted"] == "H2"
    assert decide({"H1": _stats(float("nan"), 0.1, 0.0)})["adopted"] is None


def test_judge_run_writes_to_the_named_file_and_leaves_judge_jsonl_alone(tmp_path, monkeypatch):
    data, runs = _runs(tmp_path, n=3)
    run = runs["A"]
    (run / "judge.jsonl").write_text("keep\n", encoding="utf-8")
    provider = _Counting(replies=_all_included)
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return provider

    monkeypatch.setattr(judge_run, "make_judge", fake)
    monkeypatch.setattr(judge_run, "model_fingerprint", lambda p: {"ollama_digest": "d"})
    out = run / "judge-m.jsonl"
    argv = ["--run", str(run), "--model", "m", "--out", str(out), "--num-gpu", "99", "--think", "off"]
    argv += ["--option", "presence_penalty=0"]
    assert judge_run.main(argv) == 0
    assert (run / "judge.jsonl").read_text(encoding="utf-8") == "keep\n"
    assert len(list(read_jsonl(out))) == 3
    summary = json.loads((run / "judge-m_summary.json").read_text(encoding="utf-8"))
    assert summary["n"] == 3 and summary["options"] == {"presence_penalty": 0}
    assert seen["num_gpu"] == 99 and seen["think"] is False and seen["options"] == {"presence_penalty": 0}


def test_judge_run_defaults_match_the_recorded_j1_condition(monkeypatch, tmp_path):
    data, runs = _runs(tmp_path, n=1)
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return _Counting(replies=_all_included)

    monkeypatch.setattr(judge_run, "make_judge", fake)
    monkeypatch.setattr(judge_run, "model_fingerprint", lambda p: {})
    assert judge_run.main(["--run", str(runs["A"])]) == 0
    assert seen == {
        "model": "qwen2.5:14b-instruct",
        "num_gpu": 44,
        "think": False,
        "options": {},
    }
    assert (runs["A"] / "judge.jsonl").exists() and (runs["A"] / "judge_summary.json").exists()


def test_make_judge_builds_the_j1_request():
    p = judge_calib.make_judge(model="qwen2.5:14b-instruct", num_gpu=44, think=False, options={})
    assert (p.use_schema, p.num_ctx, p.num_predict, p.num_gpu, p.think, p.temperature, p.seed) == (
        True,
        4096,
        256,
        44,
        False,
        0.0,
        0,
    )
    assert p.extra_options in (None, {})


def _write_judge(path: Path, rows: list[dict], first_wrong: bool) -> None:
    write_jsonl(
        path,
        [
            {
                "key": r["key"],
                "verdicts": (["틀림"] if first_wrong else ["포함"]) + ["포함"] * (len(r["facts"]) - 1),
                "ok": True,
            }
            for r in rows
        ],
    )


def test_report_decides_on_the_decision_strata_with_the_value_check(tmp_path):
    data, runs = _runs(tmp_path, n=12)
    sample = tmp_path / "sample.jsonl"
    argv = ["sample", "--stratum", f"S={runs['A']}:6", "--stratum", f"C={runs['B']}:4", "--seed", "0"]
    assert judge_calib.main(argv + ["--out", str(sample), "--template", str(tmp_path / "t.jsonl")]) == 0
    rows = list(read_jsonl(sample))
    _write_judge(tmp_path / "claude.jsonl", rows, first_wrong=True)  # Claude: every first fact is wrong
    _write_judge(tmp_path / "j0.jsonl", rows, first_wrong=False)  # the old judge sees nothing
    _write_judge(tmp_path / "j1.jsonl", rows, first_wrong=True)  # the new one agrees with Claude
    out = tmp_path / "report.json"
    argv = ["report", "--sample", str(sample), "--labels", str(tmp_path / "claude.jsonl")]
    argv += ["--judge", f"J0={tmp_path / 'j0.jsonl'}", "--judge", f"J1={tmp_path / 'j1.jsonl'}"]
    argv += [
        "--baseline",
        "J0",
        "--candidates",
        "J1",
        "--decide-on",
        "S",
        "--n-boot",
        "200",
        "--json",
        str(out),
    ]
    assert judge_calib.main(argv) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert set(rep["tables"]) == {"S", "C", "decision:S", "all"}
    assert [t["rater"] for t in rep["tables"]["S"]] == ["V", "J0", "V+J0", "J1", "V+J1"]
    raw_j1 = next(t for t in rep["tables"]["S"] if t["rater"] == "J1")
    assert raw_j1["kappa"] == 1.0 and raw_j1["n_errors"] == 6 and raw_j1["error_recall"] == 1.0
    assert list(rep["candidates"]) == ["V+J1"]
    assert rep["decision"]["adopted"] in (None, "V+J1")


def test_report_reads_a_recorded_judge_by_item_for_one_stratum(tmp_path):
    data, runs = _runs(tmp_path, n=12)
    hand = runs["C"] / "hand.jsonl"
    write_jsonl(
        hand, [{"item_id": _item(i).item_id, "verdicts": ["포함"] * len(_item(i).spec.facts)} for i in (0, 1)]
    )
    sample = tmp_path / "sample.jsonl"
    argv = ["sample", "--stratum", f"S={runs['A']}:3", "--hand", f"H={hand}", "--seed", "0"]
    assert judge_calib.main(argv + ["--out", str(sample), "--template", str(tmp_path / "t.jsonl")]) == 0
    rows = list(read_jsonl(sample))
    _write_judge(tmp_path / "claude.jsonl", [r for r in rows if r["stratum"] == "S"], first_wrong=False)
    _write_judge(tmp_path / "j0.jsonl", rows, first_wrong=False)
    recorded = runs["C"] / "judge.jsonl"
    write_jsonl(
        recorded,
        [
            {"item_id": _item(i).item_id, "verdicts": ["포함"] * len(_item(i).spec.facts), "ok": True}
            for i in range(12)
        ],
    )
    rep = judge_calib.report(
        sample,
        tmp_path / "claude.jsonl",
        {"H": hand},
        {"J0": tmp_path / "j0.jsonl", "J0rec@H": recorded},
        "J0",
        [],
        ["S"],
        n_boot=20,
    )
    assert "J0rec" in [t["rater"] for t in rep["tables"]["H"]]
    assert "J0rec" not in [t["rater"] for t in rep["tables"]["S"]]
    assert rep["decision"]["adopted"] is None


def test_judge_run_creates_the_output_folder(tmp_path, monkeypatch):
    data, runs = _runs(tmp_path, n=2)
    monkeypatch.setattr(judge_run, "make_judge", lambda **kw: _Counting(replies=_all_included))
    monkeypatch.setattr(judge_run, "model_fingerprint", lambda p: {})
    out = tmp_path / "outputs" / "judge2" / "smoke-m.jsonl"
    assert judge_run.main(["--run", str(runs["A"]), "--limit", "1", "--out", str(out)]) == 0
    assert len(list(read_jsonl(out))) == 1 and (out.parent / "smoke-m_summary.json").exists()


def test_judge_run_summary_records_the_commit_and_the_loaded_models(tmp_path, monkeypatch):
    data, runs = _runs(tmp_path, n=2)
    ps = [{"name": "m", "size": 10, "size_vram": 10, "context_length": 4096}]
    monkeypatch.setattr(judge_run, "make_judge", lambda **kw: _Counting(replies=_all_included))
    monkeypatch.setattr(judge_run, "model_fingerprint", lambda p: {})
    monkeypatch.setattr(judge_run, "loaded_models", lambda p: ps)
    monkeypatch.setattr(judge_run, "_git", lambda *a: "abc123" if a[0] == "rev-parse" else "")
    out = runs["A"] / "judge-m.jsonl"
    assert judge_run.main(["--run", str(runs["A"]), "--out", str(out)]) == 0
    summary = json.loads((runs["A"] / "judge-m_summary.json").read_text(encoding="utf-8"))
    assert summary["git_commit"] == "abc123" and summary["git_dirty"] is False
    assert summary["loaded_models_after"] == ps
    assert summary["started"] <= summary["finished"]


def test_loaded_models_is_none_without_a_model_server():
    assert judge_run.loaded_models(object()) is None  # no /api/ps: a record only, never a failure


def test_value_cells_count_the_value_check_against_claude():
    claude = _rater("claude", [("k1", ["포함", "틀림", "누락"]), ("k2", ["포함", "누락"])])
    value_ok = {"k1": [False, True, None], "k2": [True, False]}
    c = judge_calib.value_cells(claude, value_ok, ["k1", "k2"])
    assert c["cells"] == {
        "found_incl": 1,
        "found_not": 1,
        "miss_incl": 1,
        "miss_not": 1,
        "noval_not": 1,
        "noval_incl": 0,
    }
    assert c["v_miss_but_claude_included"] == ["k1/F1"]  # V's false misses
    assert c["v_found_but_claude_not"] == ["k1/F2:틀림"]


def test_cells_reports_strata_then_named_groups_in_their_strata_order(tmp_path):
    data, runs = _runs(tmp_path, n=12)
    sample = tmp_path / "sample.jsonl"
    argv = ["sample", "--stratum", f"S1={runs['A']}:3", "--stratum", f"S2={runs['B']}:3", "--seed", "0"]
    assert judge_calib.main(argv + ["--out", str(sample), "--template", str(tmp_path / "t.jsonl")]) == 0
    rows = list(read_jsonl(sample))
    _write_judge(tmp_path / "claude.jsonl", rows, first_wrong=True)
    out = tmp_path / "cells.json"
    argv = ["cells", "--sample", str(sample), "--labels", str(tmp_path / "claude.jsonl")]
    argv += ["--group", "S=S2,S1", "--n-boot", "50", "--json", str(out)]
    assert judge_calib.main(argv) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    first_seen = list(dict.fromkeys(r["stratum"] for r in rows))
    assert list(rep) == first_seen + ["S"]
    assert rep["S"]["kappa_table"]["n_items"] == 6
    assert sum(rep["S"]["cells"].values()) == rep["S"]["kappa_table"]["n_facts"]
    # the group's bootstrap draws from its keys in the order of its strata (S2, then S1)
    claude = judge_calib._load_labels(rows, tmp_path / "claude.jsonl", {})
    value_ok = judge_calib._value_ok(rows)
    v = Rater("V", {k: judge_calib.hybrid_verdicts((), ok) for k, ok in value_ok.items()})
    keys = [r["key"] for r in rows if r["stratum"] == "S2"] + [r["key"] for r in rows if r["stratum"] == "S1"]
    want = kappa_table(claude, v, keys, n_boot=50)
    assert rep["S"]["kappa_table"]["kappa_ci"] == list(want["kappa_ci"])
    with pytest.raises(SystemExit):
        judge_calib.main(argv[:-4] + ["--group", "X=S1,nope"])
