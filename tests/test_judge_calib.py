# [테스트 묶음] 요약 판정 재보정(docs/experiments.md "요약 판정 재보정")의 도구를 모델 없이 검증한다.
# judge_calib의 sample(보정 표본 뽑기와 눈가림 채점 양식), judge(판정 모델 실행, 이어 하기), report(Claude
# 라벨과의 kappa 표와 채택 판정), decide(채택 규칙 (a)~(d)), cells(값 확인 V와 Claude 라벨의 셀 표)와, 판정
# 실행 judge_run·judge.make_judge·judge_reply·cohen_kappa를 다룬다.
# [왜 중요한가] 판정 모델을 바꿀지는 이 계산의 kappa와 구간으로 정했다(채택 V+J1, kappa 0.603). 규칙은 측정
# 전에 커밋했으므로 코드가 규칙과 같게 세어야 하고, 표본이 층끼리 겹치거나 채점 양식에 층·모델이 드러나면
# Claude 라벨이 눈가림이 아니게 된다. 또 기록된 j1 판정 파일(judge.jsonl)을 덮어쓰지 않아야 한다.

import json
import math
from pathlib import Path

import pytest

# ScriptedProvider·Reply로 판정 모델 답을 흉내 내고, make_spec으로 사실 목록이 있는 명세를 만든다.
from call_summary import judge_calib, judge_run
from call_summary.dataset import Item, read_jsonl, write_jsonl
from call_summary.judge import cohen_kappa, judge_reply
from call_summary.judge_calib import Rater, decide, kappa_table
from call_summary.providers import Reply, ScriptedProvider
from call_summary.specs import make_spec


# Cohen's kappa를 손 계산 값과 맞춘다: 완전 일치는 1, 2×2 표(a=8, b=1, c=1, d=0)는 (관찰 일치 0.8 − 우연 일치
# 0.82) / (1 − 0.82)로 음수가 된다. 양쪽 모두 한 값뿐이면 kappa가 정의되지 않아 NaN이고, 길이가 다르면 예외다.
# 거의 모든 사실이 '포함'인 데이터라 일치율 대신 kappa를 쓰는 이유가 이 계산에 있다(2단계: 일치율 96.3%인데
# kappa 약 0.32).
def test_kappa_known_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    # 2x2 table a=8 (both yes), b=1, c=1, d=0: po 0.8, pe 0.9*0.9 + 0.1*0.1 = 0.82
    a = [True] * 8 + [True, False]
    b = [True] * 8 + [False, True]
    assert math.isclose(cohen_kappa(a, b), (0.8 - 0.82) / (1 - 0.82))
    assert math.isnan(cohen_kappa([True, True], [True, True]))  # no variation on either side
    with pytest.raises(ValueError):
        cohen_kappa([True], [True, False])


# 명세의 슬롯 값을 한 발화에 모두 말하는 가짜 건(값 확인 V가 쓰는 명세·사실이 진짜와 같다).
def _item(i: int, domain: str = "shop") -> Item:
    spec = make_spec("dev", domain, i)
    return Item(
        item_id=spec.spec_id,
        split="dev",
        domain=domain,
        spec=spec,
        turns=[{"speaker": "고객", "text": " ".join(sv.value for sv in spec.slots)}],
    )


# 같은 dev 데이터를 가리키는 가짜 평가 실행 셋(A, B, C)을 만든다. 요약은 "<실행> 요약 <번호>"이고, empty로
# 고른 건은 요약을 비운다(표본 뽑기가 빈 요약을 건너뛰는지 보려고).
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


# sample 규칙(재보정 2): 손 채점 층(H)의 건을 뺀 건 번호를 seed로 한 번 섞고, 층마다 명령줄 순서로 다음 건을
# 가져가며 그 층 실행의 요약이 빈 건은 건너뛴다. 그래서 층끼리 건이 겹치지 않고 H와도 겹치지 않는다.
# 채점 양식에는 새 층의 행만 있고 열쇠·사실·요약·빈 판정 칸만 보인다(층·실행·모델이 없음). 열쇠 순서는 층을
# 섞는다(층별로 열쇠를 나눠 주지 않음). 같은 명령을 다시 부르면 같은 표본이 나온다(결정적).
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


# 판정 모델 대용 답: 판정 프롬프트에서 사실 줄 수를 세어 모두 '포함'으로 답한다.
def _all_included(messages) -> str:
    text = messages[0]["content"]
    facts = text.split("[요약에 들어가야 할 사실]")[1].split("[채점할 요약]")[0].strip().splitlines()
    return json.dumps({"facts": ["포함"] * len(facts), "wrong_statements": 0}, ensure_ascii=False)


# 답에 프롬프트·생성 토큰 수와 done_reason을 붙이는 가짜 provider. 판정 행에 토큰 수가 남는지,
# run_checks(컨텍스트를 채운 판정)가 계산되는지 보려고 쓴다.
class _Counting(ScriptedProvider):
    def generate(self, messages, json_schema=None):
        r = super().generate(messages, json_schema)
        return Reply(text=r.text, latency_s=0.1, prompt_tokens=900, completion_tokens=20, done_reason="stop")


# judge 하위 명령: 표본 열쇠마다 판정 한 줄(토큰 수 포함)과 meta 파일(모델, num_gpu, run_checks)을 쓰는지,
# 다시 부르면 이미 판정한 열쇠는 건너뛰어 모델을 더 부르지 않는지(이어 하기) 확인한다.
# make_judge·model_fingerprint·loaded_models는 monkeypatch로 바꿔 Ollama 없이 돈다.
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


# 요약이 비면 판정 모델을 부르지 않고 모든 사실을 '누락'으로 친다(재보정 3 공통 규칙).
def test_judge_reply_skips_the_model_for_an_empty_summary():
    it = _item(0)
    p = ScriptedProvider(replies=[])
    score, reply = judge_reply(p, it.item_id, it.spec, it.transcript, "  ")
    assert reply is None and score.verdicts == ("누락",) * len(it.spec.facts) and not p.calls


# 열쇠 → 사실별 판정 목록으로 채점자(Rater)를 만든다.
def _rater(name: str, items: list[tuple[str, list[str]]]) -> Rater:
    return Rater(name, {k: tuple(v) for k, v in items})


# 판정 실패(빈 판정)는 모든 사실을 '포함'으로 친다(규칙 4). 여기서는 Claude도 그 건을 모두 포함으로 봐
# 일치하므로 kappa 1이다. 오류 수(Claude 비포함 2), 실패 수 1, 오류 재현율 1, 거짓 오류율 0이 맞게 나오는지
# 본다.
def test_kappa_table_counts_failed_judgements_as_included():
    claude = _rater("claude", [("k1", ["포함", "틀림"]), ("k2", ["포함", "포함"]), ("k3", ["누락", "포함"])])
    judge = _rater("j", [("k1", ["포함", "틀림"]), ("k2", []), ("k3", ["누락", "포함"])])
    t = kappa_table(claude, judge, ["k1", "k2", "k3"], n_boot=100)
    assert t["n_facts"] == 6 and t["n_errors"] == 2 and t["n_failed"] == 1
    assert t["error_recall"] == 1.0 and t["false_error_rate"] == 0.0 and t["kappa"] == 1.0


# decide에 넘길 후보 하나의 통계를 만든다: kappa, J0 대비 짝지은 차이(점, 하한, 상한), 같은 값 확인을 붙인 J0
# 대비 차이, 실패·컨텍스트 채움·생각 글 수, 오류 재현율.
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


# 채택 규칙(재보정 4)을 하나씩 어겨 본다: (a) kappa ≥ 0.6, (b) J0 대비 하한 > 0, (c) V+J0보다 낮지 않음, (d)
# 채택 층 판정 실패 1건 이하·컨텍스트를 채운 판정 0·생각 글 0. 하나라도 어기면 채택하지 않는다.
# 둘 다 통과하면 kappa가 높은 것, 같으면 오류 재현율이 높은 것, 그래도 같으면 나중에 등록한 후보(J2, 학생과
# 다른 계열)다. kappa가 NaN이면 (a)를 통과하지 못한다.
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


# judge_run --out으로 새 판정 모델을 돌리면 그 파일과 <이름>_summary.json에만 쓰고 기록된 j1의 judge.jsonl은
# 건드리지 않는지 확인한다(재보정 5: j1 판정 파일을 덮어쓰지 않는다). --num-gpu, --think off, --option이
# make_judge에 그대로 넘어가고 summary에 옵션이 남는지도 본다.
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


# 인수 없이 부른 judge_run은 2단계 j1과 같은 조건이어야 한다: qwen2.5:14b-instruct, GPU 44층(14B가 VRAM을
# 넘치지 않게, docs/experiments.md 재보정 판정 모델 표: j1과 같음), 생각 끔, 더한 옵션 없음, 결과는
# judge.jsonl. 그래야 J0 재실행이 j1과 바이트 단위로 같은 요청이 된다.
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


# make_judge가 j1 요청 꼴(스키마 강제 JSON, 컨텍스트 4096, 답 256토큰, temperature 0, seed 0)을 만드는지
# 고정한다(재보정 3 공통: 같은 판정 프롬프트, 같은 요청 꼴).
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


# 표본 행마다 판정 파일을 쓴다. first_wrong이면 첫 사실만 '틀림', 나머지는 '포함'.
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


# report: 층마다·채택 층·전체 표에 V, J0, V+J0, J1, V+J1 순서로 행이 나오는지, Claude와 같게 판정한 J1이 kappa
# 1·오류 재현율 1인지, 후보 목록이 V+J1 하나인지 본다. 채택은 값 확인을 붙인 형태(V+J)로만 정한다(재보정 4).
# 채택 결과는 None과 V+J1 둘 다 허용한다.
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


# NAME@STRATUM 꼴(J0rec@H)은 그 층 실행의 기록된 판정 파일을 건 id로 읽어 그 층에만 붙인다. 2단계의 j1 판정과
# J0 재실행을 hand50에서 건별로 견주는 데 썼다(재보정 3). J0rec 행은 H 표에만 있고 S 표에는 없어야 한다.
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


# --out의 폴더가 없어도 만들어 쓴다(스모크 결과를 outputs/judge2/에 쓰는 경우).
def test_judge_run_creates_the_output_folder(tmp_path, monkeypatch):
    data, runs = _runs(tmp_path, n=2)
    monkeypatch.setattr(judge_run, "make_judge", lambda **kw: _Counting(replies=_all_included))
    monkeypatch.setattr(judge_run, "model_fingerprint", lambda p: {})
    out = tmp_path / "outputs" / "judge2" / "smoke-m.jsonl"
    assert judge_run.main(["--run", str(runs["A"]), "--limit", "1", "--out", str(out)]) == 0
    assert len(list(read_jsonl(out))) == 1 and (out.parent / "smoke-m_summary.json").exists()


# judge_run summary에 측정 코드의 커밋·작업 트리 상태·시작/끝 시각·끝난 뒤 /api/ps(모델이 GPU에 다 올라갔는지,
# 컨텍스트 길이)가 남는지 확인한다. 판정 실행이 어느 코드와 조건으로 돌았는지 기록에 남기려고 더했다(커밋
# 0103ac9).
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


# /api/ps를 부를 수 없으면 None을 기록하고 실행을 실패시키지 않는다(기록용 값이다).
def test_loaded_models_is_none_without_a_model_server():
    assert judge_run.loaded_models(object()) is None  # no /api/ps: a record only, never a failure


# value_cells: 사실마다 V의 결과(값 찾음 / 값 없음 / 숫자 값이 없는 사실)와 Claude의 포함 여부로 여섯 칸을
# 센다. V가 값이 없다고 했는데 Claude가 포함으로 본 사실(V의 거짓 누락)과 V가 찾았는데 Claude가 비포함으로 본
# 사실(V가 놓친 오류, 판정 종류 함께)을 "열쇠/F<번호>"로 나열한다.
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


# cells 하위 명령: 표본에 처음 나온 순서대로 층마다, 그다음 --group으로 묶은 이름(S=S2,S1)의 표를 낸다. 묶음의
# 부트스트랩은 적은 층 순서대로 열쇠를 이어 뽑으므로, 같은 순서로 직접 계산한 kappa 구간과 같아야 한다. 문서의
# 셀 표를 이 명령이 그대로 다시 내도록 커밋한 것이다(docs/experiments.md "값 확인 결과"의 2026-10-09
# 바로잡음). 없는 층 이름을 묶으면 멈춘다.
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
