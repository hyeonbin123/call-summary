# [테스트 묶음] 평가 실행끼리 표를 만드는 call_summary.compare를 검증한다: 실행별 표(run_table), 짝지은 차이
# 표(--paired), dev에서 고르는 규칙 순서(--pick), 생각 모드가 꺼졌는지 보는 표(--think-check). 판정 파일을
# 다시 읽는 judge_run.load_scores도 함께 본다.
# [왜 중요한가] 문서의 판정("넘음", "고른 설정")은 이 표의 숫자와 순서로 내린다. 고르는 규칙의 동률 처리나 NaN
# 처리가 어긋나면 사전 등록한 규칙과 다른 것을 고르게 된다.

import io
import json
import sys

# run_items·summary_table(evaluate)과 ScriptedProvider로 모델 없이 진짜 평가 실행과 같은 꼴의 폴더를 만든다.
from call_summary.compare import main, paired_table, pick_order, pick_table, run_table, think_rows
from call_summary.dataset import write_jsonl
from call_summary.evaluate import run_items, summary_table
from call_summary.judge_run import load_scores
from call_summary.prompts import record_to_json
from call_summary.providers import ScriptedProvider
from tests.test_scoring import _item


# 정해 둔 답을 내는 가짜 provider로 실행 폴더(manifest.json, items.jsonl, summary.json)를 만든다. compare가
# 읽는 파일이 evaluate가 쓰는 것과 같은 형식이 되게 하려는 것이다. 테스트를 빠르게 하려고 부트스트랩은 50번만
# 돈다.
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


# 모두 틀린 실행(형식 실패)과 모두 맞은 실행으로 표를 만든다. 실행별 표에 두 줄이 있고 구간이 [100, 100]으로
# 나오며, 짝지은 표의 exact 차이가 +100.0 [100.0, 100.0]인지 확인한다.
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


# 이 Windows 환경에서 출력을 파일로 보내면 표준 출력이 cp949라, 머리줄의 U+2212(−)를 쓰지 못해 명령이
# UnicodeEncodeError로 죽은 적이 있다(커밋 0ea5791). 짝지은 표는 cp949로 인코딩될 수 있어야 한다. 표준 출력을
# cp949 스트림으로 바꿔 --paired가 끝까지 돌고 ASCII 하이픈 머리줄을 쓰는지 확인한다.
def test_paired_output_on_a_cp949_console(tmp_path, monkeypatch):
    items = [_item(i, category=None) for i in range(4)]
    a = _write_run(tmp_path, "a", items, lambda m: "nope")
    b = _write_run(tmp_path, "b", items, lambda m: "nope")
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="cp949"))
    assert main(["--paired", str(a), str(b)]) == 0
    sys.stdout.flush()
    assert out.getvalue().decode("cp949").startswith("B - A: `b` - `a` (n=4)")


# 판정 파일 한 줄을 읽어 사실별 판정, 어긋난 문장 수, 사실 재현율(포함 수 / 사실 수 = 0.5)이 맞게 되살아나는지
# 본다.
def test_load_scores_roundtrip(tmp_path):
    p = tmp_path / "judge.jsonl"
    write_jsonl(p, [{"item_id": "x", "verdicts": ["포함", "누락"], "wrong_statements": 1, "ok": True}])
    s = load_scores(p)[0]
    assert s.verdicts == ("포함", "누락") and s.wrong_statements == 1 and s.recall == 0.5


# --pick의 순서 규칙(docs/experiments.md 2026 비교 대상 "dev에서 설정 고르기", 3단계 고르는 기준): exact가
# 높은 것 → 핵심 값 F1이 높은 것 → 환각이 낮은 것. 예측 값이 없어 환각이 NaN인 실행은 그 단계에서 맨 뒤로
# 가고, 완전히 같으면 넘긴 순서(등록 순서)를 지킨다(정렬이 안정 정렬이라). 표의 첫 줄이 1위 실행인지도 본다.
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
    e = run("e", 0.25, 0.75, float("nan"))  # 예측한 값이 없어 환각이 정의되지 않으므로 맨 뒤
    f = run("f", 0.25, 0.75, 0.04)  # d와 완전히 같으면 주어진 순서에서 앞선 쪽이 먼저
    assert pick_order([a, b, c, d, e, f]) == [d, f, c, e, b, a]
    table = pick_table([a, d])
    assert table.splitlines()[2].startswith("| 1 | d (`d`) | 25.0 | 75.0 | 4.0 |")


# --think-check가 실행마다 생각 글이 붙은 답, 답 안의 <think> 태그, num_predict에서 잘린 답(done_reason
# length), 스키마 통과 수, 답 토큰 p50·최대를 세는지 확인한다. qwen3.5처럼 생각 모드가 기본으로 켜진 모델에
# think off가 실제로 먹었는지 이 표로 판정했다(docs/experiments.md "생각 끄기 확인").
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
