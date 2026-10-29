# [테스트 묶음] 공개 문서의 수치를 실행 기록에 묶는 선언 파일(docs/numbers.toml)과 그 대조 도구 (`python -m
# call_summary.compare --check`)를 검증한다. (1) 지금 README의 선언한 수치가 모두 실행 기록과 맞는지 (2)
# 선언의 값 하나를 바꾸면 실패하는지 (3) context가 README의 그 절에 없으면 실패하는지 (4) 점 추정· 구간·짝지은
# 차이를 요약값이 아니라 건별 점수(items.jsonl)에서 같은 seed·횟수로 다시 내는지 (5) summary.json이 건별
# 점수와 어긋나면 알리는지 (6) 같은 숫자가 다른 곳에 있어서 통과하는 일을 막는지(그 절에 꼭 한 번, context는
# 선언한 값으로 끝남) (7) 선언 꼴이 틀리면 조용히 넘기지 않는지 (8) 명령줄 종료 코드와 cp949 콘솔.
# [왜 중요한가] README 결과 요약의 숫자는 기록에서 다시 낼 수 있어야 한다. 손으로 옮기면 반올림 실수가 생기고
# (docs/experiments.md 5단계 "다시 잰 결과"의 2026-10-03 덧붙임: 2.748초를 2.8초로 적음), 문장을 고치다 숫자가
# 바뀌어도 알아차리기 어렵다. 이 테스트가 CI에서 push마다 같은 대조를 돈다.

import functools
import io
import json
import sys
from pathlib import Path

import pytest

from call_summary.compare import check_numbers, load_numbers, main, number_tokens
from call_summary.dataset import write_jsonl
from call_summary.evaluate import summary_table
from call_summary.scoring import (
    ItemScore,
    bootstrap_ci,
    exact_rate,
    hallucination_rate,
    paired_bootstrap_diff,
)

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "docs" / "numbers.toml"
# README의 결과 요약 세 절. 선언 파일은 이 세 절을 모두 덮어야 한다.
RESULT_SECTIONS = {
    "결과 요약 (3단계)",
    "결과 요약 (4단계: 음성 인식을 거친 전사)",
    "결과 요약 (5단계: 서빙)",
}


def _failures(results) -> str:
    """실패한 선언을 사람이 읽을 수 있게 모은다(assert 메시지용)."""
    return "\n".join(f"{r.decl.get('context')!r}: {r.problems}" for r in results if not r.ok)


# ---------------------------------------------------------------------------
# 실제 선언 파일과 실제 실행 기록(reports/)
# ---------------------------------------------------------------------------


# 판정 1: 지금 README의 선언한 수치가 모두 통과한다. 결과 요약 세 절과 세 종류(point·ci·paired)를 모두
# 덮는지도 본다(선언 파일이 비거나 한 절만 남으면 검사가 아무것도 지키지 않게 되므로).
def test_every_declared_readme_number_matches_its_run_record():
    decls = load_numbers(NUMBERS)
    results = check_numbers(decls, root=ROOT)
    assert results and all(r.ok for r in results), _failures(results)
    assert RESULT_SECTIONS <= {d.get("section") for d in decls}
    assert {d["kind"] for d in decls} == {"point", "ci", "paired"}
    assert {d["where"] for d in decls} == {"README.md"}


# 선언이 가리키는 실행이 모두 reports/ 아래의 커밋된 기록인지 본다. outputs/(커밋하지 않는 폴더)의 실행을
# 가리키면 CI에서는 파일이 없어 실패하고, 다른 사람은 수치를 다시 낼 수 없다.
def test_declarations_point_at_committed_report_folders():
    for d in load_numbers(NUMBERS):
        for key in ("run", "base"):
            if key in d:
                assert d[key].startswith("reports/") and (ROOT / d[key] / "items.jsonl").is_file(), d


def _bump(value, places):
    """선언 값 하나를 그 자리수의 한 단위만큼 바꾼다(목록이면 마지막 값 = 구간의 상한)."""
    step = 10**-places
    if isinstance(value, list):
        return [*value[:-1], round(value[-1] + step, places)]
    return round(value + step, places)


# 판정 2: 실제 선언에서 값 하나를 바꾸면 실패한다. 종류마다 첫 선언 하나씩을 바꿔 본다. 다시 낸 값과 다르다는
# 문제와, context(README 글)가 그 값으로 끝나지 않는다는 문제가 함께 나와야 한다.
@pytest.mark.parametrize("kind", ["point", "ci", "paired"])
def test_changing_one_declared_value_fails(kind):
    decl = next(d for d in load_numbers(NUMBERS) if d["kind"] == kind)
    changed = {**decl, "value": _bump(decl["value"], decl.get("places", 1))}
    (result,) = check_numbers([changed], root=ROOT)
    assert not result.ok
    text = " ".join(result.problems)
    assert "recomputed" in text and "does not end with" in text


# 판정 3: context가 README에 없으면 실패한다. (a) context의 마지막 숫자를 다른 숫자로 바꾼 글은 README에 없다.
# (b) README에 있는 글이라도 선언한 절 밖에 있으면 실패한다(같은 숫자가 다른 곳에 있어서 통과하는 일을
# 막는다).
def test_context_missing_from_readme_or_from_its_section_fails():
    decl = next(d for d in load_numbers(NUMBERS) if d["kind"] == "ci")
    places = decl.get("places", 1)
    last = number_tokens(decl["context"])[-1]
    head, sep, tail = decl["context"].rpartition(last)
    assert sep, decl["context"]
    other = f"{float(last) + 10**-places:.{places}f}"
    moved = {**decl, "context": head + other + tail}
    (result,) = check_numbers([moved], root=ROOT)
    assert not result.ok and any("found 0 times" in p for p in result.problems), result.problems

    elsewhere = next(s for s in sorted(RESULT_SECTIONS) if s != decl["section"])
    (result,) = check_numbers([{**decl, "section": elsewhere}], root=ROOT)
    assert not result.ok and any("found 0 times" in p for p in result.problems), result.problems


# ---------------------------------------------------------------------------
# 가짜 실행 폴더(모델 없이 evaluate와 같은 꼴의 기록)
# ---------------------------------------------------------------------------


def _score(i: int, exact: bool, hallucinated: int = 0, prefix: str = "dev-shop") -> ItemScore:
    """건 하나의 채점 결과. 값 두 개를 예측했고 그중 hallucinated개가 전사에 없다."""
    return ItemScore(
        item_id=f"{prefix}-{i:05d}",
        domain="shop",
        gold_category="주문 취소",
        json_ok=True,
        schema_ok=True,
        pred_category="주문 취소",
        category_ok=True,
        resolution_ok=exact,
        entity_tp=2 - hallucinated,
        entity_fp=hallucinated,
        entity_fn=hallucinated,
        entity_pred=2,
        entity_hallucinated=hallucinated,
        action_tp=1,
        action_fp=0,
        action_fn=0,
        follow_required_ok=True,
        follow_codes_ok=True,
        off_list=0,
        exact=exact,
    )


@functools.cache
def _summary_text(scores: tuple[ItemScore, ...]) -> str:
    """evaluate.summary_table(부트스트랩 2,000번)의 JSON 글. 같은 점수 목록이면 테스트마다 다시 계산하지
    않는다(ItemScore는 frozen이라 해시할 수 있다)."""
    return json.dumps(summary_table(list(scores)))


def _write_run(root: Path, name: str, scores: list[ItemScore]) -> str:
    """evaluate가 쓰는 꼴(manifest.json, items.jsonl의 score, summary.json)로 실행 폴더를 만든다."""
    d = root / "reports" / name
    d.mkdir(parents=True)
    manifest = {"provider": name, "data": "datasets/dev.jsonl"}
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_jsonl(d / "items.jsonl", [{"item_id": s.item_id, "score": s.to_dict()} for s in scores])
    (d / "summary.json").write_text(_summary_text(tuple(scores)), encoding="utf-8")
    return f"reports/{name}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


# 두 실행: A는 10건 중 5건, B는 10건 중 8건이 맞았다. B의 환각은 예측한 값 20개 중 1개(5.0%).
A_EXACT = [True, False, True, False, True, False, True, False, True, False]
B_EXACT = [True, True, True, False, True, True, True, False, True, True]


@pytest.fixture
def fake(tmp_path):
    """가짜 실행 둘과 그 수치를 적은 README, 그 README에 대한 선언들. README의 구간·짝지은 차이는 scoring의
    기본값(seed 0, 2,000번)으로 낸 값이라, 검사가 같은 seed·횟수로 다시 내야 통과한다."""
    sa = [_score(i, e) for i, e in enumerate(A_EXACT)]
    sb = [_score(i, e, hallucinated=int(i == 3)) for i, e in enumerate(B_EXACT)]
    a = _write_run(tmp_path, "run-a", sa)
    b = _write_run(tmp_path, "run-b", sb)
    _, lo, hi = bootstrap_ci(sb, exact_rate)
    d, dlo, dhi = paired_bootstrap_diff(sa, sb, exact_rate)
    ci = f"**{_pct(exact_rate(sb))}** [{_pct(lo)}, {_pct(hi)}]"
    paired = f"+{_pct(d)}%p [{_pct(dlo)}, {_pct(dhi)}]"
    hall = _pct(hallucination_rate(sb))
    readme = "\n".join(
        [
            "# 제목",
            "",
            "## 결과",
            "",
            "| 모델 | dev |",
            "|---|---|",
            f"| A | {_pct(exact_rate(sa))} |",
            f"| B | {ci} |",
            "",
            f"- B는 A보다 {paired} 높고, 대화에 없는 값을 적는 비율은 {hall}%이다",
            "",
            "```bash",
            "# 코드 울타리 안의 '#' 줄은 제목이 아니다",
            "```",
            "",
            "- 울타리 뒤에도 같은 절이다: A는 50.0",
            "",
            "## 다른 절",
            "",
            "- 여기에도 A는 50.0이라고 적혀 있고, 반대로 보면 \u221230.0%p다",
            "",
        ]
    )
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    base = {"where": "README.md", "section": "결과", "metric": "exact"}
    decls = {
        "point": {**base, "context": "| A | 50.0 |", "kind": "point", "run": a, "value": 50.0},
        "ci": {
            **base,
            "context": f"| B | {ci} |",
            "kind": "ci",
            "run": b,
            "value": [float(_pct(exact_rate(sb))), float(_pct(lo)), float(_pct(hi))],
        },
        "paired": {
            **base,
            "context": f"A보다 {paired}",
            "kind": "paired",
            "run": b,
            "base": a,
            "value": [float(_pct(d)), float(_pct(dlo)), float(_pct(dhi))],
        },
        "paired_point": {
            **base,
            "context": f"A보다 +{_pct(d)}",
            "kind": "paired",
            "run": b,
            "base": a,
            "value": float(_pct(d)),
        },
        "hallucination": {
            **base,
            "context": f"비율은 {hall}",
            "kind": "point",
            "metric": "hallucination",
            "run": b,
            "value": float(hall),
        },
        "after_fence": {
            **base,
            "context": "울타리 뒤에도 같은 절이다: A는 50.0",
            "kind": "point",
            "run": a,
            "value": 50.0,
        },
        "minus": {
            **base,
            "section": "다른 절",
            "context": "반대로 보면 \u221230.0",
            "kind": "paired",
            "run": a,
            "base": b,
            "value": -30.0,
        },
    }
    return tmp_path, decls, (sa, sb)


# 점 추정·구간·짝지은 차이(차이만, 차이와 구간, 음수)·환각이 모두 건별 점수에서 다시 나와 통과한다. 코드
# 울타리 안의 "# ..." 줄을 제목으로 읽으면 울타리 뒤 글이 절 밖으로 밀려 after_fence 선언이 실패한다.
def test_point_ci_and_paired_numbers_are_recomputed_from_item_scores(fake):
    root, decls, _ = fake
    results = check_numbers(list(decls.values()), root=root)
    assert all(r.ok for r in results), _failures(results)
    got = {name: r.got for name, r in zip(decls, results, strict=True)}
    assert got["point"] == ["50.0"] and got["minus"] == ["-30.0"] and got["hallucination"] == ["5.0"]
    assert len(got["ci"]) == 3 and len(got["paired"]) == 3 and got["paired_point"] == ["30.0"]


# 다시 낸 값이 다르면(여기서는 실행 A를 가리켜야 할 선언이 B를 가리킴) 실패하고, 문제 글에 다시 낸 값과 선언
# 값이 함께 나온다.
def test_a_declaration_bound_to_the_wrong_run_fails(fake):
    root, decls, _ = fake
    wrong = {**decls["point"], "run": decls["ci"]["run"]}
    (result,) = check_numbers([wrong], root=root)
    assert not result.ok
    assert any("recomputed 80.0 != declared 50.0" in p for p in result.problems), result.problems


# summary.json(evaluate가 측정 때 쓴 요약)이 건별 점수와 어긋나면 알린다. 요약만 손으로 고치거나 다른 실행의
# 요약을 옮겨 놓은 경우를 잡는다. 구간(ci95)도 같다.
def test_summary_json_that_disagrees_with_item_scores_is_reported(fake):
    root, decls, _ = fake
    path = root / decls["ci"]["run"] / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["point"]["exact"] = 0.9
    summary["ci95"]["exact"] = [0.1, 0.2]
    path.write_text(json.dumps(summary), encoding="utf-8")
    (result,) = check_numbers([decls["ci"]], root=root)
    assert not result.ok
    assert sum("summary.json" in p for p in result.problems) == 2, result.problems


# 같은 숫자가 다른 곳에 있어서 통과하는 일을 막는다. (a) context가 절 안에 두 번 나오면 어느 자리인지 모르므로
# 실패 (b) 다른 절에만 있으면 실패 (c) 선언한 값이 context의 끝이 아니라 가운데 있으면 실패(context는 그
# 값으로 끝나야 한다: 구간까지 적힌 글에 차이 하나만 선언). 짧은 context라도 그 절에 한 번뿐이고 값으로 끝나면
# 통과한다.
def test_context_must_occur_once_in_its_section_and_end_with_the_value(fake):
    root, decls, _ = fake
    readme = root / "README.md"
    twice = readme.read_text(encoding="utf-8").replace("## 다른 절", "| A | 50.0 |\n\n## 다른 절")
    readme.write_text(twice, encoding="utf-8")
    (result,) = check_numbers([decls["point"]], root=root)
    assert any("found 2 times" in p for p in result.problems), result.problems

    (result,) = check_numbers([{**decls["point"], "context": "여기에도 A는 50.0"}], root=root)
    assert any("found 0 times" in p for p in result.problems), result.problems

    (result,) = check_numbers([{**decls["paired_point"], "context": decls["paired"]["context"]}], root=root)
    assert any("does not end with" in p for p in result.problems), result.problems

    (result,) = check_numbers([{**decls["after_fence"], "context": "A는 50.0"}], root=root)
    assert result.ok, result.problems


# 짝지은 차이는 양쪽이 같은 건이어야 한다(scoring.paired_bootstrap_diff의 조건). 건 id가 다르면 실패로 알린다.
def test_paired_numbers_need_the_same_items_on_both_sides(fake):
    root, decls, _ = fake
    other = _write_run(root, "run-c", [_score(i, True, prefix="test-a-shop") for i in range(10)])
    (result,) = check_numbers([{**decls["paired"], "run": other}], root=root)
    assert not result.ok and any("same item ids" in p for p in result.problems), result.problems


# 없는 실행 폴더·파일은 그 선언만 실패로 남기고 나머지 선언은 계속 검사한다(멈추지 않고 모두 보고).
def test_missing_files_fail_only_their_own_declaration(fake):
    root, decls, _ = fake
    gone = {**decls["point"], "run": "reports/no-such-run"}
    no_readme = {**decls["point"], "where": "NOPE.md"}
    results = check_numbers([gone, no_readme, decls["point"]], root=root)
    assert [r.ok for r in results] == [False, False, True]
    assert "FileNotFoundError" in " ".join(results[0].problems)
    assert "NOPE.md" in " ".join(results[1].problems)


# 글 속 숫자 읽기: 부호(ASCII 하이픈, 유니코드 빼기 U+2212, 더하기), 천 단위 쉼표, 한글이 바로 붙은 숫자는
# 값으로 읽고, 모델 이름·판 이름에 붙은 숫자("Qwen3-4B", "q8_0", "v2", "14B")는 값으로 읽지 않는다. "−0.0"은
# "0.0"이다.
def test_number_tokens_read_signs_separators_and_skip_names():
    assert number_tokens("\u22121.7%p [\u22125.8, +2.1]") == ["-1.7", "-5.8", "2.1"]
    assert number_tokens("+4.7%p [2.4, 7.2]") == ["4.7", "2.4", "7.2"]
    assert number_tokens("Qwen3-4B (1,500건 학습) 낮은 49.5이고 v2 q8_0") == ["1500", "49.5"]
    assert number_tokens("학습한 4B 0~1.5%, 14B 0~28%") == ["0", "1.5", "0", "28"]
    assert number_tokens("차이 \u22120.0, -0.00") == ["0.0", "0.00"]
    assert number_tokens("**95.7** [93.3, 97.7]") == ["95.7", "93.3", "97.7"]


# 선언 꼴이 틀리면 계산하기 전에 실패로 알린다(오타 난 키를 조용히 무시하면 그 조건이 검사되지 않는다).
# None은 그 키를 뺀다는 뜻이다.
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"contxt": "x"}, "unknown key"),
        ({"context": None}, "missing key"),
        ({"kind": "range"}, "kind must be"),
        ({"metric": "exactness"}, "unknown metric"),
        ({"base": "reports/run-a"}, "base is only for paired"),
        ({"value": [50.0, 40.0, 60.0]}, "point value must be one number"),
        ({"value": True}, "point value must be one number"),
        ({"value": 50.04}, "more decimal places"),
        ({"places": -1}, "places must be"),
        ({"places": True}, "places must be"),
        ({"section": 3}, "section must be"),
        ({"context": ""}, "context must be"),
    ],
)
def test_malformed_declarations_fail_before_any_recomputation(fake, change, message):
    root, decls, _ = fake
    decl = {k: v for k, v in {**decls["point"], **change}.items() if v is not None}
    (result,) = check_numbers([decl], root=root)
    assert not result.ok and any(message in p for p in result.problems), result.problems
    assert result.got == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"base": None}, "paired needs base"),
        ({"value": [1.0, 2.0]}, "paired value must be"),
        ({"kind": "ci", "value": 80.0}, "ci value must be"),
    ],
)
def test_malformed_paired_and_ci_declarations_fail(fake, change, message):
    root, decls, _ = fake
    source = decls["ci"] if change.get("kind") == "ci" else decls["paired"]
    decl = {k: v for k, v in {**source, **change}.items() if v is not None}
    (result,) = check_numbers([decl], root=root)
    assert not result.ok and any(message in p for p in result.problems), result.problems


def _toml(decls: list[dict]) -> str:
    """선언 목록을 TOML 글로 적는다(JSON 문자열·숫자·배열은 TOML 값으로도 맞는 꼴이다)."""
    out = []
    for d in decls:
        out.append("[[number]]")
        out += [f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in d.items()]
        out.append("")
    return "\n".join(out)


# 선언 파일은 [[number]] 표의 목록만 받는다. 다른 꼭대기 키가 있으면(예: [[numbers]] 오타) 읽을 때 멈춘다.
def test_load_numbers_reads_tables_and_rejects_other_top_level_keys(fake, tmp_path):
    root, decls, _ = fake
    path = tmp_path / "numbers.toml"
    path.write_text(_toml([decls["point"], decls["paired"]]), encoding="utf-8")
    assert load_numbers(path) == [decls["point"], decls["paired"]]
    path.write_text(_toml([decls["point"]]).replace("[[number]]", "[[numbers]]"), encoding="utf-8")
    with pytest.raises(ValueError, match="numbers"):
        load_numbers(path)


def _cli(monkeypatch, args: list[str]) -> tuple[int, str]:
    """표준 출력을 cp949 스트림으로 바꿔 명령줄을 돌린다(이 Windows 환경에서 출력을 파일로 돌릴 때와 같다)."""
    out = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(out, encoding="cp949"))
    code = main(args)
    sys.stdout.flush()
    return code, out.getvalue().decode("cp949")


# 명령줄: 모두 맞으면 종료 코드 0, 하나라도 어긋나면 1이고 그 선언에 FAIL과 문제를 찍는다. 출력을 파일로
# 돌리면 표준 출력이 cp949인데 cp949에는 유니코드 빼기(U+2212)가 없다(커밋 0ea5791에서 짝지은 표가 같은 이유로
# 죽었다). context에 U+2212가 있어도 이스케이프해 끝까지 찍어야 한다. 경로는 저장소 뿌리(현재 폴더) 기준이다.
def test_cli_exit_code_and_output_on_a_cp949_console(fake, monkeypatch):
    root, decls, _ = fake
    monkeypatch.chdir(root)
    good = root / "good.toml"
    good.write_text(_toml([decls["point"], decls["minus"]]), encoding="utf-8")
    code, text = _cli(monkeypatch, ["--check", str(good)])
    assert code == 0, text
    assert "\\u2212" in text and "2 numbers checked, 0 failed" in text

    bad = root / "bad.toml"
    bad.write_text(_toml([decls["point"], {**decls["point"], "value": 60.0}]), encoding="utf-8")
    code, text = _cli(monkeypatch, ["--check", str(bad)])
    assert code == 1
    assert (
        "FAIL" in text
        and "recomputed 50.0 != declared 60.0" in text
        and "2 numbers checked, 1 failed" in text
    )
