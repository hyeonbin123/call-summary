"""평가 실행들로 표를 만든다.

compare RUN...            실행마다 한 줄: 대표 지표와 95% 구간, 지연
compare --paired A B      지표 차이 B - A와 짝지은 부트스트랩 구간(양쪽이 같은 건)
compare --pick RUN...     고르는 규칙 순서로 줄 세운 실행: exact, 다음 핵심 값 F1, 다음 낮은 환각
compare --think-check RUN...  실행마다: 생각 글, <think> 태그, num_predict에서 잘린 답, 답 토큰 수
compare --check FILE      수치 선언 파일(docs/numbers.toml)의 숫자를 실행 기록에서 다시 내 공개 문서와
                          대조한다. 경로는 현재 폴더(저장소 뿌리) 기준, 하나라도 어긋나면 종료 코드 1
"""

# [파일 역할] 기록된 평가 실행(outputs/runs/<id>/ 또는 reports/<id>/)을 읽어 docs/experiments.md에 붙일 표를
# 만드는 모델 없는 도구다. 실행마다 지표와 95% 구간을 나란히 보여 주는 표, 두 실행의 짝지은 차이 표, dev에서
# 고르는 규칙대로 실행을 줄 세운 표, 생각 끄기 확인 표를 낸다. --check는 거꾸로, 공개 문서(README)에 이미
# 적힌 숫자를 실행 기록에서 다시 내 맞는지 대조한다(아래 "--check" 절).
# [왜 필요한가] 이 프로젝트는 판정을 "같은 건끼리 짝지은 부트스트랩 95% 구간의 하한이 0보다 큰가"로
# 내린다(docs/experiments.md 공통). 결과 표와 판정을 손으로 계산하지 않고 실행 기록(manifest.json,
# summary.json, items.jsonl)에서 다시 계산하게 해, 누구나 같은 숫자를 다시 낼 수 있게 한다.
# [연결] evaluate.py가 쓴 실행 폴더를 읽고, 지표 계산은 scoring.py(METRICS, bootstrap_ci,
# paired_bootstrap_diff)를 그대로 쓴다. 선언 파일은 표준 라이브러리 tomllib로 읽는다(PyYAML은 의존성에 없다).

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# HEADLINE: 표에 기본으로 넣는 지표 목록(evaluate.py와 같은 것을 써서 실행 요약과 표의 열이 어긋나지 않게
# 한다).
# METRICS: 지표 이름 → 건별 점수 목록에서 값을 계산하는 함수. ItemScore: 건 하나의 채점 결과.
# bootstrap_ci: 한 실행의 지표 구간(evaluate가 summary.json에 적는 것과 같은 계산).
# paired_bootstrap_diff: 같은 건 id를 양쪽에서 함께 다시 뽑아 B − A 차이의 구간을 낸다.
from .dataset import read_jsonl
from .evaluate import HEADLINE
from .scoring import METRICS, ItemScore, bootstrap_ci, paired_bootstrap_diff


# 실행 폴더 하나에서 manifest(실행 조건)와 건별 채점 결과(items.jsonl의 score)를 읽는다. 짝지은 비교는
# 요약값이 아니라 이 건별 점수로 다시 계산한다.
def load_run(run: str | Path) -> tuple[dict, list[ItemScore]]:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    scores = [ItemScore(**d["score"]) for d in read_jsonl(run / "items.jsonl")]
    return manifest, scores


# 표에 쓸 실행 이름: 모델(provider) 이름에 shot 수와 p1이 아닌 프롬프트 판을 붙인다(예: "qwen2.5:14b p2", "...
# 2-shot"). 기본값(0-shot, p1)은 생략해 이름을 짧게 한다.
def run_name(manifest: dict) -> str:
    shots = manifest.get("shots") or 0
    prompt = manifest.get("prompt_version", "p1")
    return (
        manifest["provider"] + (f" {shots}-shot" if shots else "") + (f" {prompt}" if prompt != "p1" else "")
    )


# 숫자를 표의 칸 글자로 바꾼다. 비율은 백분율 소수 한 자리, 초 단위 값(pct=False)은 소수 두 자리. None이나
# NaN(예: 핵심 값을 하나도 내지 않아 정의되지 않는 정밀도)은 "-"로 적는다.
def fmt(x: float, pct: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.1f}" if pct else f"{x:.2f}"


# 실행마다 한 줄: 건수, 지표마다 "점 추정 [95% 하한, 상한]", 지연 p50(초). 값은 evaluate가 미리 계산해 둔
# summary.json에서 읽는다. 구간이 없는 지표는 NaN으로 두어 "-"로 보인다.
def run_table(runs: list[str], metrics: tuple[str, ...] = HEADLINE) -> str:
    head = "| run | n | " + " | ".join(metrics) + " | p50 s |"
    lines = [head, "|" + "---|" * (len(metrics) + 3)]
    for r in runs:
        manifest, _ = load_run(r)
        summary = json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))
        cells = []
        for m in metrics:
            p = summary["point"][m]
            lo, hi = summary["ci95"].get(m, [float("nan"), float("nan")])
            cells.append(f"{fmt(p)} [{fmt(lo)}, {fmt(hi)}]")
        lat = summary.get("latency_s", {}).get("p50")
        lines.append(
            f"| {run_name(manifest)} | {summary['n']} | " + " | ".join(cells) + f" | {fmt(lat, False)} |"
        )
    return "\n".join(lines)


# 두 실행 A, B의 지표마다 A 값, B 값, B − A 차이와 짝지은 부트스트랩 95% 구간(기본 2,000번)을 표로 낸다. 같은
# 데이터의 같은 건이 양쪽에 있어야 하며, 맞지 않으면 paired_bootstrap_diff가 오류를 낸다.
# 판정 문장("하한 > 0이면 넘음" 등)은 이 표의 diff 구간으로 내린다(예: 2026년 비교 대상 판정 `compare --paired
# <비교 대상> <학생>`).
def paired_table(a: str, b: str, metrics: tuple[str, ...] = HEADLINE, n_boot: int = 2000) -> str:
    ma, sa = load_run(a)
    mb, sb = load_run(b)
    lines = [
        # 제목 줄의 빼기 기호는 ASCII "-"로 쓴다. Windows 한국어 환경에서 출력을 파일로 돌리면 표준 출력이
        # cp949인데 cp949에는 유니코드 빼기(U+2212)가 없다. 돌린 출력도 동작하게 고친 것이다(커밋 0ea5791).
        f"B - A: `{run_name(mb)}` - `{run_name(ma)}` (n={len(sa)})",
        "",
        "| metric | A | B | diff [95% CI] |",
        "|---|---|---|---|",
    ]
    for m in metrics:
        d, lo, hi = paired_bootstrap_diff(sa, sb, METRICS[m], n_boot=n_boot)
        lines.append(
            f"| {m} | {fmt(METRICS[m](sa))} | {fmt(METRICS[m](sb))} | {fmt(d)} [{fmt(lo)}, {fmt(hi)}] |"
        )
    return "\n".join(lines)


# 고르는 규칙의 정렬 키: exact 높은 순 → 핵심 값 F1 높은 순 → 환각 낮은 순. 높은 것이 앞에 오도록 앞의 둘은
# 부호를 뒤집는다.
# 값이 정의되지 않으면(핵심 값을 하나도 내지 않은 실행 등) 가장 나쁜 값으로 쳐 맨 뒤로 보낸다.
def _pick_key(point: dict) -> tuple[float, float, float]:
    def num(x: float | None, worst: float) -> float:
        return worst if x is None or (isinstance(x, float) and math.isnan(x)) else x

    # 정의되지 않은 값(예측한 핵심 값이 없음)은 맨 뒤로 간다.
    return (
        -num(point.get("exact"), -math.inf),
        -num(point.get("entity_f1"), -math.inf),
        num(point.get("hallucination"), math.inf),
    )


# dev 결과로 후보를 고르는 규칙의 순서대로 실행을 줄 세운다(첫 번째가 고른 것). 예: 2026-10-05 비교 대상 8개
# 설정에서 C*를 고를 때 등록 순서로 넘겼다(docs/experiments.md).
# sorted는 안정 정렬이라 세 기준이 모두 같으면 넘긴 순서가 유지된다. 그래서 규칙의 마지막 동률 기준("등록
# 순서가 앞선 것")을 맞추려면 등록 순서대로 넘겨야 한다.
def pick_order(runs: list[str]) -> list[str]:
    """dev exact, 다음 핵심 값 F1, 다음 낮은 환각 순으로 좋은 실행부터 줄 세운다. 모두 같으면 넘긴
    순서를 유지한다(실행을 등록 순서대로 넘긴다)."""
    points = {r: json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))["point"] for r in runs}
    return sorted(runs, key=lambda r: _pick_key(points[r]))


# pick_order의 결과를 순위, 실행 이름(폴더 이름), exact, 핵심 값 F1, 환각으로 표에 적는다.
def pick_table(runs: list[str]) -> str:
    lines = ["| rank | run | exact | entity_f1 | hallucination |", "|---|---|---|---|---|"]
    for i, r in enumerate(pick_order(runs), 1):
        manifest = json.loads((Path(r) / "manifest.json").read_text(encoding="utf-8"))
        point = json.loads((Path(r) / "summary.json").read_text(encoding="utf-8"))["point"]
        cells = " | ".join(fmt(point.get(m)) for m in ("exact", "entity_f1", "hallucination"))
        lines.append(f"| {i} | {run_name(manifest)} (`{Path(r).name}`) | {cells} |")
    return "\n".join(lines)


# 생각 모드를 끈 설정이 실제로 꺼졌는지 실행마다 확인할 값들을 모은다(2026-10-05 비교 대상의 "생각 끄기 확인"
# 단계용 도구, docs/experiments.md).
# 세는 것: 생각 글(Ollama message.thinking)이 붙어 온 답, 답 안의 <think> 태그, num_predict에서 잘린
# 답(done_reason "length"), 스키마 통과 수, 답 토큰 수의 p50·최대. 생각 글이 남으면 답 상한(1,024토큰)에서
# 잘려 JSON이 사라지므로, 이 값들로 측정 조건이 지켜졌는지 판단한다.
def think_rows(runs: list[str]) -> list[dict]:
    """실행마다 생각 끄기가 지켜졌는지 보여 주는 값: 생각 글(Ollama의 message.thinking)이 붙어 온 답,
    답 안에 <think> 태그가 있는 답, num_predict에서 잘린 답, 답 토큰 수(evaluate와 같은 p50, 최대)."""
    out = []
    for r in runs:
        manifest = json.loads((Path(r) / "manifest.json").read_text(encoding="utf-8"))
        rows = list(read_jsonl(Path(r) / "items.jsonl"))
        # 답 토큰 수는 백엔드가 알려 준 건만 쓴다. p50은 정렬한 목록의 가운데 원소다(evaluate와 같은 방식).
        toks = sorted(x["completion_tokens"] for x in rows if x.get("completion_tokens") is not None)
        out.append(
            {
                "run": Path(r).name,
                "think": manifest.get("think"),
                "ollama_options": manifest.get("ollama_options"),
                "n": len(rows),
                "thinking_items": sum(1 for x in rows if x.get("thinking_chars")),
                "think_tags": sum(1 for x in rows if "<think>" in x["reply"] or "</think>" in x["reply"]),
                "length_stops": sum(1 for x in rows if x.get("done_reason") == "length"),
                "schema_ok": sum(1 for x in rows if x["score"]["schema_ok"]),
                "completion_p50": toks[len(toks) // 2] if toks else None,
                "completion_max": toks[-1] if toks else None,
            }
        )
    return out


# ---------------------------------------------------------------------------
# --check: 공개 문서에 적힌 숫자를 실행 기록에 대조한다
# ---------------------------------------------------------------------------
# [무엇] 선언 파일(docs/numbers.toml)의 [[number]] 표 하나가 공개 문서의 숫자 하나(또는 점 추정과 구간 한
# 벌)를 실행 기록 하나(짝지은 차이면 둘)에 묶는다. 선언마다 (1) 문서의 그 절에 context 글이 꼭 한 번 있고 그
# 글이 선언한 값으로 끝나는지 (2) 실행 기록의 건별 점수(items.jsonl)에서 다시 낸 값이 선언한 소수 자리까지
# 같은지 (3) 측정 때 evaluate가 쓴 요약(summary.json)도 같은 값인지를 본다.
# [왜] README의 숫자는 실행 기록에서 손으로 옮긴 것이라 반올림 실수가 생길 수 있고(docs/experiments.md 5단계
# "다시 잰 결과"의 2026-10-03 덧붙임: 2.748초를 2.8초로 적음), 문장을 고치다 숫자가 바뀌어도 알아차리기
# 어렵다. 숫자만 찾으면 같은 숫자가 문서의 다른 곳에 있어도 통과하므로, 숫자를 둘러싼 글(context)을 선언한 절
# 안에서 찾고 그 글이 꼭 한 번 나오게 한다.
# [다시 내는 방법] 점 추정은 scoring.METRICS, 구간은 bootstrap_ci, 짝지은 차이는 paired_bootstrap_diff를
# evaluate(summary.json의 ci95)·--paired와 같은 seed 0·2,000번으로 부른다. 그래서 기록된 구간이 그대로 다시
# 나온다. 지표는 모두 비율이라 100을 곱해 백분율(차이는 %p)로 비교한다.
# [범위] 지금 선언하는 것은 README 머리말과 결과 요약의 평가 실행 수치뿐이다. 지연·효율(bench, loadtest),
# 요약 판정(judge_run, judge_calib, value_facts), 번호 표시(verify), 음성 조건 입력 분석(asr_report,
# asr_arms)의 수치는 이 검사로 다시 내지 않는다.

# 선언의 종류. point = 점 추정 하나, ci = [점 추정, 95% 하한, 상한], paired = 짝지은 차이 run − base 하나
# 또는 [차이, 하한, 상한].
NUMBER_KINDS = ("point", "ci", "paired")
_NUMBER_REQUIRED = ("where", "context", "kind", "metric", "run", "value")
_NUMBER_OPTIONAL = ("section", "places", "base")
# 구간을 다시 낼 때의 부트스트랩 횟수와 seed. evaluate.summary_table과 paired_table의 기본값(scoring 함수의
# 기본값)과 같아야 기록된 구간이 그대로 나온다.
CHECK_N_BOOT = 2000
CHECK_SEED = 0

# 글 속의 숫자 하나. 앞에 부호(ASCII 하이픈, 유니코드 빼기 U+2212, 더하기)가 붙을 수 있고 천 단위 쉼표와
# 소수점을 받는다. 영문자·숫자·밑줄·점 바로 뒤의 숫자(모델 이름 "Qwen3"·"q8_0", 판 "v2")와 영문자·숫자 바로
# 앞의 숫자("4B", "14B", "1.5b")는 값이 아니라 이름의 일부라 읽지 않는다. 뒤의 "\.\d" 막기는 "1.5b"가 되돌아가
# "1"만 숫자로 잡히는 것을 막는다. 한글은 막지 않는다: "49.5이고"처럼 한국어 문장에서는 조사가 숫자에 바로
# 붙는다.
_NUMBER = re.compile(
    r"(?<![0-9A-Za-z_.])([+\-\u2212]?)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![0-9A-Za-z]|\.\d)"
)
# Markdown 제목 줄(# 1~6개 + 공백 + 제목, 끝의 닫는 #은 뺀다). 코드 울타리(```, ~~~) 안의 줄은 제목이 아니다
# (README "실행" 절의 bash 주석 "# 명세 만들기 ..."가 그런 줄이다).
_HEADING = re.compile(r"(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")


def number_tokens(text: str) -> list[str]:
    """글 속 숫자들을 나온 순서대로 정규화한 글로 돌려준다: 빼기 기호는 "-", 더하기 기호와 천 단위 쉼표는
    없앰, 0의 부호는 뺌("−0.0" → "0.0"). 소수 자리 수는 글에 적힌 그대로 둔다("0"과 "0.0"은 다르다)."""
    out = []
    for m in _NUMBER.finditer(text):
        digits = m.group(2).replace(",", "")
        negative = m.group(1) in ("-", "\u2212") and float(digits) != 0
        out.append(("-" if negative else "") + digits)
    return out


def _fixed(x: float, places: int) -> str:
    """number_tokens와 같은 꼴의 글로 반올림한다(f-string 반올림 = compare.fmt와 같은 방식, 0의 부호는 뺌)."""
    s = f"{x:.{places}f}"
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _show(vals: list[str]) -> str:
    """값 하나는 그대로, 셋은 "점 [하한, 상한]" 꼴로 적는다(문제 글과 명령줄 출력용)."""
    return f"{vals[0]} [{vals[1]}, {vals[2]}]" if len(vals) == 3 else " ".join(vals)


def _value_list(value) -> list[float] | None:
    """선언 값(숫자 하나 또는 숫자 목록)을 float 목록으로 바꾼다. 숫자가 아닌 것(bool 포함)·무한·NaN이 있으면
    None."""
    items = value if isinstance(value, list) else [value]
    ok = all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v) for v in items)
    return [float(v) for v in items] if ok else None


def load_numbers(path: str | Path) -> list[dict]:
    """수치 선언 파일(TOML)을 읽어 [[number]] 표의 목록을 돌려준다. 다른 꼭대기 키가 있으면(오타 등) 그 키의
    선언이 조용히 빠지지 않게 ValueError로 멈춘다."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    extra = sorted(set(data) - {"number"})
    if extra:
        raise ValueError(
            f"{path}: unknown top-level key(s) {', '.join(extra)}; declarations are [[number]] tables"
        )
    numbers = data.get("number", [])
    if not isinstance(numbers, list) or not all(isinstance(d, dict) for d in numbers):
        raise ValueError(f"{path}: 'number' must be an array of tables ([[number]])")
    return numbers


def _decl_problems(d: dict) -> list[str]:
    """선언 하나의 꼴을 검사한다(계산 전에). 빠진 키·모르는 키·종류·지표·base·값 꼴·소수 자리를 본다."""
    problems = []
    missing = [k for k in _NUMBER_REQUIRED if k not in d]
    unknown = sorted(set(d) - {*_NUMBER_REQUIRED, *_NUMBER_OPTIONAL})
    if missing:
        problems.append(f"missing key(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown key(s): {', '.join(unknown)}")
    for key in ("where", "context", "run", "section", "base"):
        if key in d and not (isinstance(d[key], str) and d[key]):
            problems.append(f"{key} must be a non-empty string")
    kind = d.get("kind")
    if "kind" in d and kind not in NUMBER_KINDS:
        problems.append(f"kind must be one of {', '.join(NUMBER_KINDS)}")
    if "metric" in d and d["metric"] not in METRICS:
        problems.append(f"unknown metric {d['metric']!r} (one of {', '.join(METRICS)})")
    if kind == "paired" and "base" not in d:
        problems.append("paired needs base (the run that is subtracted: value = run - base)")
    if kind in ("point", "ci") and "base" in d:
        problems.append("base is only for paired")
    places = d.get("places", 1)
    if isinstance(places, bool) or not isinstance(places, int) or places < 0:
        problems.append("places must be an integer >= 0")
        return problems
    if "value" not in d or kind not in NUMBER_KINDS:
        return problems
    vals = _value_list(d["value"])
    many = isinstance(d["value"], list)
    if kind == "point" and (vals is None or many):
        problems.append("point value must be one number")
    elif kind == "ci" and (vals is None or not many or len(vals) != 3):
        problems.append("ci value must be [point, low, high]")
    elif kind == "paired" and (vals is None or (many and len(vals) != 3)):
        problems.append("paired value must be one number (run - base) or [difference, low, high]")
    elif any(float(_fixed(v, places)) != v for v in vals):
        problems.append(f"value {d['value']} has more decimal places than places={places}")
    return problems


def _section_body(text: str, section: str | None) -> tuple[str | None, int]:
    """section 제목 아래의 글(다음 같은 수준 이상의 제목 전까지)과 그 제목이 나온 수. section이 None이면 파일
    전체. 제목이 없거나 둘 이상이면 (None, 나온 수)."""
    lines = text.splitlines()
    if section is None:
        return "\n".join(lines), 1
    heads: list[tuple[int, int, str]] = []
    fence = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if fence:
            if stripped.startswith(fence):
                fence = None
            continue
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            continue
        if m := _HEADING.match(line):
            heads.append((i, len(m.group(1)), m.group(2)))
    hits = [h for h in heads if h[2] == section]
    if len(hits) != 1:
        return None, len(hits)
    start, level, _ = hits[0]
    end = next((i for i, lv, _ in heads if i > start and lv <= level), len(lines))
    return "\n".join(lines[start + 1 : end]), 1


def _context_problems(decl: dict, want: list[str], root: Path, texts: dict) -> list[str]:
    """문서 쪽 검사: 선언한 절에 context가 꼭 한 번 있고, context의 마지막 숫자(들)가 선언한 값과 같다."""
    where, section = decl["where"], decl.get("section")
    if where not in texts:
        try:
            texts[where] = (root / where).read_text(encoding="utf-8")
        except OSError as e:
            texts[where] = e
    text = texts[where]
    if isinstance(text, OSError):
        return [f"cannot read {where}: {type(text).__name__}: {text}"]
    body, heads = _section_body(text, section)
    if body is None:
        return [f"section {section!r} found {heads} times in {where} (must be exactly once)"]
    place = where + (f" section {section!r}" if section else "")
    problems = []
    count = body.count(decl["context"])
    if count != 1:
        problems.append(f"context found {count} times in {place} (must be exactly once)")
    tail = number_tokens(decl["context"])[-len(want) :]
    if tail != want:
        problems.append(
            f"context does not end with the declared value(s) {_show(want)}; "
            f"its last number(s): {' '.join(tail) or 'none'}"
        )
    return problems


def _run_record(root: Path, run: str, cache: dict) -> tuple[list[ItemScore], dict]:
    """실행 폴더의 건별 점수와 summary.json(한 번 읽은 것은 다시 읽지 않는다)."""
    if run not in cache:
        _, scores = load_run(root / run)
        summary = json.loads((root / run / "summary.json").read_text(encoding="utf-8"))
        cache[run] = (scores, summary)
    return cache[run]


def _summary_problems(run: str, summary: dict, name: str, point: float, places: int) -> list[str]:
    """summary.json의 점 추정이 건별 점수에서 다시 낸 값과 같은 자리까지 같은지(그 지표가 없으면 넘어감)."""
    recorded = summary.get("point", {}).get(name)
    if recorded is None or _fixed(100 * recorded, places) == _fixed(100 * point, places):
        return []
    return [
        f"summary.json of {run} says {name} {_fixed(100 * recorded, places)} "
        f"but items.jsonl gives {_fixed(100 * point, places)}"
    ]


def _recompute(decl: dict, root: Path, runs: dict, places: int) -> tuple[list[float], list[str]]:
    """선언의 값을 실행 기록에서 다시 낸다(비율 그대로, 100을 곱하기 전). 둘째 값은 summary.json과 어긋난
    점들."""
    name, kind = decl["metric"], decl["kind"]
    metric = METRICS[name]
    scores, summary = _run_record(root, decl["run"], runs)
    notes = _summary_problems(decl["run"], summary, name, metric(scores), places)
    if kind == "point":
        return [metric(scores)], notes
    if kind == "ci":
        vals = list(bootstrap_ci(scores, metric, n_boot=CHECK_N_BOOT, seed=CHECK_SEED))
        recorded = summary.get("ci95", {}).get(name)
        if recorded is not None:
            rec = [_fixed(100 * v, places) for v in recorded]
            new = [_fixed(100 * v, places) for v in vals[1:]]
            if rec != new:
                notes.append(
                    f"summary.json of {decl['run']} says {name} ci95 [{', '.join(rec)}] "
                    f"but items.jsonl gives [{', '.join(new)}]"
                )
        return vals, notes
    base_scores, base_summary = _run_record(root, decl["base"], runs)
    notes += _summary_problems(decl["base"], base_summary, name, metric(base_scores), places)
    # 차이만 적은 선언은 구간이 필요 없다. n_boot=0이면 짝(같은 건 id) 확인과 점 차이만 계산한다.
    with_ci = isinstance(decl["value"], list)
    n_boot = CHECK_N_BOOT if with_ci else 0
    vals = list(paired_bootstrap_diff(base_scores, scores, metric, n_boot=n_boot, seed=CHECK_SEED))
    return (vals if with_ci else vals[:1]), notes


@dataclass
class NumberCheck:
    """선언 하나의 대조 결과: 선언 그대로, 다시 낸 값(선언한 소수 자리로 반올림한 글, 계산하지 못했으면 빈
    목록), 어긋난 점(비어 있으면 통과)."""

    decl: dict
    got: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_numbers(decls: list[dict], root: str | Path = ".") -> list[NumberCheck]:
    """선언마다 문서 쪽(절 안의 context)과 기록 쪽(다시 낸 값, summary.json)을 대조한다. where·run·base 경로는
    root(저장소 뿌리) 기준이다. 한 선언이 실패해도(파일 없음, 짝이 맞지 않음 등) 나머지 선언은 계속 본다."""
    root = Path(root)
    texts: dict = {}
    runs: dict = {}
    out = []
    for decl in decls:
        res = NumberCheck(decl, problems=_decl_problems(decl))
        out.append(res)
        if res.problems:
            continue
        places = decl.get("places", 1)
        want = [_fixed(v, places) for v in _value_list(decl["value"])]
        res.problems += _context_problems(decl, want, root, texts)
        try:
            vals, notes = _recompute(decl, root, runs, places)
        except (OSError, ValueError, KeyError, TypeError) as e:
            res.problems.append(f"cannot recompute from {decl['run']}: {type(e).__name__}: {e}")
            continue
        res.got = [_fixed(100 * v, places) for v in vals]
        if res.got != want:
            res.problems.append(f"recomputed {_show(res.got)} != declared {_show(want)}")
        res.problems += notes
    return out


def _console(text: str) -> str:
    """표준 출력 인코딩으로 쓸 수 없는 글자를 \\uXXXX로 바꾼다. 출력을 파일로 돌리면 이 Windows 환경의 표준
    출력은 cp949인데 cp949에는 유니코드 빼기(U+2212)가 없어, context를 그대로 찍으면 UnicodeEncodeError로
    죽는다(짝지은 표가 같은 이유로 죽은 적이 있다, 커밋 0ea5791)."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, "backslashreplace").decode(enc)


def check_report(results: list[NumberCheck]) -> list[str]:
    """명령줄 출력 줄: 선언마다 "ok/FAIL 번호 파일 [절] context -> 다시 낸 값", 실패면 문제를 한 줄씩, 끝에
    합계."""
    lines = []
    for i, r in enumerate(results, 1):
        d = r.decl
        place = f"{d.get('where')}" + (f" [{d['section']}]" if d.get("section") else "")
        got = f"  -> {_show(r.got)}" if r.got else ""
        lines.append(f"{'ok' if r.ok else 'FAIL':<4} {i:>3}  {place}  {d.get('context')}{got}")
        lines += [f"           - {p}" for p in r.problems]
    failed = sum(not r.ok for r in results)
    lines.append(f"{len(results)} numbers checked, {failed} failed")
    return lines


# 명령줄 진입점. 위치 인자 실행들은 run_table로, --paired는 짝지은 표, --pick은 고르는 순서, --think-check는
# 생각 끄기 확인 줄(JSON)을 찍는다. 여러 옵션을 한 번에 주면 pick → think-check → paired → 실행 표 → --check
# 순서로 모두 찍는다. --check에서 하나라도 어긋나면 종료 코드 1이다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--paired", nargs=2, metavar=("A", "B"))
    ap.add_argument("--metrics", nargs="+", default=list(HEADLINE))
    ap.add_argument(
        "--pick", nargs="+", metavar="RUN", help="고르는 규칙 순서로 실행을 줄 세운다(첫 번째 = 고른 것)"
    )
    ap.add_argument(
        "--think-check", nargs="+", metavar="RUN", help="실행마다 생각 글, <think> 태그, 잘린 답 수를 센다"
    )
    ap.add_argument(
        "--check",
        metavar="FILE",
        help="수치 선언 파일(TOML, 예: docs/numbers.toml)의 숫자를 실행 기록에서 다시 내 공개 문서와 "
        "대조한다. 선언의 경로는 현재 폴더(저장소 뿌리) 기준이고, 하나라도 어긋나면 종료 코드 1",
    )
    args = ap.parse_args(argv)
    metrics = tuple(args.metrics)
    if args.pick:
        print(pick_table(args.pick))
    if args.think_check:
        for row in think_rows(args.think_check):
            print(json.dumps(row, ensure_ascii=False))
    if args.paired:
        print(paired_table(args.paired[0], args.paired[1], metrics))
    if args.runs:
        print(run_table(args.runs, metrics))
    if args.check:
        results = check_numbers(load_numbers(args.check), root=".")
        for line in check_report(results):
            print(_console(line))
        return 1 if any(not r.ok for r in results) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
