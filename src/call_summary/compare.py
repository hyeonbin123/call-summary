"""평가 실행들로 표를 만든다.

compare RUN...            실행마다 한 줄: 대표 지표와 95% 구간, 지연
compare --paired A B      지표 차이 B - A와 짝지은 부트스트랩 구간(양쪽이 같은 건)
compare --pick RUN...     고르는 규칙 순서로 줄 세운 실행: exact, 다음 핵심 값 F1, 다음 낮은 환각
compare --think-check RUN...  실행마다: 생각 글, <think> 태그, num_predict에서 잘린 답, 답 토큰 수
"""

# [파일 역할] 기록된 평가 실행(outputs/runs/<id>/ 또는 reports/<id>/)을 읽어 docs/experiments.md에 붙일 표를
# 만드는 모델 없는 도구다. 실행마다 지표와 95% 구간을 나란히 보여 주는 표, 두 실행의 짝지은 차이 표, dev에서
# 고르는 규칙대로 실행을 줄 세운 표, 생각 끄기 확인 표를 낸다.
# [왜 필요한가] 이 프로젝트는 판정을 "같은 건끼리 짝지은 부트스트랩 95% 구간의 하한이 0보다 큰가"로
# 내린다(docs/experiments.md 공통). 결과 표와 판정을 손으로 계산하지 않고 실행 기록(manifest.json,
# summary.json, items.jsonl)에서 다시 계산하게 해, 누구나 같은 숫자를 다시 낼 수 있게 한다.
# [연결] evaluate.py가 쓴 실행 폴더를 읽고, 지표 계산은 scoring.py(METRICS, paired_bootstrap_diff)를 그대로
# 쓴다.

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# HEADLINE: 표에 기본으로 넣는 지표 목록(evaluate.py와 같은 것을 써서 실행 요약과 표의 열이 어긋나지 않게
# 한다).
# METRICS: 지표 이름 → 건별 점수 목록에서 값을 계산하는 함수. ItemScore: 건 하나의 채점 결과.
# paired_bootstrap_diff: 같은 건 id를 양쪽에서 함께 다시 뽑아 B − A 차이의 구간을 낸다.
from .dataset import read_jsonl
from .evaluate import HEADLINE
from .scoring import METRICS, ItemScore, paired_bootstrap_diff


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


# 명령줄 진입점. 위치 인자 실행들은 run_table로, --paired는 짝지은 표, --pick은 고르는 순서, --think-check는
# 생각 끄기 확인 줄(JSON)을 찍는다. 여러 옵션을 한 번에 주면 pick → think-check → paired → 실행 표 순서로 모두
# 찍는다.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
