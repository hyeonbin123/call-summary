"""train과 다른 분할 사이에서 정답 기록이 글자까지 같은 건을 센다(보고만 한다. 분할은 바꾸지 않는다).

정답 기록 = 업종, 문의 유형, 처리 결과, 핵심 값의 (유형, 값) 집합, 처리 목록, 후속 조치(필요 여부와 코드).
참고 요약은 넣지 않는다. 핵심 값은 명세에 적힌 말투 그대로 견주고, 핵심 값·처리·후속 조치 코드의 순서는
보지 않는다.

python -m call_summary.overlap SPLIT...               분할마다 train과 정답 기록이 같은 건 수와 업종/유형별 수
python -m call_summary.overlap SPLIT... --run RUN...  기록된 실행의 exact를 겹친 건과 나머지로 나눠 센다
"""

# [파일 역할] 분할 확정(finalize)이 빼지 않는 "정답 기록 전체가 train의 한 건과 같은 건"을 세는 모델 없는 보고
# 도구다. 데이터셋 파일(datasets/<분할>.jsonl)마다 그런 건의 수와 업종/유형별 수를 내고, 기록된 평가 실행
# (reports/<run_id>)을 주면 그 실행의 exact를 겹친 건과 나머지로 나눠 센다.
# [왜 필요한가] finalize는 1단계 규칙("겹침 확인: test 세트의 핵심 값(번호)이 train에 나오면 그 건을 뺀다",
# docs/experiments.md 1단계)대로 번호 유형(주문번호, 운송장번호, 접수번호, 카드 끝자리)의 값만 본다. 번호가
# 없는 유형(통신의 요금제 변경·분실·정지처럼 요금제·기기 이름만 있는 것)은 값 풀이 작아, 명세 생성기가 train과
# 같은 조합을 다시 뽑으면 구조 필드가 학습 정답과 글자까지 같은 건이 dev·test에 남는다(대화는 따로 생성된다).
# 그 크기와 측정에 준 영향을 커밋된 도구로 다시 낼 수 있게 남긴다(docs/experiments.md 1단계 "생성 결과와
# 확정"의 2026-10-29 덧붙임). 분할과 기록 수치는 바꾸지 않는다.
# [설계 결정] (1) 핵심 값은 정규화하지 않는다: 학습 목표 글(prompts.record_to_json)은 명세의 말투를 그대로
# 담으므로, 정규화하면 같은 값이어도 말투가 다르면("85만 8천 원"과 "858,000원") 학습 정답의 글이 다르다. (2)
# 핵심 값·처리·후속 조치 코드의 순서는 보지 않는다: 채점(scoring.score_item)이 셋을 순서 없이 견주므로 순서만
# 다른 기록은 같은 정답이다. (3) 참고 요약은 뺀다: 교사가 건마다 따로 쓴 글이고 주 지표 exact는 요약을 보지
# 않는다.
# [흐름] record_key(건 하나의 열쇠) → train_keys(train 열쇠 집합) → split_report(분할 하나)·run_report(실행
# 하나) → split_table·run_table(문서에 옮길 표) → main(명령줄).

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

# Item: 데이터셋 한 건(정답 기록은 Item.gold()가 명세에서 만든다). load_items·read_jsonl: 데이터셋과 실행의
# items.jsonl을 읽는다. manifest_data: 실행 manifest의 데이터셋 경로(Windows에서 쓴 기록의 역슬래시 꼴)를
# 슬래시 꼴로 바꾼다(다른 분석 도구와 같은 함수).
from .dataset import Item, load_items, manifest_data, read_jsonl


# 건 하나의 정답 기록을 견줄 수 있는 열쇠(튜플)로 바꾼다. 입력: 데이터셋 한 건. 출력: (업종, 유형, 처리 결과,
# 정렬한 (유형, 값) 목록, 정렬한 처리 목록, 후속 조치 필요 여부, 정렬한 후속 조치 코드 목록).
# 업종을 넣는 이유: 라벨 이름이 업종끼리 같더라도 다른 업종의 기록은 다른 프롬프트(업종별 라벨 목록)에서 나온
# 다른 정답이다. 핵심 값은 Spec.gold()가 같은 (유형, 값)을 한 번만 넣으므로 정렬한 목록이 곧 집합이다.
def record_key(item: Item) -> tuple:
    """정답 기록의 구조 필드를 순서와 상관없이 견줄 수 있는 열쇠(요약은 뺀다, 값은 말투 그대로)."""
    g = item.gold()
    return (
        item.domain,
        g.category,
        g.resolution,
        tuple(sorted((e.type, e.value) for e in g.entities)),
        tuple(sorted(g.actions_taken)),
        g.follow_up.required,
        tuple(sorted(g.follow_up.codes)),
    )


# train 전체의 열쇠 집합. train 안에 같은 기록이 여러 건 있어도 한 번만 들어가므로, 다른 분할의 한 건은 몇
# 건과 같든 한 번만 센다. 입력은 보통 datasets/train.jsonl(학습에 쓴 1,500건)이다. 확정 전 생성 원본 전체가
# 아니라 실제로 학습한 건을 기준으로 삼는 이유: 모델이 본 정답이 문제이기 때문이다(finalize의 번호 검사는 원본
# 전체로 넓게 잡는다).
def train_keys(train: Iterable[Item]) -> set[tuple]:
    return {record_key(it) for it in train}


# 분할 하나에서 train과 정답 기록이 같은 건을 센다. 입력: 분할의 건 목록, train_keys의 집합. 출력 dict:
# n(건수), overlap(겹친 건 수), by_category("업종/유형" → 건수, 많은 순·같으면 이름순이라 출력이 늘 같다),
# ids(겹친 건의 item_id, 입력 순서).
def split_report(items: Sequence[Item], keys: set[tuple]) -> dict:
    hits = [it for it in items if record_key(it) in keys]
    by_cat = Counter(f"{it.domain}/{it.spec.category}" for it in hits)
    return {
        "n": len(items),
        "overlap": len(hits),
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: (-kv[1], kv[0]))),
        "ids": [it.item_id for it in hits],
    }


# 기록된 평가 실행 하나(reports/<run_id> 또는 outputs/runs/<run_id>)의 exact를 겹친 건과 나머지로 나눠 센다.
# 새 추론 없이 실행 폴더의 items.jsonl에 남은 건별 채점(score.exact)만 쓴다. 데이터셋은 manifest의 data
# 경로(현재 작업 폴더 기준의 상대 경로일 수 있다)를 쓰고, data를 주면 그것을 쓴다(테스트·다른 작업 폴더용).
# 출력 dict: run(실행 폴더 이름), data(슬래시 꼴 경로), n(채점된 건수), overlap·rest([맞은 건수, 건수]). 주의:
# 실행에 데이터셋에 없는 건이 있으면 다른 데이터셋을 가리킨 것이므로 ValueError로 멈춘다(조용히 건너뛰면
# 비율이 틀린다).
def run_report(run: str | Path, keys: set[tuple], data: str | Path | None = None) -> dict:
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    data_path = Path(data) if data is not None else Path(manifest_data(manifest))
    items = {it.item_id: it for it in load_items(data_path)}
    hit = {item_id for item_id, it in items.items() if record_key(it) in keys}
    counts = {"overlap": [0, 0], "rest": [0, 0]}
    for row in read_jsonl(run / "items.jsonl"):
        item_id = row["item_id"]
        if item_id not in items:
            raise ValueError(f"{run}: {item_id} is not in {data_path.as_posix()}")
        side = counts["overlap" if item_id in hit else "rest"]
        side[0] += bool(row["score"]["exact"])
        side[1] += 1
    return {
        "run": run.name,
        "data": data_path.as_posix(),
        "n": counts["overlap"][1] + counts["rest"][1],
        **counts,
    }


# "맞은 수/건수 (백분율)" 꼴의 표 칸. 건수가 0이면(겹친 건이 없는 분할) 백분율 대신 "-"를 적는다.
def _frac(hit: int, n: int) -> str:
    return f"{hit}/{n} ({100 * hit / n:.1f}%)" if n else f"{hit}/{n} (-)"


# 분할 표(마크다운). 입력: {분할 이름(명령줄에 준 경로 그대로) → split_report}. 업종/유형별 칸은 "업종/유형
# 건수"를 쉼표로 잇고, 겹친 건이 없으면 "-".
def split_table(reports: dict[str, dict]) -> str:
    lines = ["| 분할 | 건수 | train과 정답 기록이 같은 건 | 업종/유형별 |", "|---|---|---|---|"]
    for name, r in reports.items():
        pct = f"{100 * r['overlap'] / r['n']:.1f}%" if r["n"] else "-"
        cats = ", ".join(f"{k} {v}" for k, v in r["by_category"].items()) or "-"
        lines.append(f"| {name} | {r['n']} | {r['overlap']} ({pct}) | {cats} |")
    return "\n".join(lines)


# 실행 표(마크다운). 입력: run_report 결과 목록. 겹친 건과 나머지의 exact를 나란히 적어, 겹친 건에서만 점수가
# 높은지(학습에서 본 정답을 외운 효과가 있는지) 바로 보이게 한다.
def run_table(reports: Sequence[dict]) -> str:
    lines = ["| 실행 | 데이터 | 겹친 건 exact | 나머지 exact |", "|---|---|---|---|"]
    for r in reports:
        lines.append(f"| {r['run']} | {r['data']} | {_frac(*r['overlap'])} | {_frac(*r['rest'])} |")
    return "\n".join(lines)


# 명령줄 입구. 분할 파일(위치 인수)과 --run 실행 폴더를 받아 train 요약 한 줄, 분할 표, 실행 표를 찍는다.
# --show는 분할마다 겹친 건의 item_id도 찍는다. 분할도 실행도 없으면 사용법 오류(종료 코드 2)다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "splits",
        nargs="*",
        metavar="SPLIT",
        help="train이 아닌 데이터셋 파일들 (예: datasets/dev.jsonl datasets/test-a.jsonl)",
    )
    ap.add_argument(
        "--train",
        default="datasets/train.jsonl",
        help="학습에 쓴 train 파일 (기본 datasets/train.jsonl). 이 파일의 정답 기록과 견준다",
    )
    ap.add_argument(
        "--run",
        nargs="+",
        default=[],
        metavar="RUN",
        help="기록된 평가 실행 폴더들 (reports/<run_id>). 실행마다 manifest의 데이터셋으로 겹친 건을 정하고 "
        "exact를 겹친 건과 나머지로 나눈다. 여러 개를 받으므로 분할 파일 뒤에 둔다",
    )
    ap.add_argument("--show", action="store_true", help="분할마다 겹친 건의 item_id도 찍는다")
    args = ap.parse_args(argv)
    if not args.splits and not args.run:
        ap.error("give at least one dataset file or --run")

    train = load_items(args.train)
    keys = train_keys(train)
    print(f"train {args.train}: {len(train):,}건, 서로 다른 정답 기록 {len(keys):,}개")
    splits = {name: split_report(load_items(name), keys) for name in args.splits}
    if splits:
        print()
        print(split_table(splits))
    runs = [run_report(r, keys) for r in args.run]
    if runs:
        print()
        print(run_table(runs))
    if args.show:
        for name, r in splits.items():
            print(f"\n{name} 겹친 건 {r['overlap']}: " + (", ".join(r["ids"]) or "-"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
