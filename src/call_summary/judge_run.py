"""평가 실행 하나의 요약들을 판정 모델로 채점한다.

<run>/items.jsonl(예측 기록)과 <run>/manifest.json에 적힌 데이터셋 파일을 읽어, <run>/judge.jsonl(건마다
SummaryScore 하나와 답의 토큰 수)과 <run>/judge_summary.json을 쓴다. --out으로 다른 출력 파일을 정할 수
있고(요약은 그 옆에 <stem>_summary.json으로 간다), 그래서 두 번째 판정 모델이 기록된 j1 판정을 덮어쓰지
않는다. --hand <file>을 주면 사람 채점 JSONL({"item_id", "verdicts": [...], "wrong_statements": n})과
사실 단위로 비교해 일치율을 요약에 더한다. 요약에는 코드 커밋과 실행 뒤의 Ollama /api/ps도 기록한다.
"""

# [파일 역할] 기록된 평가 실행 하나의 요약을 판정 모델로 채점하는 명령줄 도구다. 건마다 판정(SummaryScore)과
# 토큰 수를 JSONL로 남기고, 사실 재현율·오답 비율·측정 조건 확인값을 요약 JSON으로 쓴다. 사람 채점 양식을
# 만드는 hand_template_main도 함께 있다.
# [왜 필요한가] 판정은 모델 추론과 따로 돌린다. 평가 실행(evaluate)이 남긴 예측 기록을 나중에 판정하므로, 판정
# 모델을 바꿔도 학생 모델을 다시 돌릴 필요가 없다. 2단계에서는 j1(qwen2.5 14B)로 실행 11개를 판정했고, 재보정
# 뒤에는 채택한 qwen3.5:9b로 서빙 학생의 실행 넷을 판정했다(docs/experiments.md "요약 판정 재보정").
# [설계 이유] --out으로 다른 파일 이름을 주면 기록된 j1 판정(judge.jsonl)을 덮어쓰지 않는다. 결과 파일에 이어
# 쓰기를 해 중간에 멈춰도 이미 판정한 건은 건너뛴다. 요약 JSON에 코드 커밋과 Ollama /api/ps를 남겨 어떤
# 코드·어떤 GPU 적재 상태로 판정했는지 추적한다(커밋 0103ac9).

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# evaluate.py의 도우미를 그대로 쓴다: THINK(--think 값 → 요청의 think), _git(커밋·작업 트리 상태),
# loaded_models(/api/ps), model_fingerprint(Ollama digest 등 모델 지문), parse_option(KEY=VALUE 옵션 해석),
# run_checks(잘린 답·생각 글·컨텍스트를 채운 답 세기). 평가와 판정이 같은 방식으로 실행 조건을 기록하게 하려는
# 것이다.
from .dataset import load_items, manifest_data, read_jsonl, write_jsonl
from .evaluate import THINK, _git, loaded_models, model_fingerprint, parse_option, run_checks
from .judge import (
    JUDGE_NUM_CTX,
    JUDGE_VERSION,
    SummaryScore,
    agreement,
    fact_recall,
    judge_reply,
    make_judge,
    reply_fields,
    wrong_rate,
)


# 판정 결과 JSONL(또는 같은 꼴의 사람 채점 파일)을 SummaryScore 목록으로 읽는다. wrong_statements와 ok가 없는
# 사람 채점 행은 0과 True로 친다. 이어 하기 때 이미 쓴 판정을 다시 불러 집계에 넣는 데도 쓴다.
def load_scores(path: Path) -> list[SummaryScore]:
    return [
        SummaryScore(
            d["item_id"], tuple(d["verdicts"]), int(d.get("wrong_statements", 0)), bool(d.get("ok", True))
        )
        for d in read_jsonl(path)
    ]


# 실행 하나를 판정한다. 기본값(모델 qwen2.5:14b-instruct, GPU 44층, 생각 끔, 옵션 없음)은 2단계 j1의 요청
# 그대로다. 2026년 판정 모델은 --model, --num-gpu 99, --option으로 바꿔 부른다(docs/experiments.md 재보정 규칙
# 3의 표).
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="평가 실행 폴더")
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--data", help="데이터셋 JSONL(기본: 실행 manifest에 적힌 파일)")
    ap.add_argument("--hand", help="일치율을 잴 사람 채점 JSONL")
    # 14B 판정 모델의 층을 모두 GPU에 올리면 VRAM이 차서 Windows가 시스템 메모리로 밀어내 요청이 아주
    # 느려진다. 44층만 올려 여유를 남긴다(docs/experiments.md 1단계 진행 기록).
    ap.add_argument("--num-gpu", type=int, default=44, help="14B 판정 모델의 GPU 층 수(VRAM 여유)")
    ap.add_argument(
        "--think", choices=sorted(THINK), default="off", help="판정 모델에 보내는 생각 모드 설정(j1: off)"
    )
    ap.add_argument(
        "--option",
        type=parse_option,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="더할 Ollama 옵션, 여러 번 줄 수 있다(j1은 보내지 않음)",
    )
    ap.add_argument("--out", help="출력 JSONL(기본 <run>/judge.jsonl)")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)

    # 판정 시작 시각과 코드 상태(커밋, 추적 파일이 바뀌었는지)를 먼저 기록한다. 판정 도중 코드를 바꿔도 시작
    # 때의 상태가 남는다.
    started = time.strftime("%Y%m%d-%H%M%S")
    git_commit = _git("rev-parse", "HEAD")
    git_dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    # 판정에는 명세의 사실 목록과 전사가 필요하므로, 실행 manifest에 적힌 데이터 파일(또는 --data)에서 건을
    # item_id로 찾는다. 기록된 manifest의 경로는 Windows 역슬래시 꼴이라 manifest_data로 슬래시 꼴로 읽는다.
    # --limit은 스모크(앞 몇 건만)용이다.
    items = {it.item_id: it for it in load_items(args.data or manifest_data(manifest))}
    rows = list(read_jsonl(run / "items.jsonl"))[: args.limit]
    options = dict(args.option)
    judge = make_judge(model=args.model, num_gpu=args.num_gpu, think=THINK[args.think], options=options)
    # 출력 파일과 그 옆의 요약 파일(<stem>_summary.json)을 정한다. 기본은 실행 폴더의 judge.jsonl이다.
    out_path = Path(args.out) if args.out else run / "judge.jsonl"
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 이어 하기: 출력 파일에 이미 있는 item_id는 건너뛰고, 그 판정은 집계에 다시 넣는다. 집계에는 이번에 고른
    # 건(rows, --limit 적용)만 넣는다. 파일에 그보다 많은 행이 있어도(예: 전체 판정 뒤 --limit으로 다시 부름)
    # 요약의 건수와 지표가 이번 호출의 범위와 맞게 하려는 것이다.
    wanted = {row["item_id"] for row in rows}
    done = {d["item_id"] for d in read_jsonl(out_path)} if out_path.exists() else set()
    scores: list[SummaryScore] = [s for s in load_scores(out_path) if s.item_id in wanted] if done else []
    # 한 건 판정할 때마다 한 줄 쓰고 flush해, 판정 모델이 죽거나 중간에 멈춰도 이미 쓴 줄은 남는다. 예측
    # 기록(pred)이 없거나 요약이 빈 건은 빈 요약으로 판정에 넘겨 모두 누락으로 친다.
    with open(out_path, "a", encoding="utf-8", newline="\n") as f:
        for n, row in enumerate(rows, 1):
            if row["item_id"] in done:
                continue
            it = items[row["item_id"]]
            summary = (row.get("pred") or {}).get("summary", "") or ""
            s, reply = judge_reply(judge, it.item_id, it.spec, it.transcript, summary)
            scores.append(s)
            f.write(json.dumps(s.to_dict() | reply_fields(reply), ensure_ascii=False) + "\n")
            f.flush()
            # 20건마다 지금까지의 사실 재현율을 표준 오류로 찍어 긴 판정의 진행을 본다.
            if n % 20 == 0 or n == len(rows):
                print(f"[{n}/{len(rows)}] recall={fact_recall(scores):.3f}", file=sys.stderr, flush=True)

    # 판정 요약: 판정 모델·판·설정, 모델 지문(digest), 코드 상태, 시작·끝 시각, 판정 뒤 /api/ps(모델이 GPU에
    # 다 올라갔는지), run_checks(잘린 답, 생각 글, 컨텍스트를 채운 답 수), 건수, 판정 실패 수, 사실 재현율,
    # 오답 비율.
    # run_checks는 이번 호출에서 쓴 줄만이 아니라 이어 하기 전에 쓴 줄까지 출력 파일에서 다시 읽어 세되, 위
    # 집계처럼 이번에 고른 건만 센다.
    table: dict = {
        "judge_model": judge.name,
        "judge_version": JUDGE_VERSION,
        "num_gpu": args.num_gpu,
        "think": args.think,
        "options": options,
        "fingerprint": model_fingerprint(judge),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "started": started,  # 이번 호출의 시작 시각(이어 쓴 파일은 앞선 줄을 그대로 둔다)
        "finished": time.strftime("%Y%m%d-%H%M%S"),
        "loaded_models_after": loaded_models(judge),
        "run_checks": run_checks([d for d in read_jsonl(out_path) if d["item_id"] in wanted], JUDGE_NUM_CTX),
        "n": len(scores),
        "n_judge_failed": sum(1 for s in scores if not s.ok),
        "fact_recall": fact_recall(scores),
        "wrong_rate": wrong_rate(scores),
    }
    # 사람 채점 파일을 주면 판정 모델과의 사실 단위 일치율을 더한다. 2단계 신뢰도 규칙(일치율 0.8 미만이면
    # 요약 지표는 참고치)을 이 값으로 판단했다.
    if args.hand:
        hand = load_scores(Path(args.hand))
        table["hand_n"] = len(hand)
        table["hand_agreement"] = agreement(scores, hand)
        table["hand_fact_recall"] = fact_recall(hand)
    summary_path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False))
    return 0


# 사람(Claude) 채점용 양식을 만든다: 실행의 건을 seed로 섞어 앞 n건(기본 50)을 뽑고, 사실 목록과 예측 요약에
# 빈 판정 칸을 붙인다. 2단계 판정 신뢰도 측정의 hand50이 이 양식에서 나왔다("무작위 50건(seed 0)",
# docs/experiments.md 2단계).
# __main__에서는 부르지 않으므로 README처럼 python -c로 이 함수를 직접 부른다.
def hand_template_main(argv: list[str] | None = None) -> int:
    """사람 채점 양식을 쓴다: 건, 사실 목록, 예측 요약, 빈 판정 칸."""
    ap = argparse.ArgumentParser(description=hand_template_main.__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    import random

    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    # 실행 manifest의 데이터 파일(역슬래시 꼴 기록은 manifest_data가 슬래시 꼴로 바꿈)에서 건을 찾는다.
    items = {it.item_id: it for it in load_items(manifest_data(manifest))}
    rows = list(read_jsonl(run / "items.jsonl"))
    # seed를 고정한 random.Random으로 섞어 같은 seed면 늘 같은 건이 뽑힌다.
    random.Random(args.seed).shuffle(rows)
    out = []
    for row in rows[: args.n]:
        it = items[row["item_id"]]
        out.append(
            {
                "item_id": it.item_id,
                "facts": list(it.spec.facts),
                "summary": (row.get("pred") or {}).get("summary", "") or "",
                "verdicts": [],
                "wrong_statements": 0,
            }
        )
    write_jsonl(args.out, out)
    print(f"{len(out)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
