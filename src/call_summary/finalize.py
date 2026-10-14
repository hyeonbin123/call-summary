"""생성 원본 파일을 확정 데이터셋 분할로 바꾼다.

업종마다 명세 번호 순서로 통과한 건 중 앞의 N건만 남기고(docs/experiments.md 1단계), 번호가 train에도
나오는 test 건을 빼며, 거절된 명세 수를 유형별로 보고한다.
"""

# [파일 역할] 데이터 만들기의 마지막 단계다. 교사가 생성한 원본(data/raw/<분할>.jsonl)을 읽어, 지금의 자동
# 검사를 다시 통과한 것 중 업종마다 앞의 N건만 골라 확정 분할(data/<분할>.jsonl)을 쓰고, 거절 비율
# 보고서(finalize_report.json)를 남긴다. 확정 분할은 저장소의 datasets/로 복사해 커밋했다(docs/experiments.md
# 1단계).
# [왜 필요한가] (1) 생성은 분할마다 다른 시점에 돌았고 그 사이 검사가 늘었다(본인 확인 일치 검사 f4d0efe).
# 그래서 확정할 때 모든 분할에 현재 검사를 다시 적용해, 최종 데이터가 같은 검사를 통과한 것만 남게 한다. (2)
# test 세트의 번호가 train에 나오면 외운 값으로 맞힐 수 있으므로 그 건을 뺀다(1단계 규칙 "겹침 확인"). (3)
# 버린 비율을 유형별로 보고하라는 규칙(1단계)을 채운다.
# [흐름] train 원본에서 번호 집합을 모음 → 분할마다 현재 검사로 다시 거름 → (train 외) train과 번호가 겹치는
# 건 제외 → 업종마다 번호 순서로 앞의 N건 → 저장, 거절 보고.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# check_dialogue: gen.py의 대화 자동 검사(명세 값이 글자 그대로 나오는지, 명세에 없는 금액·번호가 없는지 등).
# 생성 때와 같은 함수를 다시 불러 지금 기준으로 거른다.
# normalize: 번호를 비교 가능한 꼴로 정규화한다(겹침 확인에 쓴다). DOMAINS: 업종마다 어떤 값 유형이 번호(kind
# "id")인지 알려 준다.
from .dataset import Item, load_items, read_jsonl, write_jsonl
from .domains import DOMAINS
from .gen import check_dialogue
from .specs import Spec
from .values import normalize

# 분할별 업종당 확정 건수(docs/experiments.md 1단계 규칙: train 500, dev 80, test-a 100, test-c 카드 200,
# test-b 업종당 20). 명세는 이보다 20% 더 만들어 거절을 메운다.
# 처음 초안은 train 700이었지만 파일럿에서 14B가 받아들인 대화 한 건에 약 61초가 걸려 500으로 줄였다(같은 절).
# 키 순서가 기본 처리 순서이며 train이 맨 앞이다.
TARGETS = {"train": 500, "dev": 80, "test-a": 100, "test-c": 200, "test-b": 20}


# item_id 끝의 번호("dev-shop-00012" → 12)를 꺼낸다. 명세를 만든 순서이므로 "앞의 N건"을 정하는 기준이 된다.
def _index(item_id: str) -> int:
    return int(item_id.rsplit("-", 1)[1])


# 업종마다 명세 번호 순서로 앞의 per_domain건을 고른다. 결과를 보고 고르는 것이 아니라 번호 순서로 자르므로 몇
# 번을 다시 돌려도 같은 분할이 나오고, 골라 담기로 분할이 치우칠 여지가 없다. 통과한 건이 모자라면 있는 만큼만
# 들어가고 main이 short_by_domain으로 보고한다.
def take_first(items: list[Item], per_domain: int) -> list[Item]:
    by_dom: dict[str, list[Item]] = defaultdict(list)
    for it in sorted(items, key=lambda it: (it.domain, _index(it.item_id))):
        by_dom[it.domain].append(it)
    out: list[Item] = []
    for dom in sorted(by_dom):
        out.extend(by_dom[dom][:per_domain])
    return out


# 한 건의 명세에서 번호 유형(주문번호, 운송장번호, 접수번호, 카드 끝자리) 값만 정규화해 모은다. 1단계 규칙이
# 겹침을 번호로 정했으므로 금액·날짜·이름은 보지 않는다.
def id_values(item: Item) -> set[str]:
    """한 건의 정규화한 번호 유형 값(주문번호, 운송장번호, 접수번호, 카드 끝자리)."""
    d = DOMAINS[item.domain]
    out = set()
    for sv in item.spec.slots:
        if d.entity_type(sv.type).kind == "id":
            n = normalize("id", sv.value)
            if n:
                out.add(n)
    return out


# 한 분할의 생성 거절 통계를 만든다: 시도한 명세 수, 통과, 거절, 거절률, 유형별 거절 수, 이유별 거절 수.
# 명세는 specs_dir/<분할>.jsonl에서 읽는다(build_specs의 출력, main의 --specs, 기본 data/specs). 거절 기록은
# gen.py가 남긴 <분할>.rejects.jsonl이다. 예전에는 명세 폴더가 현재 작업 폴더 기준 data/specs로 고정돼,
# --raw로 다른 원본을 주어도 명세는 그 폴더에서 읽었다.
def rejection_report(raw_dir: Path, split: str, specs_dir: Path = Path("data/specs")) -> dict:
    rej_path = raw_dir / f"{split}.rejects.jsonl"
    specs = {d["spec_id"]: Spec.from_dict(d) for d in read_jsonl(specs_dir / f"{split}.jsonl")}
    rejected = list(read_jsonl(rej_path)) if rej_path.exists() else []
    ok = load_items(raw_dir / f"{split}.jsonl")
    by_cat: Counter = Counter()
    reasons: Counter = Counter()
    for r in rejected:
        s = specs.get(r["spec_id"])
        if s:
            by_cat[f"{s.domain}/{s.category}"] += 1
        # 문제 문장에서 대괄호·괄호부터 뒤의 구체적인 값과 ' 따옴표로 시작하는 인용을 잘라 이유의 종류만
        # 남긴다. 그래야 "extra amount ['32,000원']"과 "extra amount ['5,000원']"이 "extra amount" 하나로,
        # "missing 주문번호 'D1234567'"은 "missing 주문번호"로 묶인다.
        for p in r["problems"]:
            reasons[re.sub(r"[\[\(].*", "", p).strip().split(" '")[0]] += 1
    total = len(ok) + len(rejected)
    return {
        "attempted": total,
        "accepted": len(ok),
        "rejected": len(rejected),
        "reject_rate": round(len(rejected) / total, 4) if total else None,
        "rejected_by_category": dict(by_cat.most_common()),
        "reasons": dict(reasons.most_common()),
    }


# 명령줄 진입점. --raw(생성 원본 폴더), --specs(명세 폴더, 거절 건의 유형을 찾는 데 쓴다), --out(확정 분할을
# 쓸 폴더), --splits(처리할 분할, 기본은 TARGETS의 다섯 개).
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--specs", default="data/specs", help="분할별 명세 파일 폴더(build_specs 출력)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--splits", nargs="+", default=list(TARGETS))
    args = ap.parse_args(argv)
    raw = Path(args.raw)
    report: dict = {}
    # train 원본 전체(확정되지 않을 건까지)에서 번호를 모은다. 확정된 train만이 아니라 원본 전체를 기준으로
    # 삼아 겹침을 넓게 잡는다.
    train_ids: set[str] = set()
    train_path = raw / "train.jsonl"
    if train_path.exists():
        for it in load_items(train_path):
            train_ids |= id_values(it)
    for split in args.splits:
        items = load_items(raw / f"{split}.jsonl")
        # 분할을 생성한 뒤에 검사가 늘었을 수 있으므로, 지금의 검사를 모든 분할에 다시 적용한다.
        # check_dialogue는 문제 목록을 돌려주므로, 목록이 비어 있지 않은 건만 recheck에 남겨 빼고, 뺀 이유는
        # 보고서에 그대로 적는다(dev 3건, test-a 2건이 이렇게 빠졌다, docs/experiments.md 1단계 표).
        failed = {it.item_id: check_dialogue(it.spec, it.turns) for it in items}
        recheck = {k: v for k, v in failed.items() if v}
        items = [it for it in items if it.item_id not in recheck]
        overlap = []
        # train이 아닌 분할에서만 train과 번호가 하나라도 겹치는 건을 뺀다(확정 때 겹친 건은 모든 분할에서
        # 0건이었다).
        if split != "train":
            overlap = [it.item_id for it in items if id_values(it) & train_ids]
            items = [it for it in items if it.item_id not in set(overlap)]
        chosen = take_first(items, TARGETS[split])
        # 목표보다 모자란 업종과 모자란 건수. 모자라면 화면에도 찍어 바로 알 수 있게 한다.
        short = {
            dom: TARGETS[split] - n
            for dom, n in Counter(it.domain for it in chosen).items()
            if n < TARGETS[split]
        }
        n = write_jsonl(Path(args.out) / f"{split}.jsonl", (it.to_dict() for it in chosen))
        # test-b는 Claude가 직접 쓴 대화를 따로 조립한 것이라(docs/experiments.md 1단계) 생성 거절 기록이 없어
        # 거절 통계를 만들지 않는다. 나머지 분할은 거절 통계에 다시 거른 건, train 겹침, 모자란 건수를 더한다.
        rep = rejection_report(raw, split, Path(args.specs)) if split != "test-b" else {}
        rep.update(
            {
                "written": n,
                "dropped_recheck": recheck,
                "dropped_train_overlap": overlap,
                "short_by_domain": short,
            }
        )
        report[split] = rep
        print(f"{split}: {n} items" + (f", short {short}" if short else ""))
    # 모든 분할의 보고서를 한 파일로 남긴다. docs/experiments.md 1단계의 확정 표가 이 보고서에서 나왔다.
    Path(args.out, "finalize_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
