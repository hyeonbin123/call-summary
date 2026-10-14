"""분할 하나의 명세 JSONL을 쓴다. 명세는 결정적이므로 언제 다시 실행해도 같은 파일이 나온다."""

# [파일 역할] 데이터 만들기의 첫 단계(명세)를 실행하는 명령줄 진입점이다. 분할 하나(train, dev, test-a,
# test-b, test-c)의 명세를 만들어 JSONL로 저장한다.
# [왜 필요한가] 이 프로젝트는 정답을 먼저 만들고 대화를 나중에 만든다(README "데이터를 만드는 방식"). 명세가
# 구조 필드 정답의 유일한 출처이므로, 명세를 파일로 고정해 두고 대화 생성(gen.py)과 분할 확정(finalize.py)이
# 같은 명세를 읽게 한다.
# [설계 이유] 명세는 (분할, 업종, 번호, 유형)으로 seed를 정한 random.Random으로 뽑으므로(specs.make_spec) 이
# 명령을 언제 다시 돌려도 같은 파일이 나온다. 그래서 명세 파일을 잃어도 다시 만들 수 있다(docstring).
# [연결] 출력 파일은 gen.py의 --specs 입력이 되고, finalize.py가 data/specs/<분할>.jsonl을 다시 읽어 거절 건의
# 명세를 찾는다.

from __future__ import annotations

import argparse

# write_jsonl: 한 줄에 JSON 하나씩, 한글을 그대로(ensure_ascii=False) LF 줄바꿈으로 쓰는 공용 저장 함수.
# SPLIT_DOMAINS: 분할마다 어떤 업종을 넣는지 정한 표(test-c만 학습에서 뺀 카드 업종). make_specs: 업종마다
# 유형을 고르게 돌려 가며 명세를 만든다.
from .dataset import write_jsonl
from .specs import SPLIT_DOMAINS, make_specs


# 명령줄 인자를 받아 명세를 만들고 저장한 뒤 개수를 출력한다. argv를 받게 해 테스트에서 sys.argv 없이 부를 수
# 있다.
# --per-domain은 업종당 명세 수다. 실제 생성에는 dev 288(업종당 96), test-a 360, test-c 240, train 1,800
# 명세를 썼다(docs/experiments.md 1단계). 대화 생성에서 일부가 거절되므로 확정 분할 크기(finalize.TARGETS)보다
# 20% 더 만든다(1단계 규칙).
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # choices를 SPLIT_DOMAINS의 키로 묶어, 정의되지 않은 분할 이름을 넣으면 argparse가 바로 거부한다.
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_DOMAINS))
    ap.add_argument("--per-domain", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    # 분할 이름에 맞는 업종 목록으로 명세를 만들고, dataclass를 dict로 바꿔 JSONL에 쓴다.
    specs = make_specs(args.split, SPLIT_DOMAINS[args.split], args.per_domain)
    n = write_jsonl(args.out, (s.to_dict() for s in specs))
    print(f"{n} specs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
