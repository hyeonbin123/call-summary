"""Dataset items on disk: one JSON object per line.

{"item_id", "split", "domain", "spec": {...}, "turns": [{"speaker", "text"}], "summary": str | null,
 "source": "teacher:<model>" | "claude" | ..., "meta": {...}}

The gold record is always rebuilt from the spec (`Item.gold`), never stored separately.
"""

# [파일 역할] 디스크에 있는 데이터셋 한 건(Item)의 형식과, JSONL을 읽고 쓰는 공용 함수를 모아 둔 모듈이다.
# 생성(gen), 분할 확정(finalize), 평가(evaluate), 학습(train), 음성 조건 분석(asr_*), 판정(judge_*)이 모두 이
# 형식으로 데이터를 주고받는다.
# [설계 이유] 정답 기록을 파일에 따로 저장하지 않고 명세(spec)에서 매번 다시 만든다(Item.gold). 정답의 출처를
# 명세 생성기 하나로 묶어, LLM이 만든 구조화 정답이 섞일 여지를 없애려는 것이다(docs/experiments.md 공통
# "정답: 명세 생성기에서만 나온다").
# [주요 구성] Item(한 건의 dataclass와 dict 변환) → read_jsonl/write_jsonl(줄 단위 입출력) → load_items(파일 →
# Item 목록).

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

# format_transcript: 발화 목록을 "화자: 내용" 줄로 이어 붙여, 모델이 받는 전사 문자열과 같은 꼴을 만든다.
# AfterCallRecord: 출력 스키마(pydantic). Spec: 명세 dataclass이며 정답 기록을 만드는 gold()를 가진다.
from .prompts import format_transcript
from .schema import AfterCallRecord
from .specs import Spec


# 데이터셋 한 건. item_id·split·domain은 식별 정보, spec은 정답의 원천, turns는 대화 발화 목록, summary는
# 교사가 쓴 참고 요약(학습 목표로만 쓰고 채점에는 쓰지 않는다), source는 대화를 쓴 쪽(교사 모델 이름이나
# claude), meta는 생성 시도 횟수 같은 부가 정보다.
@dataclass
class Item:
    item_id: str
    split: str
    domain: str
    spec: Spec
    turns: list[dict]
    summary: str | None = None  # reference summary (training target); scoring never uses it
    source: str = ""
    meta: dict = field(default_factory=dict)

    # 모델에 넣을 전사 문자열. 평가·학습·서비스가 모두 같은 format_transcript를 거치게 해 입력 꼴이 어긋나지
    # 않게 한다.
    @property
    def transcript(self) -> str:
        return format_transcript(self.turns)

    # 정답 기록을 명세에서 다시 만든다. 구조 필드는 명세에서 바로 나오고, summary 칸에만 참고 요약을
    # 채운다(없으면 빈 문자열).
    def gold(self) -> AfterCallRecord:
        return self.spec.gold(self.summary or "")

    # 파일에 쓰는 dict 형태. 위 모듈 docstring의 키 순서와 같다.
    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "split": self.split,
            "domain": self.domain,
            "spec": self.spec.to_dict(),
            "turns": self.turns,
            "summary": self.summary,
            "source": self.source,
            "meta": self.meta,
        }

    # 파일의 dict에서 Item을 되살린다. summary·source·meta는 .get으로 읽어, 이 키가 빠진 dict도 기본값(None,
    # 빈 문자열, 빈 dict)으로 읽는다.
    @classmethod
    def from_dict(cls, d: dict) -> Item:
        return cls(
            item_id=d["item_id"],
            split=d["split"],
            domain=d["domain"],
            spec=Spec.from_dict(d["spec"]),
            turns=d["turns"],
            summary=d.get("summary"),
            source=d.get("source", ""),
            meta=d.get("meta", {}),
        )


# JSONL을 한 줄씩 dict로 내준다(제너레이터라 큰 파일도 한 번에 메모리에 올리지 않는다). 빈 줄과 줄 끝 공백은
# 건너뛰어, 끝에 빈 줄이 있거나 CRLF로 저장된 파일도 그대로 읽는다.
def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# dict들을 JSONL로 쓰고 쓴 줄 수를 돌려준다. 상위 폴더가 없으면 만든다.
# ensure_ascii=False로 한글을 그대로 남겨 사람이 파일을 바로 읽을 수 있게 하고, newline="\n"으로 Windows에서도
# LF로 쓴다(저장소의 .gitattributes도 jsonl을 LF로 고정한다, 커밋 9a846c3).
def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


# 데이터셋 파일 전체를 Item 목록으로 읽는다. 평가·분석 도구 대부분이 이 함수로 datasets/*.jsonl을 연다.
def load_items(path: str | Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]
