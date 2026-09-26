"""Dataset items on disk: one JSON object per line.

{"item_id", "split", "domain", "spec": {...}, "turns": [{"speaker", "text"}], "summary": str | null,
 "source": "teacher:<model>" | "claude" | ..., "meta": {...}}

The gold record is always rebuilt from the spec (`Item.gold`), never stored separately.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .prompts import format_transcript
from .schema import AfterCallRecord
from .specs import Spec


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

    @property
    def transcript(self) -> str:
        return format_transcript(self.turns)

    def gold(self) -> AfterCallRecord:
        return self.spec.gold(self.summary or "")

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


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_items(path: str | Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]
