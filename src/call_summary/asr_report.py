"""Speech-condition reports that need no model.

asr_report survival DATA...   per entity type, the share of gold values still findable in each transcript
                              file (the input analysis: run it before any model sees the data)
asr_report unfound RUN...     predicted values not found in the transcript, split into restored (equal to
                              gold) and invented, per type (stage 4 rule)

Values are found as in scoring: `values.occurs_in_transcript`, spoken=True for `*-asr` splits.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .dataset import load_items, read_jsonl
from .schema import AfterCallRecord
from .scoring import unfound_split, value_survival


def survival(paths: list[str | Path]) -> dict[str, dict]:
    """{path: {"items", "all_found", "types": {type: [found, n]}}}."""
    out: dict[str, dict] = {}
    for path in paths:
        types: dict[str, list[int]] = {}
        all_found = 0
        items = load_items(path)
        for it in items:
            got = value_survival(it.domain, it.gold(), it.transcript, spoken=it.split.endswith("-asr"))
            for t, found in got.items():
                cell = types.setdefault(t, [0, 0])
                cell[0] += sum(found)
                cell[1] += len(found)
            all_found += all(all(v) for v in got.values())
        out[str(path)] = {"items": len(items), "all_found": all_found, "types": dict(sorted(types.items()))}
    return out


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}" if b else "-"


def survival_table(got: dict[str, dict]) -> str:
    paths = list(got)
    lines = ["| 값 유형 | " + " | ".join(f"`{p}`" for p in paths) + " |", "|---|" + "---|" * len(paths)]
    types = sorted({t for g in got.values() for t in g["types"]})
    for t in types:
        cells = []
        for p in paths:
            found, n = got[p]["types"].get(t, [0, 0])
            cells.append(f"{_pct(found, n)} ({found}/{n})")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    cells = [f"{_pct(g['all_found'], g['items'])} ({g['all_found']}/{g['items']})" for g in got.values()]
    lines.append("| 모든 값이 남은 건 | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def unfound(run: str | Path, data: str | Path | None = None) -> dict:
    """Restored and invented values of one evaluation run (reports/<run_id>), against its dataset."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(data or manifest["data"])}
    predicted = 0
    restored: Counter = Counter()
    invented: Counter = Counter()
    for row in read_jsonl(run / "items.jsonl"):
        if not row.get("pred"):
            continue
        it = items[row["item_id"]]
        pred = AfterCallRecord.model_validate(row["pred"])
        predicted += len(pred.entities)
        r, i = unfound_split(it.domain, it.gold(), pred, it.transcript, spoken=it.split.endswith("-asr"))
        restored += r
        invented += i
    return {
        "predicted": predicted,
        "restored": dict(restored.most_common()),
        "invented": dict(invented.most_common()),
    }


def unfound_table(got: dict[str, dict]) -> str:
    lines = [
        "| 실행 | 예측한 값 | 전사에 없는 값 | 복원 (정답과 같음) | 지어낸 값 |",
        "|---|---|---|---|---|",
    ]
    for name, g in got.items():
        n = g["predicted"]
        r, i = sum(g["restored"].values()), sum(g["invented"].values())

        def detail(c: dict) -> str:
            return ", ".join(f"{t} {k}" for t, k in c.items())

        lines.append(
            f"| `{name}` | {n} | {_pct(r + i, n)}% ({r + i}) | {_pct(r, n)}% ({r}: {detail(g['restored'])}) "
            f"| {_pct(i, n)}% ({i}: {detail(g['invented'])}) |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survival", help="gold values still findable in transcript files")
    s.add_argument("data", nargs="+")
    u = sub.add_parser("unfound", help="restored vs invented values of evaluation runs")
    u.add_argument("runs", nargs="+")
    for p in (s, u):
        p.add_argument("--json", help="also write the numbers here")
    args = ap.parse_args(argv)
    if args.cmd == "survival":
        got = survival(args.data)
        print(survival_table(got))
    else:
        got = {str(r): unfound(r) for r in args.runs}
        print(unfound_table(got))
    if args.json:
        Path(args.json).write_text(json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
