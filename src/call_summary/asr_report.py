"""Speech-condition reports that need no model.

asr_report survival DATA...   per entity type, the share of gold values still findable in each transcript
                              file (the input analysis: run it before any model sees the data)
asr_report unfound RUN...     predicted values not found in the transcript, split into restored (equal to
                              gold) and invented, per type (stage 4 rule); also the empty values among them
asr_report tracking-forms DATA...
                              how the recogniser wrote each tracking number of the written utterances
                              (input analysis of the speech condition v2)

Values are found as in scoring: `values.occurs_in_transcript`, spoken=True for `*-asr` splits.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from .dataset import load_items, read_jsonl
from .domains import DOMAINS
from .schema import AfterCallRecord
from .scoring import unfound_split, value_survival
from .values import occurs_in_transcript


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
    predicted = empty = 0
    restored: Counter = Counter()
    invented: Counter = Counter()
    for row in read_jsonl(run / "items.jsonl"):
        if not row.get("pred"):
            continue
        it = items[row["item_id"]]
        pred = AfterCallRecord.model_validate(row["pred"])
        spoken = it.split.endswith("-asr")
        predicted += len(pred.entities)
        r, i = unfound_split(it.domain, it.gold(), pred, it.transcript, spoken=spoken)
        restored += r
        invented += i
        kinds = {et.label: et.kind for et in DOMAINS[it.domain].entity_types}
        empty += sum(
            1
            for e in pred.entities
            if not e.value.strip()
            and not occurs_in_transcript(kinds.get(e.type, "text"), e.value, it.transcript, spoken)
        )
    return {
        "predicted": predicted,
        "restored": dict(restored.most_common()),
        "invented": dict(invented.most_common()),
        "empty": empty,  # unfound values that are empty strings (all invented), as stage 3 counted them
    }


def unfound_table(got: dict[str, dict]) -> str:
    lines = [
        "| 실행 | 예측한 값 | 전사에 없는 값 | 복원 (정답과 같음) | 지어낸 값 | 그중 빈 값 "
        "| 빈 값을 뺀 전사에 없는 값 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, g in got.items():
        n = g["predicted"]
        r, i = sum(g["restored"].values()), sum(g["invented"].values())

        def detail(c: dict) -> str:
            return ", ".join(f"{t} {k}" for t, k in c.items())

        lines.append(
            f"| `{name}` | {n} | {_pct(r + i, n)}% ({r + i}) | {_pct(r, n)}% ({r}: {detail(g['restored'])}) "
            f"| {_pct(i, n)}% ({i}: {detail(g['invented'])}) | {g.get('empty', 0)} "
            f"| {_pct(r + i - g.get('empty', 0), n)}% ({r + i - g.get('empty', 0)}) |"
        )
    return "\n".join(lines)


_TRACKING = re.compile(r"\d{4}-\d{4}-\d{4}")
_HANGUL_DIGIT_NAMES = re.compile(r"[공영일이삼사오육칠팔구]{4,}")
_HANGUL = re.compile(r"[가-힣]")
_RANGE_WORD = "에서"  # the v1 range reading ("A에서 B"), counted on its own row


def heard_span(written: str, heard: str, start: int, end: int) -> str:
    """The part of `heard` lined up (difflib, by character) with written[start:end], widened over digits
    that touch it on either side so inserted digits at its edges are kept."""
    hs, he = 0, len(heard)
    found_start = found_end = False
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, written, heard, autojunk=False).get_opcodes():
        if not found_start and i1 <= start < i2:
            hs, found_start = (j1 + start - i1 if tag == "equal" else j1), True
        if not found_end and i1 < end <= i2:
            he, found_end = (j1 + end - i1 if tag == "equal" else j2), True
    while hs > 0 and heard[hs - 1].isdigit():
        hs -= 1
    while he < len(heard) and heard[he].isdigit():
        he += 1
    return heard[hs:he]


def tracking_form(number: str, written: str, heard: str, at: int = 0) -> tuple[str, str]:
    """(form, heard span) of one tracking number said in `written` (at index `at`) and heard as `heard`.

    Forms: "as_written"; "split_found" (found by scoring's spoken rule, e.g. groups split by commas);
    "hangul_digit_names" (not found, 4+ Hangul digit names in the heard utterance); else lost, by the heard
    span lined up with the number: "lost_hangul" (a Hangul syllable in it, "에서" aside, e.g. "651일 5502"),
    "lost_fewer" / "lost_more" (fewer / more than its digits) or "lost_changed" (as many digits).
    """
    if number in heard:
        return "as_written", number
    if occurs_in_transcript("id", number, heard, spoken=True):
        return "split_found", ""
    if _HANGUL_DIGIT_NAMES.search(heard):
        return "hangul_digit_names", ""
    span = heard_span(written, heard, at, at + len(number))
    if _HANGUL.search(span.replace(_RANGE_WORD, "")):
        return "lost_hangul", span
    got, want = sum(c.isdigit() for c in span), sum(c.isdigit() for c in number)
    if got < want:
        return "lost_fewer", span
    return ("lost_more" if got > want else "lost_changed"), span


TRACKING_FORMS = (
    "as_written",
    "split_found",
    "hangul_digit_names",
    "lost_hangul",
    "lost_fewer",
    "lost_more",
    "lost_changed",
)


def tracking_forms(paths: list[str | Path]) -> dict[str, dict]:
    """{path: {"numbers", <form>: count..., "heard_has_에서", "lost": [[item_id, number, form, span]]}}.

    Every tracking number in each utterance's written text (meta.asr.written), against what was heard."""
    out: dict[str, dict] = {}
    for path in paths:
        n: Counter = Counter({f: 0 for f in TRACKING_FORMS})
        lost: list[list[str]] = []
        numbers = in_range_word = 0
        for it in load_items(path):
            for written, turn in zip(it.meta["asr"]["written"], it.turns, strict=True):
                heard = turn["text"]
                for m in _TRACKING.finditer(written):
                    form, span = tracking_form(m.group(), written, heard, m.start())
                    numbers += 1
                    n[form] += 1
                    in_range_word += _RANGE_WORD in heard
                    if form.startswith("lost_"):
                        lost.append([it.item_id, m.group(), form, span])
        out[str(path)] = {"numbers": numbers, **n, "heard_has_에서": in_range_word, "lost": lost}
    return out


_TRACKING_ROWS = {
    "numbers": "번호 수",
    "as_written": "원래 꼴 그대로",
    "split_found": "묶음이 나뉘었지만 채점 규칙으로 찾음",
    "hangul_digit_names": "한글 숫자 이름 네 자 이상",
    "lost": "찾지 못함",
    "lost_hangul": "  번호 자리에 한글 음절",
    "lost_fewer": "  숫자가 적음",
    "lost_more": "  숫자가 많음",
    "lost_changed": "  숫자 수는 같고 다름",
    "heard_has_에서": '들린 글에 "에서"가 있음',
}


def tracking_forms_table(got: dict[str, dict]) -> str:
    lines = ["| | " + " | ".join(f"`{p}`" for p in got) + " |", "|---|" + "---|" * len(got)]
    for key, label in _TRACKING_ROWS.items():
        cells = [str(len(g["lost"]) if key == "lost" else g[key]) for g in got.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survival", help="gold values still findable in transcript files")
    s.add_argument("data", nargs="+")
    u = sub.add_parser("unfound", help="restored vs invented values of evaluation runs")
    u.add_argument("runs", nargs="+")
    t = sub.add_parser("tracking-forms", help="how the recogniser wrote tracking numbers")
    t.add_argument("data", nargs="+")
    t.add_argument("--show-lost", action="store_true", help="also print every lost number and its heard span")
    for p in (s, u, t):
        p.add_argument("--json", help="also write the numbers here")
    args = ap.parse_args(argv)
    if args.cmd == "survival":
        got = survival(args.data)
        print(survival_table(got))
    elif args.cmd == "tracking-forms":
        got = tracking_forms(args.data)
        print(tracking_forms_table(got))
        if args.show_lost:
            for path, g in got.items():
                print(f"\n{path}")
                for item_id, number, form, span in g["lost"]:
                    print(f"  {item_id} {number} {form}: {span}")
    else:
        got = {str(r): unfound(r) for r in args.runs}
        print(unfound_table(got))
    if args.json:
        Path(args.json).write_text(json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
