"""Speech condition v3: recogniser arms on the same audio, the CPU side (no model, no audio).

Pass 1 (scripts/asr_condition.py --keep-audio) keeps every utterance's WAV and the base recogniser T's text in
outputs/asr3/audio/index.jsonl; pass 2 (asr_condition.py --recognise, scripts/asr_qwen.py) writes one
hypothesis file per arm. This module turns them into datasets and compares the arms (docs/experiments.md
"음성 조건 v3"). An arm whose name ends in "+I" gets the identifier rule (call_summary.id_itn) on every turn;
the others get the recogniser's text as it is. A source is "index" (T's text from pass 1) or a hyp file.

    python -m call_summary.asr_arms context --out outputs/asr3/context.json
    python -m call_summary.asr_arms identity --split dev --data datasets/dev-asr-v2.jsonl
    python -m call_summary.asr_arms gate --split dev --source datasets/dev.jsonl --arm T=index --arm Q+I=H ...
    python -m call_summary.asr_arms build --split dev --source datasets/dev.jsonl --arm Q+I --from H --out F
    python -m call_summary.asr_arms compare --base T=<data> --arm Q+I=<data> ... [--catalog Qc+I ...]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dataset import Item, load_items, read_jsonl, write_jsonl
from .domains import DOMAINS, Domain, catalog
from .id_itn import ITN_VERSION, id_itn
from .prompts import format_transcript
from .scoring import paired_bootstrap_diff, value_survival
from .values import digit_runs, occurs_in_transcript, values_in_transcript

DEFAULT_AUDIO = Path("outputs/asr3/audio")
CONTEXT_VERSION = "ctx-1"
N_BOOT, SEED = 2000, 0
HeardOf = Callable[[dict], str]


# --- context ----------------------------------------------------------------------------------------------


def context_prompt(domain: Domain) -> str:
    """The company name and every catalog value, comma-separated (hotwords for Th, the prompt for Qc)."""
    return ", ".join((domain.company, *catalog(domain)))


def contexts() -> dict[str, str]:
    return {key: context_prompt(d) for key, d in DOMAINS.items()}


# --- sources ----------------------------------------------------------------------------------------------


def uses_itn(arm: str) -> bool:
    return arm.endswith("+I")


def read_index(audio: str | Path = DEFAULT_AUDIO, split: str | None = None) -> list[dict]:
    rows = list(read_jsonl(Path(audio) / "index.jsonl"))
    return [r for r in rows if split is None or r["split"] == split]


def by_turn(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(r["item_id"], r["turn"]): r for r in rows}


def from_index(row: dict) -> str:
    """T's text, recognised in pass 1 from the bytes it kept."""
    return row["heard"]


def from_hyp(hyp: dict[str, dict]) -> HeardOf:
    def heard(row: dict) -> str:
        return hyp[row["key"]]["heard"]

    return heard


def read_hyp(path: str | Path) -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(path)}


def source(spec: str) -> tuple[HeardOf, dict]:
    """`index` or a hyp file -> (text of an index row, what made it)."""
    if spec == "index":
        return from_index, {"source": "index", "model": "large-v3-turbo"}
    meta_path = Path(spec + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.pop("context", None)
    return from_hyp(read_hyp(spec)), {"source": Path(spec).as_posix(), **meta}


def heard_text(arm: str, heard_of: HeardOf, row: dict) -> str:
    text = "" if row["wav"] is None else heard_of(row)
    return id_itn(text) if uses_itn(arm) else text


# --- build ------------------------------------------------------------------------------------------------


def build(
    items: list[Item], index: dict[tuple[str, int], dict], heard_of: HeardOf, itn: bool, meta: dict
) -> list[dict]:
    """Items as the student gets them from one arm: the arm's text (after the identifier rule if `itn`) in
    `turns`, the written text in meta.asr.written. Every turn must be in the kept audio."""
    out = []
    for it in items:
        turns = []
        for i, t in enumerate(it.turns):
            row = index[(it.item_id, i)]
            text = "" if row["wav"] is None else heard_of(row)
            turns.append({"speaker": t["speaker"], "text": id_itn(text) if itn else text})
        d = it.to_dict()
        d["turns"] = turns
        d["split"] = f"{it.split}-asr"
        d["meta"] = {
            **it.meta,
            "asr": {
                **meta,
                "verbalizer": "v2",
                "itn": ITN_VERSION if itn else None,
                "written": [t["text"] for t in it.turns],
            },
        }
        out.append(d)
    return out


# --- identity: T on the new audio against the recorded transcripts ----------------------------------------


def identity(index: dict[tuple[str, int], dict], recorded: list[Item]) -> dict:
    """Turns where T's text on the re-synthesised audio equals the recorded transcript (and where the text
    cache of the same key equals it, a check of the key)."""
    got: dict = {"turns": 0, "same": 0, "different": [], "missing": 0, "cache_same": 0}
    for it in recorded:
        for i, t in enumerate(it.turns):
            got["turns"] += 1
            row = index.get((it.item_id, i))
            if row is None:
                got["missing"] += 1
                continue
            if row["heard"] == t["text"]:
                got["same"] += 1
            else:
                got["different"].append([it.item_id, i, t["text"], row["heard"]])
            got["cache_same"] += row["wav"] is not None and row.get("cached_heard") == t["text"]
    return got


# --- the cheap gate: identifiers in the identifier turns --------------------------------------------------


def _id_values(it: Item) -> list[tuple[str, str]]:
    kinds = {et.label: et.kind for et in DOMAINS[it.domain].entity_types}
    return [(e.type, e.value) for e in it.gold().entities if kinds.get(e.type) == "id"]


def gate(items: list[Item], index: dict[tuple[str, int], dict], arms: dict[str, HeardOf]) -> dict[str, dict]:
    """Per arm, gold identifiers found (scoring's spoken rule) in the text of the kept turns of each item."""
    out: dict[str, dict] = {}
    for arm, heard_of in arms.items():
        types: dict[str, list[int]] = {}
        found = n = 0
        for it in items:
            values = _id_values(it)
            if not values:
                continue
            rows = [index[(it.item_id, i)] for i in range(len(it.turns)) if (it.item_id, i) in index]
            text = format_transcript(
                [{"speaker": r.get("speaker", ""), "text": heard_text(arm, heard_of, r)} for r in rows]
            )
            for t, v in values:
                hit = occurs_in_transcript("id", v, text, spoken=True)
                cell = types.setdefault(t, [0, 0])
                cell[0] += hit
                cell[1] += 1
                found += hit
                n += 1
        out[arm] = {"found": found, "ids": n, "types": types}
    return out


def gate_verdict(got: dict[str, dict], base: str = "T") -> str:
    """`stop` when no candidate finds more gold identifiers than the base arm, else `go`."""
    return "go" if any(g["found"] > got[base]["found"] for a, g in got.items() if a != base) else "stop"


# --- stage A: compare built datasets ----------------------------------------------------------------------


def _cer_norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", unicodedata.normalize("NFKC", text)).upper()


def cer_counts(written: str, heard: str) -> tuple[int, int]:
    """(character edits, reference length) with spaces and punctuation removed, NFKC, Latin upper-cased."""
    ref, hyp = _cer_norm(written), _cer_norm(heard)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


def wrong_digit_runs(written: str, heard: str) -> tuple[int, int]:
    """(heard digit runs of 4+ digits that are no digit run or amount of the written turn, heard runs of 4+
    digits). Amounts count by value, so "36만 1천 원" written and "361,000원" heard agree."""
    said = set(digit_runs(written)) | values_in_transcript("amount", written)
    runs = [r for r in digit_runs(heard) if len(r) >= 4]
    return sum(r not in said for r in runs), len(runs)


_HANGUL_DIGIT_RUN = re.compile(r"[공영일이삼사오육칠팔구]{4,}")


def hangul_digit_runs(text: str) -> int:
    return len(_HANGUL_DIGIT_RUN.findall(text))


def has_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


@dataclass(frozen=True)
class _Flag:
    item_id: str
    value: float


def _mean(xs) -> float:
    return sum(x.value for x in xs) / len(xs) if xs else math.nan


def describe(path: str | Path) -> tuple[dict, list[_Flag]]:
    """Survival and text counts of one built dataset, and its per-item all-values-survive flags."""
    items = load_items(path)
    types: dict[str, list[int]] = {}
    flags = []
    edits = length = wrong = runs = hangul = empty = 0
    for it in items:
        got = value_survival(it.domain, it.gold(), it.transcript, spoken=True)
        for t, found in got.items():
            cell = types.setdefault(t, [0, 0])
            cell[0] += sum(found)
            cell[1] += len(found)
        flags.append(_Flag(it.item_id, float(all(all(v) for v in got.values()))))
        for written, turn in zip(it.meta["asr"]["written"], it.turns, strict=True):
            e, n = cer_counts(written, turn["text"])
            edits, length = edits + e, length + n
            w, r = wrong_digit_runs(written, turn["text"])
            wrong, runs = wrong + w, runs + r
            hangul += hangul_digit_runs(turn["text"])
            empty += not turn["text"].strip() and has_hangul(written)
    stats = {
        "items": len(items),
        "all_found": int(sum(f.value for f in flags)),
        "types": dict(sorted(types.items())),
        "cer": edits / length if length else math.nan,
        "wrong_digit_runs": wrong,
        "digit_runs": runs,
        "hangul_digit_runs": hangul,
        "empty_turns": empty,
    }
    return stats, flags


def compare(
    base: str | Path,
    arms: dict[str, str | Path],
    catalog: set[str] = frozenset(),
    base_name: str = "T",
    reference: set[str] = frozenset(),
) -> dict[str, dict]:
    """Stage A: every arm against the base on the per-item all-values-survive rate (paired bootstrap).
    `reference` arms are reported the same way but never ranked (not candidates)."""
    base_stats, base_flags = describe(base)
    out = {base_name: {**base_stats, "path": str(base)}}
    for order, (arm, path) in enumerate(arms.items()):
        stats, flags = describe(path)
        diff, low, high = paired_bootstrap_diff(base_flags, flags, _mean, n_boot=N_BOOT, seed=SEED)
        out[arm] = {
            **stats,
            "path": str(path),
            "diff": diff,
            "low": low,
            "high": high,
            "passed": low > 0,
            "catalog": arm in catalog,
            "reference": arm in reference,
            "order": order,
        }
    return out


def ranking(got: dict[str, dict]) -> list[str]:
    """Passing candidate arms, best first: higher lower bound, higher difference, non-catalog, registration
    order."""
    passed = [a for a, g in got.items() if g.get("passed") and not g.get("reference")]
    return sorted(passed, key=lambda a: (-got[a]["low"], -got[a]["diff"], got[a]["catalog"], got[a]["order"]))


def _pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.1f}"


def compare_table(got: dict[str, dict]) -> str:
    types = sorted({t for g in got.values() for t in g["types"]})
    head = (
        ["팔", "모든 값이 남은 건", "팔 − T [95%]", "관문"]
        + types
        + [
            "CER",
            "틀린 숫자열",
            "한글 숫자 이름 4자+",
            "빈 발화",
        ]
    )
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm, g in got.items():
        n = g["items"]
        cells = [arm, f"{_pct(g['all_found'] / n)} ({g['all_found']}/{n})"]
        if "diff" in g:
            flag = (" (카탈로그)" if g["catalog"] else "") + (" (참고)" if g["reference"] else "")
            cells += [
                f"{_pct(g['diff'])} [{_pct(g['low'])}, {_pct(g['high'])}]",
                ("통과" if g["passed"] else "못 넘음") + flag,
            ]
        else:
            cells += ["기준", "-"]
        for t in types:
            f, k = g["types"].get(t, [0, 0])
            cells.append(f"{_pct(f / k) if k else '-'} ({f}/{k})")
        cells += [
            _pct(g["cer"]),
            f"{g['wrong_digit_runs']}/{g['digit_runs']}",
            str(g["hangul_digit_runs"]),
            str(g["empty_turns"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- CLI --------------------------------------------------------------------------------------------------


def _pairs(specs: list[str]) -> dict[str, str]:
    out = {}
    for spec in specs:
        name, _, value = spec.partition("=")
        if not value:
            raise SystemExit(f"expected NAME=VALUE, got {spec!r}")
        out[name] = value
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("context", help="per-domain context (company and catalog) as JSON")
    c.add_argument("--out")
    i = sub.add_parser("identity", help="T on the kept audio against the recorded transcripts")
    i.add_argument("--data", required=True, help="recorded transcripts, e.g. datasets/dev-asr-v2.jsonl")
    g = sub.add_parser("gate", help="the cheap gate: gold identifiers in the identifier turns")
    g.add_argument("--source", required=True, help="written items, e.g. datasets/dev.jsonl")
    g.add_argument("--arm", action="append", required=True, help="NAME=index|<hyp file> (T=index first)")
    b = sub.add_parser("build", help="one arm's dataset")
    b.add_argument("--source", required=True)
    b.add_argument("--arm", required=True, help="arm name (+I: the identifier rule)")
    b.add_argument("--from", dest="src", required=True, help="index or a hyp file")
    b.add_argument("--out", required=True)
    m = sub.add_parser("compare", help="stage A: arms against the base on value survival")
    m.add_argument("--base", required=True, help="NAME=<built dataset>")
    m.add_argument(
        "--arm", action="append", required=True, help="NAME=<built dataset>, in registration order"
    )
    m.add_argument("--catalog", action="append", default=[], help="arms that use the catalog context")
    m.add_argument("--reference", action="append", default=[], help="arms reported but never ranked")
    for p in (i, g, b):
        p.add_argument("--split", required=True)
        p.add_argument("--audio", default=str(DEFAULT_AUDIO))
    for p in (i, g, m):
        p.add_argument("--json", help="also write the numbers here")
    args = ap.parse_args(argv)

    if args.cmd == "context":
        text = json.dumps(contexts(), ensure_ascii=False, indent=1)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    if args.cmd == "compare":
        ((base_name, base),) = _pairs([args.base]).items()
        got = compare(base, _pairs(args.arm), set(args.catalog), base_name, set(args.reference))
        print(compare_table(got))
        print("ranking:", " ".join(ranking(got)) or "(none passed)")
    else:
        index = by_turn(read_index(args.audio, args.split))
        if args.cmd == "identity":
            got = identity(index, load_items(args.data))
            print(json.dumps({k: (len(v) if k == "different" else v) for k, v in got.items()}))
        elif args.cmd == "gate":
            arms = {name: source(spec)[0] for name, spec in _pairs(args.arm).items()}
            got = gate(load_items(args.source), index, arms)
            for arm, r in got.items():
                print(f"{arm}: {r['found']}/{r['ids']} {r['types']}")
            print("verdict:", gate_verdict(got))
        else:
            heard_of, meta = source(args.src)
            meta = {**meta, "arm": args.arm}
            rows = build(load_items(args.source), index, heard_of, uses_itn(args.arm), meta)
            write_jsonl(args.out, rows)
            print(f"wrote {len(rows)} items to {args.out}")
            return 0
    if args.json:
        Path(args.json).write_text(json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
