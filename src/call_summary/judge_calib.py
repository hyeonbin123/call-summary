"""Summary-judge calibration against Claude's fact labels (docs/experiments.md "요약 판정 재보정").

  sample  draw the strata from recorded runs and write the sample plus a blind labelling template
  judge   run one judge model over the sample (resumable, one row per sample key)
  report  agreement with Claude per stratum (raw, with the value check, value check alone) and the
          adoption decision on the decision strata

Sampling: one permutation (random.Random(seed)) of the first stratum run's item ids, without the ids of
the hand-labelled strata; the strata take the next ids in their command-line order, skipping an id whose
summary in that stratum's run is empty, so the strata are disjoint. The template rows carry only an opaque
key, the facts and the summary, in an order that mixes the strata.

Agreement is binary per fact (포함 against 누락 or 틀림). A failed judgement counts as 포함 for every fact.
Intervals resample sample items (all facts of an item together), paired for differences.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import Item, load_items, read_jsonl, write_jsonl
from .evaluate import THINK, _git, model_fingerprint, parse_option, run_checks
from .judge import (
    JUDGE_NUM_CTX,
    JUDGE_NUM_PREDICT,
    JUDGE_VERSION,
    VERDICTS,
    cohen_kappa,
    judge_reply,
    make_judge,
    reply_fields,
)
from .value_facts import fact_value_ok, hybrid_verdicts

INCLUDED = VERDICTS[0]
KAPPA_MIN = 0.6  # adoption: kappa on the decision strata at least this
MAX_FAILED = 1  # adoption: at most this many failed judgements on the decision strata


# --------------------------------------------------------------------------------------------- sampling
def _summaries(run: Path) -> dict[str, str]:
    return {
        r["item_id"]: (r.get("pred") or {}).get("summary", "") or "" for r in read_jsonl(run / "items.jsonl")
    }


def _data_path(run: Path) -> str:
    """The run's dataset path with forward slashes (manifests written on Windows hold backslashes)."""
    data = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["data"]
    return Path(data.replace("\\", "/")).as_posix()


def _parse_stratum(text: str) -> tuple[str, Path, int]:
    name, sep, rest = text.partition("=")
    run, sep2, n = rest.rpartition(":")
    if not (sep and sep2 and name and run and n.isdigit()):
        raise argparse.ArgumentTypeError(f"expected NAME=RUN:N, got {text!r}")
    return name, Path(run), int(n)


def _parse_hand(text: str) -> tuple[str, Path, Path]:
    """NAME=HANDFILE; the hand file lies in the run it labels (reports/<run>/hand50.jsonl)."""
    name, sep, path = text.partition("=")
    if not (sep and name and path):
        raise argparse.ArgumentTypeError(f"expected NAME=HANDFILE, got {text!r}")
    return name, Path(path).parent, Path(path)


def build_sample(
    strata: Sequence[tuple[str, Path, int]], hands: Sequence[tuple[str, Path, Path]], seed: int = 0
) -> list[dict]:
    if not strata:
        raise ValueError("at least one stratum")
    first = [r["item_id"] for r in read_jsonl(strata[0][1] / "items.jsonl")]
    summaries = {name: _summaries(run) for name, run, _ in strata}
    for name, _, _ in strata:
        if set(summaries[name]) != set(first):
            raise ValueError(f"stratum {name}: its run does not hold the same items as {strata[0][0]}")
    hand_rows = []
    hand_ids: set[str] = set()
    for name, run, path in hands:
        sums = _summaries(run)
        for h in read_jsonl(path):
            item_id = h["item_id"]
            hand_ids.add(item_id)
            row = {"stratum": name, "run": run.as_posix(), "item_id": item_id, "summary": sums[item_id]}
            hand_rows.append(row)
    perm = [i for i in first if i not in hand_ids]
    random.Random(seed).shuffle(perm)
    rows = []
    pos = 0
    for name, run, n in strata:
        taken = 0
        while taken < n:
            if pos >= len(perm):
                raise ValueError(f"not enough items for stratum {name}")
            item_id = perm[pos]
            pos += 1
            summary = summaries[name][item_id]
            if not summary.strip():
                continue
            rows.append({"stratum": name, "run": run.as_posix(), "item_id": item_id, "summary": summary})
            taken += 1
    order = list(range(len(rows)))
    random.Random(f"{seed}:keys").shuffle(order)
    keyed = [None] * len(rows)
    for k, idx in enumerate(order):
        keyed[k] = {"key": f"k{k + 1:03d}", **rows[idx]}
    keyed_hand = [{"key": f"h{k + 1:03d}", **r} for k, r in enumerate(hand_rows)]
    items: dict[str, dict[str, Item]] = {}
    out = []
    for r in keyed + keyed_hand:
        data = _data_path(Path(r["run"]))
        if data not in items:
            items[data] = {it.item_id: it for it in load_items(data)}
        out.append({**r, "data": data, "facts": list(items[data][r["item_id"]].spec.facts)})
    return out


def template_rows(sample: Sequence[dict]) -> list[dict]:
    """Blind labelling rows (new strata only): no stratum, run or model."""
    return [
        {
            "key": r["key"],
            "facts": r["facts"],
            "summary": r["summary"],
            "verdicts": [],
            "wrong_statements": 0,
            "note": "",
        }
        for r in sample
        if r["key"].startswith("k")
    ]


# --------------------------------------------------------------------------------------------- judging
def loaded_models(provider) -> list[dict] | None:
    try:
        return provider.loaded_models()
    except Exception:  # noqa: BLE001 - only a record; the judgements are already written
        return None


def judge_sample(
    sample_path: Path, out: Path, *, model: str, num_gpu: int | None, think: str, options: dict
) -> dict:
    sample = list(read_jsonl(sample_path))
    provider = make_judge(model=model, num_gpu=num_gpu, think=THINK[think], options=options)
    done = {d["key"] for d in read_jsonl(out)} if out.exists() else set()
    items: dict[str, dict[str, Item]] = {}
    started = time.strftime("%Y%m%d-%H%M%S")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as f:
        for n, r in enumerate(sample, 1):
            if r["key"] in done:
                continue
            if r["data"] not in items:
                items[r["data"]] = {it.item_id: it for it in load_items(r["data"])}
            it = items[r["data"]][r["item_id"]]
            score, reply = judge_reply(provider, it.item_id, it.spec, it.transcript, r["summary"])
            row = {"key": r["key"], "item_id": r["item_id"], **score.to_dict(), **reply_fields(reply)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(sample):
                print(f"[{n}/{len(sample)}] {model}", file=sys.stderr, flush=True)
    rows = list(read_jsonl(out))
    meta = {
        "model": model,
        "judge_version": JUDGE_VERSION,
        "num_gpu": num_gpu,
        "think": think,
        "options": options,
        "num_ctx": JUDGE_NUM_CTX,
        "num_predict": JUDGE_NUM_PREDICT,
        "temperature": getattr(provider, "temperature", None),
        "seed": getattr(provider, "seed", None),
        "fingerprint": model_fingerprint(provider),
        "git_commit": _git("rev-parse", "HEAD"),
        "sample": str(sample_path),
        "started": started,
        "finished": time.strftime("%Y%m%d-%H%M%S"),
        "n": len(rows),
        "n_failed": sum(1 for d in rows if not d["ok"]),
        "run_checks": run_checks(rows, JUDGE_NUM_CTX),
        "loaded_models_after": loaded_models(provider),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------------------------- agreement
@dataclass(frozen=True)
class Rater:
    name: str
    verdicts: dict[str, tuple[str, ...]]  # sample key -> one verdict per fact; empty = failed judgement
    failed: frozenset[str] = field(default_factory=frozenset)

    def binary(self, key: str, n: int) -> list[bool]:
        v = self.verdicts.get(key, ())
        if not v:
            return [True] * n
        if len(v) != n:
            raise ValueError(f"{self.name}: {key} has {len(v)} verdicts for {n} facts")
        return [x == INCLUDED for x in v]

    def is_failed(self, key: str) -> bool:
        return key in self.failed or not self.verdicts.get(key)


def _pairs(claude: Rater, rater: Rater, keys: Sequence[str]) -> list[tuple[list[bool], list[bool]]]:
    out = []
    for k in keys:
        c = claude.binary(k, len(claude.verdicts[k]))
        out.append((c, rater.binary(k, len(c))))
    return out


def _kappa_of(pairs: Sequence[tuple[list[bool], list[bool]]]) -> float:
    a = [x for c, _ in pairs for x in c]
    b = [x for _, r in pairs for x in r]
    return cohen_kappa(a, b)


def _ci(stats: list[float]) -> tuple[float, float]:
    stats = sorted(v for v in stats if v == v)
    if not stats:
        return math.nan, math.nan
    return stats[int(0.025 * (len(stats) - 1))], stats[int(0.975 * (len(stats) - 1))]


def kappa_table(claude: Rater, rater: Rater, keys: Sequence[str], n_boot: int = 2000, seed: int = 0) -> dict:
    pairs = _pairs(claude, rater, keys)
    errors = [(c, r) for cs, rs in pairs for c, r in zip(cs, rs, strict=True) if not c]
    included = [(c, r) for cs, rs in pairs for c, r in zip(cs, rs, strict=True) if c]
    rng = random.Random(seed)
    k = len(pairs)
    boot = [_kappa_of([pairs[rng.randrange(k)] for _ in range(k)]) for _ in range(n_boot)] if k else []
    return {
        "rater": rater.name,
        "n_items": k,
        "n_facts": len(errors) + len(included),
        "n_errors": len(errors),
        "n_failed": sum(1 for key in keys if rater.is_failed(key)),
        "kappa": _kappa_of(pairs),
        "kappa_ci": _ci(boot),
        "error_recall": sum(1 for _, r in errors if not r) / len(errors) if errors else math.nan,
        "false_error_rate": sum(1 for _, r in included if not r) / len(included) if included else math.nan,
    }


def paired_kappa_diff(
    claude: Rater, a: Rater, b: Rater, keys: Sequence[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """kappa(a) - kappa(b) against Claude, with a CI from resampling the same items for both."""
    pa, pb = _pairs(claude, a, keys), _pairs(claude, b, keys)
    rng = random.Random(seed)
    k = len(keys)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        stats.append(_kappa_of([pa[i] for i in idx]) - _kappa_of([pb[i] for i in idx]))
    lo, hi = _ci(stats)
    return _kappa_of(pa) - _kappa_of(pb), lo, hi


def decide(stats: dict[str, dict]) -> dict:
    """The adoption rule over the candidates in registration order (later = other family on ties): kappa
    at least KAPPA_MIN, the paired lower bound against the baseline judge above 0, not below the baseline
    judge with the same value check, at most MAX_FAILED failed judgements, no judgement that filled the
    context and none with thinking text."""
    reasons: dict[str, list[str]] = {}
    passed = []
    for i, (name, s) in enumerate(stats.items()):
        why = []
        if not (s["kappa"] >= KAPPA_MIN):  # nan fails
            why.append(f"kappa {s['kappa']:.3f} < {KAPPA_MIN}")
        if not (s["diff_vs_baseline"][1] > 0):
            why.append(f"lower bound vs the baseline {s['diff_vs_baseline'][1]:.3f} <= 0")
        if not (s["diff_vs_same_check"] >= 0):
            why.append(
                f"kappa below the baseline judge with the same value check ({s['diff_vs_same_check']:.3f})"
            )
        if s["n_failed"] > MAX_FAILED:
            why.append(f"{s['n_failed']} failed judgements")
        if s["context_full"] != 0:
            why.append(f"context_full {s['context_full']}")
        if s["thinking_items"] != 0:
            why.append(f"{s['thinking_items']} judgements came with thinking text")
        reasons[name] = why
        if not why:
            recall = s["error_recall"] if s["error_recall"] == s["error_recall"] else -1.0
            passed.append(((s["kappa"], recall, i), name))
    return {"adopted": max(passed)[1] if passed else None, "reasons": reasons}


# --------------------------------------------------------------------------------------------- report
def _load_labels(sample: Sequence[dict], labels: Path | None, hand: dict[str, Path]) -> Rater:
    verdicts: dict[str, tuple[str, ...]] = {}
    if labels is not None:
        for d in read_jsonl(labels):
            verdicts[d["key"]] = tuple(d["verdicts"])
    for stratum, path in hand.items():
        by_item = {d["item_id"]: tuple(d["verdicts"]) for d in read_jsonl(path)}
        for r in sample:
            if r["stratum"] == stratum:
                verdicts[r["key"]] = by_item[r["item_id"]]
    for r in sample:
        v = verdicts.get(r["key"])
        if v is None or len(v) != len(r["facts"]) or any(x not in VERDICTS for x in v):
            raise ValueError(
                f"Claude label for {r['key']} is missing or does not match its {len(r['facts'])} facts"
            )
    return Rater("claude", verdicts)


def _load_judge(name: str, path: Path, sample: Sequence[dict]) -> tuple[Rater, dict[str, dict]]:
    """Judge rows keyed by sample key. NAME@STRATUM reads rows keyed by item id (a recorded judge.jsonl of
    the stratum's run) for that stratum only."""
    name, _, stratum = name.partition("@")
    rows_in = list(read_jsonl(path))
    if stratum:
        by_item = {d["item_id"]: d for d in rows_in}
        rows = {r["key"]: by_item[r["item_id"]] for r in sample if r["stratum"] == stratum}
    else:
        rows = {d["key"]: d for d in rows_in}
    failed = frozenset(k for k, d in rows.items() if not d.get("ok", True))
    return Rater(name, {k: tuple(d["verdicts"]) for k, d in rows.items()}, failed), rows


def _value_ok(sample: Sequence[dict]) -> dict[str, list[bool | None]]:
    items: dict[str, dict[str, Item]] = {}
    out = {}
    for r in sample:
        if r["data"] not in items:
            items[r["data"]] = {it.item_id: it for it in load_items(r["data"])}
        out[r["key"]] = fact_value_ok(items[r["data"]][r["item_id"]].spec, r["summary"])
    return out


def _hybrid(rater: Rater, value_ok: dict[str, list[bool | None]]) -> Rater:
    keys = [k for k in value_ok if k in rater.verdicts]
    return Rater(
        f"V+{rater.name}",
        {k: hybrid_verdicts(rater.verdicts[k], value_ok[k]) for k in keys},
        frozenset(k for k in keys if rater.is_failed(k)),
    )


def report(
    sample_path: Path,
    labels: Path | None,
    hand: dict[str, Path],
    judges: dict[str, Path],
    baseline: str,
    candidates: Sequence[str],
    decide_on: Sequence[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    sample = list(read_jsonl(sample_path))
    claude = _load_labels(sample, labels, hand)
    value_ok = _value_ok(sample)
    raters: list[Rater] = [Rater("V", {k: hybrid_verdicts((), ok) for k, ok in value_ok.items()})]
    rows_of: dict[str, dict[str, dict]] = {}
    for name, path in judges.items():
        r, rows = _load_judge(name, path, sample)
        rows_of[r.name] = rows
        raters += [r, _hybrid(r, value_ok)]
    by_name = {r.name: r for r in raters}
    groups: dict[str, list[str]] = {}
    for r in sample:
        groups.setdefault(r["stratum"], []).append(r["key"])
    groups["decision:" + "+".join(decide_on)] = [r["key"] for r in sample if r["stratum"] in decide_on]
    groups["all"] = [r["key"] for r in sample]
    tables: dict[str, list[dict]] = {}
    for g, keys in groups.items():
        tables[g] = [
            kappa_table(claude, r, keys, n_boot, seed) for r in raters if all(k in r.verdicts for k in keys)
        ]
    dkeys = groups["decision:" + "+".join(decide_on)]
    base = by_name[baseline]
    base_h = by_name[f"V+{baseline}"]
    stats = {}
    for name in candidates:
        h = by_name[f"V+{name}"]
        t = kappa_table(claude, h, dkeys, n_boot, seed)
        checks = run_checks(list(rows_of[name].values()), JUDGE_NUM_CTX)  # every judged sample row
        stats[h.name] = {
            "kappa": t["kappa"],
            "kappa_ci": t["kappa_ci"],
            "error_recall": t["error_recall"],
            "diff_vs_baseline": paired_kappa_diff(claude, h, base, dkeys, n_boot, seed),
            "diff_vs_same_check": t["kappa"] - kappa_table(claude, base_h, dkeys, 0, seed)["kappa"],
            "diff_vs_same_check_ci": paired_kappa_diff(claude, h, base_h, dkeys, n_boot, seed),
            "n_failed": t["n_failed"],
            "context_full": checks["context_full"],
            "length_stops": checks["length_stops"],
            "thinking_items": checks["thinking_items"],
        }
    return {"tables": tables, "candidates": stats, "decision": decide(stats), "baseline": baseline}


def _f(x: float) -> str:
    return "-" if x != x else f"{x:.3f}"


def _print_report(rep: dict) -> None:
    for g, rows in rep["tables"].items():
        print(f"\n## {g}")
        print(
            "rater | items | facts | Claude errors | kappa [95%] | error recall | false error rate | failed"
        )
        for t in rows:
            lo, hi = t["kappa_ci"]
            print(
                f"{t['rater']} | {t['n_items']} | {t['n_facts']} | {t['n_errors']} | {_f(t['kappa'])} "
                f"[{_f(lo)}, {_f(hi)}] | {_f(t['error_recall'])} | {_f(t['false_error_rate'])} | "
                f"{t['n_failed']}"
            )
    print(f"\n## candidates (baseline {rep['baseline']})")
    for name, s in rep["candidates"].items():
        d, lo, hi = s["diff_vs_baseline"]
        print(
            f"{name}: kappa {_f(s['kappa'])}, vs baseline {_f(d)} [{_f(lo)}, {_f(hi)}], "
            f"vs baseline with value check {_f(s['diff_vs_same_check'])}, failed {s['n_failed']}, "
            f"context_full {s['context_full']}, reasons: {rep['decision']['reasons'][name] or 'none'}"
        )
    print(f"adopted: {rep['decision']['adopted']}")


def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    out = {}
    for v in values:
        name, sep, path = v.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=PATH, got {v!r}")
        out[name] = Path(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the strata and write the sample and the labelling template")
    s.add_argument("--stratum", type=_parse_stratum, action="append", required=True, help="NAME=RUN:N")
    s.add_argument(
        "--hand", type=_parse_hand, action="append", default=[], help="NAME=reports/<run>/<hand labels>.jsonl"
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--template", required=True)
    j = sub.add_parser("judge", help="run one judge model over the sample")
    j.add_argument("--sample", required=True)
    j.add_argument("--model", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--num-gpu", type=int)
    j.add_argument("--think", choices=sorted(THINK), default="off")
    j.add_argument("--option", type=parse_option, action="append", default=[], metavar="KEY=VALUE")
    r = sub.add_parser("report", help="agreement tables and the adoption decision")
    r.add_argument("--sample", required=True)
    r.add_argument("--labels", help="Claude labels by sample key (the filled template)")
    r.add_argument("--hand-labels", action="append", default=[], help="STRATUM=FILE, labels by item id")
    r.add_argument(
        "--judge",
        action="append",
        default=[],
        help="NAME=FILE (judge rows by sample key); NAME@STRATUM=FILE for rows by item id of that stratum",
    )
    r.add_argument("--baseline", required=True, help="the judge name the candidates must beat")
    r.add_argument("--candidates", required=True, help="comma-separated judge names, in registration order")
    r.add_argument("--decide-on", required=True, help="comma-separated strata the decision uses")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--json")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    if args.cmd == "sample":
        sample = build_sample(args.stratum, args.hand, args.seed)
        write_jsonl(args.out, sample)
        write_jsonl(args.template, template_rows(sample))
        counts: dict[str, int] = {}
        for row in sample:
            counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
        print(json.dumps({"rows": len(sample), "strata": counts}, ensure_ascii=False))
    elif args.cmd == "judge":
        meta = judge_sample(
            Path(args.sample),
            Path(args.out),
            model=args.model,
            num_gpu=args.num_gpu,
            think=args.think,
            options=dict(args.option),
        )
        print(json.dumps({k: meta[k] for k in ("model", "n", "n_failed", "run_checks")}, ensure_ascii=False))
    else:
        rep = report(
            Path(args.sample),
            Path(args.labels) if args.labels else None,
            _named_paths(args.hand_labels),
            _named_paths(args.judge),
            args.baseline,
            args.candidates.split(","),
            args.decide_on.split(","),
            args.n_boot,
        )
        _print_report(rep)
        if args.json:
            Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
