"""Claude의 사실 라벨로 요약 판정을 보정한다(docs/experiments.md "요약 판정 재보정").

  sample  기록된 실행에서 층을 뽑아 표본과, 누구 것인지 가린 라벨 양식을 쓴다
  judge   표본 전체에 판정 모델 하나를 돌린다(이어 하기 가능, 표본 열쇠마다 한 줄)
  report  층마다 Claude와의 일치(판정 그대로, 값 확인을 더한 것, 값 확인 혼자)와 채택 층에서의
          채택 판정
  cells   값 확인 혼자를 Claude와 비교한 셀별 사실 수와, 둘이 갈리는 사실

표본 뽑기: 첫 층 실행의 건 id를 한 번 섞고(random.Random(seed)) 사람 라벨 층의 id는 뺀다. 층들은 명령줄
순서대로 다음 id를 가져가되, 그 층 실행의 요약이 빈 id는 건너뛰므로 층끼리 겹치지 않는다. 양식 행에는
속을 알 수 없는 열쇠, 사실 목록, 요약만 들어가고, 층이 섞인 순서로 놓인다.

일치는 사실마다 이진값이다(포함 대 누락 또는 틀림). 판정이 실패한 건은 모든 사실을 포함으로 친다.
구간은 표본 건(한 건의 사실을 함께) 단위로 다시 뽑고, 차이는 짝지어 낸다.
"""

# [파일 역할] 요약 판정 재보정(2026-10-08 규칙)의 도구다. 기록된 실행에서 보정 표본을 층별로 뽑고(sample),
# 판정 모델을 표본에 돌리고(judge), Claude 라벨과의 일치(kappa, 오류 재현율)와 채택 판정을 내고(report), 값
# 확인 혼자의 셀 표를 만든다(cells).
# [왜 필요한가] 2단계 판정 모델(j1, qwen2.5 14B)은 사실 일치율 96.3%였지만 kappa가 약 0.32라 요약 지표를
# 참고치로만 썼다. 재보정은 (1) 판정 모델 없이 사실의 숫자 값을 확인하는 값 확인 V를 더하고 (2) 2026년 판정
# 모델을 Claude의 새 라벨로 보정해, 쓸 만한 판정 형태가 있는지 미리 정한 규칙으로 고른다(docs/experiments.md
# "요약 판정 재보정").
# [설계 이유] 라벨이 판정 모델 결과에 끌려가지 않도록, 라벨 양식에는 열쇠·사실·요약만 넣고 층과 모델을 숨기며
# 층을 섞은 순서로 낸다. 표본은 dev에서만 뽑아 판정 모델을 고르는 데 test를 쓰지 않는다. 채택은 서빙 학생의
# 층(S-asr, S-dev)으로만 정한다.
# [결과] V+J1(qwen3.5:9b + 값 확인)이 kappa 0.603으로 채택됐다. 다만 그 대부분은 값 확인 몫이다(V 혼자 0.556,
# J1 혼자 0.149, README 5단계).

from __future__ import annotations

# random: seed를 받은 random.Random만 써서 표본과 열쇠 순서, 부트스트랩이 늘 같게 나온다.
import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# judge.py의 요청 꼴(make_judge, judge_reply)과 kappa 계산을 그대로 쓰고, evaluate.py의 실행 조건 기록
# 도우미를 쓴다. value_facts의 fact_value_ok(사실마다 숫자 값이 요약에 있는지)와 hybrid_verdicts(판정에 값
# 확인을 덮기)로 V와 V+J를 만든다.
from .dataset import Item, load_items, read_jsonl, write_jsonl
from .evaluate import THINK, _git, loaded_models, model_fingerprint, parse_option, run_checks
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

# INCLUDED: '포함' 판정. 일치는 사실마다 포함 대 비포함(누락·틀림)의 이진값으로 본다.
# KAPPA_MIN, MAX_FAILED: 채택 조건 (a) 채택 층 kappa ≥ 0.6, (d) 채택 층 판정 실패 1건 이하(규칙 4).
INCLUDED = VERDICTS[0]
KAPPA_MIN = 0.6  # 채택: 채택 층의 kappa가 이 값 이상
MAX_FAILED = 1  # 채택: 채택 층의 판정 실패가 이 수 이하


# ------------------------------------------------------------------------------------------ 표본 뽑기(sample)
# 실행의 건별 예측 요약(예측이 없으면 빈 문자열).
def _summaries(run: Path) -> dict[str, str]:
    return {
        r["item_id"]: (r.get("pred") or {}).get("summary", "") or "" for r in read_jsonl(run / "items.jsonl")
    }


# 실행 manifest의 데이터 경로를 슬래시 꼴로 바꾼다. Windows에서 쓴 manifest에는 역슬래시가 들어 있어 표본
# 파일의 경로를 한 가지 꼴로 통일한다.
def _data_path(run: Path) -> str:
    """실행의 데이터셋 경로를 슬래시 꼴로(Windows에서 쓴 manifest에는 역슬래시가 들어 있다)."""
    data = json.loads((run / "manifest.json").read_text(encoding="utf-8"))["data"]
    return Path(data.replace("\\", "/")).as_posix()


# --stratum NAME=RUN:N 인자를 (층 이름, 실행 폴더, 건수)로. 경로에 콜론이 있을 수 있어 마지막 콜론으로
# 나눈다(rpartition).
def _parse_stratum(text: str) -> tuple[str, Path, int]:
    name, sep, rest = text.partition("=")
    run, sep2, n = rest.rpartition(":")
    if not (sep and sep2 and name and run and n.isdigit()):
        raise argparse.ArgumentTypeError(f"expected NAME=RUN:N, got {text!r}")
    return name, Path(run), int(n)


# --hand NAME=HANDFILE: 사람(2단계 Claude) 라벨 파일. 파일이 라벨을 매긴 실행 폴더 안에 있으므로 그 부모를
# 실행으로 본다.
def _parse_hand(text: str) -> tuple[str, Path, Path]:
    """NAME=HANDFILE. 사람 라벨 파일은 그것이 라벨을 매긴 실행 안에 있다(reports/<run>/hand50.jsonl)."""
    name, sep, path = text.partition("=")
    if not (sep and name and path):
        raise argparse.ArgumentTypeError(f"expected NAME=HANDFILE, got {text!r}")
    return name, Path(path).parent, Path(path)


# 보정 표본을 만든다. 첫 층 실행의 건 id에서 사람 라벨 층(hand50)의 건을 빼고 seed로 한 번 섞은 뒤, 층마다
# 명령줄 순서대로 다음 건을 가져간다. 그 층의 요약이 빈 건은 건너뛴다. 그래서 층끼리 건이 겹치지 않아 같은
# 통화의 글 전사 요약과 음성 전사 요약이 함께 들어가지 않는다(규칙 2).
# 모든 층 실행이 같은 건 집합을 가져야 한다(dev 240건). 새 층 행은 "k001"처럼 섞인 순서의 열쇠를, 사람 라벨
# 행은 "h001" 열쇠를 받고, 행마다 데이터 경로와 명세의 사실 목록을 붙인다.
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
    # 열쇠 순서를 별도 seed("<seed>:keys")로 섞어, 열쇠 번호만 보고는 층을 짐작할 수 없게 한다(블라인드
    # 라벨링).
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


# Claude가 채울 라벨 양식: 새 층(k 열쇠) 행만, 열쇠·사실·요약과 빈 판정 칸만 넣는다. 층, 실행, 모델 이름은
# 넣지 않아 누구의 요약인지 모른 채 채점하게 한다. 대화도 넣지 않는다(2단계 hand50과 같은 기준: 사실 목록과
# 요약만 본다).
def template_rows(sample: Sequence[dict]) -> list[dict]:
    """누구 것인지 가린 라벨 양식 행(새 층만): 층, 실행, 모델을 넣지 않는다."""
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


# --------------------------------------------------------------------------------------------- 판정(judge)
# 판정 모델 하나를 표본 전체에 돌린다. 이어 하기가 되어 이미 판정한 열쇠는 건너뛰고, 한 건마다 한 줄 쓰고
# flush한다. 끝나면 <출력>.meta.json에 모델·판·요청 설정(num_ctx, num_predict, temperature, seed), 모델 지문,
# 커밋, 실패 수, run_checks, 판정 뒤 /api/ps를 남긴다. run_checks는 채택 조건 (d)(컨텍스트를 채운 판정 0, 생각
# 글 0)를 확인하는 데 쓴다.
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


# --------------------------------------------------------------------------------------------- 일치
# 채점자 하나(Claude, 판정 모델, V, V+J). verdicts는 표본 열쇠 → 사실별 판정이고, 비어 있으면 판정 실패다.
@dataclass(frozen=True)
class Rater:
    name: str
    verdicts: dict[str, tuple[str, ...]]  # 표본 열쇠 -> 사실마다 판정 하나. 비면 판정 실패
    failed: frozenset[str] = field(default_factory=frozenset)

    # 열쇠 하나의 판정을 포함(True) / 비포함(False) 목록으로. 판정 실패는 모든 사실을 포함으로 친다(규칙 4).
    # 그러면 실패한 판정은 오류를 하나도 잡지 못한 것으로 계산된다. 사실 수가 맞지 않으면 멈춘다.
    def binary(self, key: str, n: int) -> list[bool]:
        v = self.verdicts.get(key, ())
        if not v:
            return [True] * n
        if len(v) != n:
            raise ValueError(f"{self.name}: {key} has {len(v)} verdicts for {n} facts")
        return [x == INCLUDED for x in v]

    # 판정 실패인지(실패로 표시됐거나 판정이 비어 있음).
    def is_failed(self, key: str) -> bool:
        return key in self.failed or not self.verdicts.get(key)


# 열쇠마다 (Claude의 이진 판정, 비교 채점자의 이진 판정) 쌍. 건 단위로 묶어 두어 부트스트랩이 건의 사실을
# 한꺼번에 다시 뽑게 한다.
def _pairs(claude: Rater, rater: Rater, keys: Sequence[str]) -> list[tuple[list[bool], list[bool]]]:
    out = []
    for k in keys:
        c = claude.binary(k, len(claude.verdicts[k]))
        out.append((c, rater.binary(k, len(c))))
    return out


# 쌍 목록의 모든 사실을 펼쳐 Cohen's kappa를 낸다(Claude가 기준).
def _kappa_of(pairs: Sequence[tuple[list[bool], list[bool]]]) -> float:
    a = [x for c, _ in pairs for x in c]
    b = [x for _, r in pairs for x in r]
    return cohen_kappa(a, b)


# 부트스트랩 값의 2.5%·97.5% 위치(보간 없음)를 95% 구간으로. kappa가 정의되지 않은 표본(NaN)은 뺀다.
def _ci(stats: list[float]) -> tuple[float, float]:
    stats = sorted(v for v in stats if v == v)
    if not stats:
        return math.nan, math.nan
    return stats[int(0.025 * (len(stats) - 1))], stats[int(0.975 * (len(stats) - 1))]


# Claude 대비 한 채점자의 일치 표: 건 수, 사실 수, Claude가 본 오류 수, 판정 실패 수, kappa와 건 단위
# 부트스트랩 95% 구간, 오류 재현율(Claude가 비포함으로 본 사실 중 채점자도 비포함으로 본 비율), 거짓
# 오류율(Claude가 포함으로 본 사실 중 채점자가 비포함으로 본 비율).
# 2단계처럼 사실 대부분이 '포함'이면 일치율은 높게 나오므로, 오류를 잡는 능력을 따로 보려고 오류 재현율을 함께
# 낸다.
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


# 두 채점자 a, b의 kappa 차이(a − b)와, 같은 건을 양쪽에 똑같이 다시 뽑은 짝지은 부트스트랩 95% 구간. 채택
# 조건 (b)(지금의 j1 그대로보다 하한 > 0)와 (c)(값 확인을 붙인 옛 판정 모델보다 낮지 않음)에 쓴다.
def paired_kappa_diff(
    claude: Rater, a: Rater, b: Rater, keys: Sequence[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Claude 기준 kappa(a) - kappa(b)와, 양쪽에 같은 건을 다시 뽑아 낸 구간."""
    pa, pb = _pairs(claude, a, keys), _pairs(claude, b, keys)
    rng = random.Random(seed)
    k = len(keys)
    stats = []
    for _ in range(n_boot):
        idx = [rng.randrange(k) for _ in range(k)]
        stats.append(_kappa_of([pa[i] for i in idx]) - _kappa_of([pb[i] for i in idx]))
    lo, hi = _ci(stats)
    return _kappa_of(pa) - _kappa_of(pb), lo, hi


# 채택 규칙을 후보마다 적용하고 통과하지 못한 이유를 남긴다. 조건: (a) kappa ≥ 0.6, (b) 기준 판정(J0 그대로)
# 대비 짝지은 하한 > 0, (c) 같은 값 확인을 붙인 기준(V+J0)보다 kappa가 낮지 않음(값 확인만으로 생긴 이득으로
# 판정 모델을 바꾸지 않으려는 조건), (d) 실패 1건 이하, 컨텍스트를 채운 판정 0, 생각 글 0.
# 여럿이 통과하면 (kappa, 오류 재현율, 등록 순서)가 가장 큰 것을 고른다. 등록 순서가 뒤인 J2(학생과 다른
# 계열인 Gemma)가 끝까지 같을 때 이기도록 한 것이다(규칙 4).
def decide(stats: dict[str, dict]) -> dict:
    """등록 순서대로의 후보에 채택 규칙을 적용한다(동률이면 뒤의 것 = 다른 계열): kappa가 KAPPA_MIN
    이상, 기준 판정 대비 짝지은 하한이 0보다 큼, 같은 값 확인을 붙인 기준 판정보다 낮지 않음, 판정 실패가
    MAX_FAILED 이하, 컨텍스트를 채운 판정과 생각 글이 붙은 판정이 없음."""
    reasons: dict[str, list[str]] = {}
    passed = []
    for i, (name, s) in enumerate(stats.items()):
        why = []
        # not (x >= 기준) 꼴로 써서 NaN이면 비교가 False가 되어 조건에 걸리게 한다.
        if not (s["kappa"] >= KAPPA_MIN):  # nan이면 실패
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
            # 오류 재현율이 NaN이면 -1로 바꿔 동률 비교에서 가장 낮게 친다.
            recall = s["error_recall"] if s["error_recall"] == s["error_recall"] else -1.0
            passed.append(((s["kappa"], recall, i), name))
    return {"adopted": max(passed)[1] if passed else None, "reasons": reasons}


# --------------------------------------------------------------------------------------------- 보고(report)
# Claude 라벨을 모은다: 새 라벨 파일(열쇠 기준)과 사람 라벨 층 파일(건 id 기준, hand50)을 표본 열쇠로 맞춘다.
# 모든 표본 행에 사실 수만큼 올바른 판정이 있어야 하며, 빠지면 바로 멈춘다(라벨이 덜 채워진 채 보고서가 나가지
# 않게).
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


# 판정 결과 파일을 Rater로 읽는다. 이름이 NAME@STRATUM이면 그 층의 실행에 기록된 judge.jsonl(건 id 기준)을 그
# 층 열쇠에만 맞춰 읽는다. 2단계에 기록된 j1 판정(J0rec@H)을 같은 조건으로 다시 돈 J0와 견주는 데 썼다(규칙
# 3).
def _load_judge(name: str, path: Path, sample: Sequence[dict]) -> tuple[Rater, dict[str, dict]]:
    """표본 열쇠로 찾는 판정 행. NAME@STRATUM이면 건 id로 찾는 행(그 층 실행에 기록된 judge.jsonl)을
    그 층에만 맞춰 읽는다."""
    name, _, stratum = name.partition("@")
    rows_in = list(read_jsonl(path))
    if stratum:
        by_item = {d["item_id"]: d for d in rows_in}
        rows = {r["key"]: by_item[r["item_id"]] for r in sample if r["stratum"] == stratum}
    else:
        rows = {d["key"]: d for d in rows_in}
    failed = frozenset(k for k, d in rows.items() if not d.get("ok", True))
    return Rater(name, {k: tuple(d["verdicts"]) for k, d in rows.items()}, failed), rows


# 표본 행마다 값 확인 결과(사실마다 None: 숫자 값 없음, True: 값이 모두 요약에 있음, False: 빠짐)를 계산한다.
# 판정 모델 없이 채점과 같은 정규화로 센다.
def _value_ok(sample: Sequence[dict]) -> dict[str, list[bool | None]]:
    items: dict[str, dict[str, Item]] = {}
    out = {}
    for r in sample:
        if r["data"] not in items:
            items[r["data"]] = {it.item_id: it for it in load_items(r["data"])}
        out[r["key"]] = fact_value_ok(items[r["data"]][r["item_id"]].spec, r["summary"])
    return out


# V+J: 판정 모델의 판정에 값 확인을 덮은 채점자. 값이 빠진 사실의 '포함'만 '누락'으로 바꾸고 '틀림'은 둔다.
# 판정 실패 표시는 그대로 옮긴다.
def _hybrid(rater: Rater, value_ok: dict[str, list[bool | None]]) -> Rater:
    keys = [k for k in value_ok if k in rater.verdicts]
    return Rater(
        f"V+{rater.name}",
        {k: hybrid_verdicts(rater.verdicts[k], value_ok[k]) for k in keys},
        frozenset(k for k in keys if rater.is_failed(k)),
    )


# 보정 보고서: 채점자 V(값 확인만), 판정 모델마다 J(그대로)와 V+J를 만들고, 층별·채택 층·전체 표를 낸 뒤
# 후보마다 채택 통계를 모아 decide에 넘긴다.
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
    # V 혼자는 빈 판정에서 시작해(모두 포함) 값이 빠진 사실만 비포함으로 만든 채점자다.
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
    # 그룹의 모든 열쇠에 판정이 있는 채점자만 그 그룹 표에 넣는다(J0rec처럼 한 층에만 있는 채점자는 그 층
    # 표에만 나온다).
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
        # (d)의 컨텍스트·생각 글 검사는 채택 층만이 아니라 판정한 표본 행 전체로 센다(규칙: "보정 세트
        # 전체에서").
        checks = run_checks(list(rows_of[name].values()), JUDGE_NUM_CTX)  # 판정한 표본 행 전체
        stats[h.name] = {
            "kappa": t["kappa"],
            "kappa_ci": t["kappa_ci"],
            "error_recall": t["error_recall"],
            "diff_vs_baseline": paired_kappa_diff(claude, h, base, dkeys, n_boot, seed),
            # (c)는 점 추정끼리 비교한다(n_boot 0으로 점만 얻음). 구간은 참고로 diff_vs_same_check_ci에 따로
            # 남긴다.
            "diff_vs_same_check": t["kappa"] - kappa_table(claude, base_h, dkeys, 0, seed)["kappa"],
            "diff_vs_same_check_ci": paired_kappa_diff(claude, h, base_h, dkeys, n_boot, seed),
            "n_failed": t["n_failed"],
            "context_full": checks["context_full"],
            "length_stops": checks["length_stops"],
            "thinking_items": checks["thinking_items"],
        }
    return {"tables": tables, "candidates": stats, "decision": decide(stats), "baseline": baseline}


# ------------------------------------------------------------------------------------------ 값 확인 셀(cells)
# 값 확인 V와 Claude 라벨을 사실마다 견준 셀 수: 값을 찾음/빠짐/값 없음 × Claude 포함/비포함. 어긋난 사실(V는
# 빠졌다는데 Claude는 포함 = V의 거짓 누락, V는 찾았는데 Claude는 비포함)을 "열쇠/F번호"로 남긴다.
def value_cells(claude: Rater, value_ok: dict[str, list[bool | None]], keys: Sequence[str]) -> dict:
    """값 확인 혼자를 Claude와 사실마다 견준다: 값 찾음/빠짐/값 없음 × Claude 포함/비포함, 그리고 둘이
    갈리는 사실(key/F<n>)."""
    cells = dict.fromkeys(("found_incl", "found_not", "miss_incl", "miss_not", "noval_not", "noval_incl"), 0)
    miss_incl: list[str] = []
    found_not: list[str] = []
    for k in keys:
        for i, (ok, c) in enumerate(zip(value_ok[k], claude.verdicts[k], strict=True)):
            inc = c == INCLUDED
            if ok is None:
                cells["noval_incl" if inc else "noval_not"] += 1
            elif ok:
                cells["found_incl" if inc else "found_not"] += 1
                if not inc:
                    found_not.append(f"{k}/F{i + 1}:{c}")
            else:
                cells["miss_incl" if inc else "miss_not"] += 1
                if inc:
                    miss_incl.append(f"{k}/F{i + 1}")
    return {"cells": cells, "v_miss_but_claude_included": miss_incl, "v_found_but_claude_not": found_not}


# cells 하위 명령: 층마다(표본에 처음 나온 순서), 그리고 --group으로 이름 붙인 층 묶음마다 V의 kappa 표와 셀
# 수를 낸다. 묶음의 열쇠는 지정한 층 순서대로 이어 붙이므로 부트스트랩이 그 순서로 다시 뽑는다.
# 검토에서 "값 확인 V와 Claude 라벨" 표가 커밋하지 않은 확인 스크립트로 만들어진 것을 찾아, 같은 계산을 커밋한
# 도구로 옮긴 것이다(커밋 0103ac9, docs/experiments.md 2026-10-09 바로잡음).
def cells_report(
    sample_path: Path,
    labels: Path | None,
    hand: dict[str, Path],
    groups: dict[str, Sequence[str]],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """값 확인 V를 층마다(표본에 처음 나온 순서) Claude와 견주고, 이어서 이름 붙인 묶음마다 견준다.
    묶음의 열쇠는 지정한 층 순서를 따른다(부트스트랩이 그 순서로 뽑는다)."""
    sample = list(read_jsonl(sample_path))
    claude = _load_labels(sample, labels, hand)
    value_ok = _value_ok(sample)
    v = Rater("V", {k: hybrid_verdicts((), ok) for k, ok in value_ok.items()})
    keys_of: dict[str, list[str]] = {}
    for r in sample:
        keys_of.setdefault(r["stratum"], []).append(r["key"])
    for name, strata in groups.items():
        unknown = [s for s in strata if s not in keys_of]
        if unknown:
            raise ValueError(f"group {name}: no stratum {unknown}")
        keys_of[name] = [k for s in strata for k in keys_of[s]]
    return {
        g: {"kappa_table": kappa_table(claude, v, keys, n_boot, seed), **value_cells(claude, value_ok, keys)}
        for g, keys in keys_of.items()
    }


# 소수 셋째 자리 문자열. NaN이면 "-".
def _f(x: float) -> str:
    return "-" if x != x else f"{x:.3f}"


# report 결과를 그룹별 표와 후보별 한 줄(kappa, 기준 대비 차이와 구간, 값 확인을 붙인 기준 대비 차이, 실패,
# 컨텍스트, 탈락 이유), 마지막에 채택 결과로 찍는다.
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


# NAME=PATH 인자 목록을 dict로. 형식이 틀리면 멈춘다.
def _named_paths(values: Sequence[str]) -> dict[str, Path]:
    out = {}
    for v in values:
        name, sep, path = v.partition("=")
        if not sep:
            raise SystemExit(f"expected NAME=PATH, got {v!r}")
        out[name] = Path(path)
    return out


# 하위 명령 넷(sample, judge, report, cells). README "요약 판정 재보정"에 실제로 쓴 명령이 있다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="층을 뽑아 표본과 라벨 양식을 쓴다")
    s.add_argument("--stratum", type=_parse_stratum, action="append", required=True, help="NAME=RUN:N")
    s.add_argument(
        "--hand", type=_parse_hand, action="append", default=[], help="NAME=reports/<run>/<사람 라벨>.jsonl"
    )
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.add_argument("--template", required=True)
    j = sub.add_parser("judge", help="표본 전체에 판정 모델 하나를 돌린다")
    j.add_argument("--sample", required=True)
    j.add_argument("--model", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--num-gpu", type=int)
    j.add_argument("--think", choices=sorted(THINK), default="off")
    j.add_argument("--option", type=parse_option, action="append", default=[], metavar="KEY=VALUE")
    r = sub.add_parser("report", help="일치 표와 채택 판정")
    r.add_argument("--sample", required=True)
    r.add_argument("--labels", help="표본 열쇠 기준 Claude 라벨(채운 양식)")
    r.add_argument("--hand-labels", action="append", default=[], help="STRATUM=FILE, 건 id 기준 라벨")
    r.add_argument(
        "--judge",
        action="append",
        default=[],
        help="NAME=FILE(표본 열쇠 기준 판정 행). 그 층의 건 id 기준 행이면 NAME@STRATUM=FILE",
    )
    r.add_argument("--baseline", required=True, help="후보가 넘어야 할 기준 판정 이름")
    r.add_argument("--candidates", required=True, help="쉼표로 이은 판정 이름, 등록 순서대로")
    r.add_argument("--decide-on", required=True, help="채택 판정에 쓰는 층, 쉼표로 이음")
    r.add_argument("--n-boot", type=int, default=2000)
    r.add_argument("--json")
    c = sub.add_parser("cells", help="값 확인 혼자를 Claude와 셀별로 비교")
    c.add_argument("--sample", required=True)
    c.add_argument("--labels", help="표본 열쇠 기준 Claude 라벨(채운 양식)")
    c.add_argument("--hand-labels", action="append", default=[], help="STRATUM=FILE, 건 id 기준 라벨")
    c.add_argument(
        "--group", action="append", default=[], help="NAME=STRATUM,STRATUM...(열쇠는 이 층 순서대로)"
    )
    c.add_argument("--n-boot", type=int, default=2000)
    c.add_argument("--json")
    args = ap.parse_args(argv)
    # 표준 출력을 UTF-8로 고정해, 출력을 파일로 돌릴 때도 콘솔 기본 인코딩(Windows 한국어 환경에서는 cp949,
    # compare.py 참고)과 상관없이 같은 인코딩으로 쓴다.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    # sample: 표본 파일과 라벨 양식을 함께 쓰고 층별 건수를 찍는다. 규칙대로 라벨 파일을 커밋·push한 뒤에 판정
    # 모델을 돌렸다.
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
    elif args.cmd == "cells":
        groups = {}
        for g in args.group:
            name, sep, strata = g.partition("=")
            if not (sep and name and strata):
                raise SystemExit(f"expected NAME=STRATUM,STRATUM..., got {g!r}")
            groups[name] = strata.split(",")
        try:
            rep = cells_report(
                Path(args.sample),
                Path(args.labels) if args.labels else None,
                _named_paths(args.hand_labels),
                groups,
                args.n_boot,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        for g, d in rep.items():
            t = d["kappa_table"]
            lo, hi = t["kappa_ci"]
            print(
                f"{g}: kappa {_f(t['kappa'])} [{_f(lo)}, {_f(hi)}], error recall {_f(t['error_recall'])}, "
                f"false error {_f(t['false_error_rate'])}, cells {d['cells']}"
            )
            print(f"   V miss but Claude 포함: {d['v_miss_but_claude_included']}")
            print(f"   V found but Claude not: {d['v_found_but_claude_not']}")
        if args.json:
            Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
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
