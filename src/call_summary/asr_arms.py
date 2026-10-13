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

# [파일 역할] 음성 조건 v3(인식기 바꾸기)의 CPU 쪽 도구다. GPU로 소리를 다시 합성·인식한 결과(1패스
# index.jsonl, 2패스 팔별 hyp 파일)를 읽어, 업종 맥락 글 만들기(context), 다시 합성한 소리가 예전 전사와
# 같은지 확인(identity), 번호 발화만으로 하는 싼 관문(gate), 팔마다 학생이 받을 데이터셋 만들기(build), 팔끼리
# 값 보존율 짝지은 비교(compare)를 한다.
# [왜 필요한가] 서빙 학생은 음성 조건에서 인식기의 천장 가까이 있어(test-d-v2 exact 79.3, 정답 값이 모두 남은
# 건 81.7%), 학생을 바꾸지 않고 수치를 움직일 수 있는 것은 입력, 곧 인식기다(docs/experiments.md "음성 조건
# v3" 왜). 같은 소리를 여러 인식기(팔)로 받아 적고, 학생을 돌리기 전에 "정답 값이 전사에 남는가"로 먼저
# 거른다. GPU 시간을 아끼려고 싼 단계부터 차례로 걸렀다(관문 → 단계 A → 단계 B).
# [설계 이유] 팔 이름이 "+I"로 끝나면 식별자 규칙(id_itn)을 씌운다. 판정은 학생이 실제로 받는 글(규칙 뒤)로
# 한다. 값 찾기는 채점과 같은 spoken 규칙(values.occurs_in_transcript)을 쓴다. 결과(2026-10-05): Th+I·L+I가
# 단계 A를 통과했지만 단계 B에서 학생 exact 차이의 구간이 0을 포함해 인식기를 바꾸지 않았다(README).

from __future__ import annotations

# unicodedata: CER 계산 전에 NFKC로 정규화한다. dataclass: 짝지은 부트스트랩에 넘길 건별 값(_Flag).
import argparse
import json
import math
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# catalog: 업종의 닫힌 카탈로그 값(상품명, 요금제, 기기명 등). id_itn: 번호를 숫자로 되돌리는 식별자 규칙.
# paired_bootstrap_diff·value_survival: 채점기와 같은 짝지은 부트스트랩과 값 보존 계산.
# digit_runs·values_in_transcript: 숫자 덩어리와 금액 읽기(2차 지표용).
from .dataset import Item, load_items, read_jsonl, write_jsonl
from .domains import DOMAINS, Domain, catalog
from .id_itn import ITN_VERSION, id_itn
from .prompts import format_transcript
from .scoring import paired_bootstrap_diff, value_survival
from .values import digit_runs, occurs_in_transcript, values_in_transcript

# DEFAULT_AUDIO: 1패스가 소리(WAV)와 기준 인식기 T의 글을 남기는 폴더(약 1.8 GB, 커밋하지 않음).
# CONTEXT_VERSION: 업종 맥락 글의 판(ctx-1). N_BOOT, SEED: 짝지은 부트스트랩 2,000번, seed 0(규칙). HeardOf:
# index 행 하나 → 그 팔이 들은 글을 돌려주는 함수 타입.
DEFAULT_AUDIO = Path("outputs/asr3/audio")
CONTEXT_VERSION = "ctx-1"
N_BOOT, SEED = 2000, 0
HeardOf = Callable[[dict], str]


# --- context ----------------------------------------------------------------------------------------------


# 업종 맥락 글: 회사 이름과 카탈로그 값 전부를 쉼표로 잇는다. Th 팔에서는 faster-whisper hotwords로, Qc
# 팔에서는 Qwen3-ASR의 시스템 프롬프트로 준다. 정답 이름이 같은 닫힌 카탈로그에서 나오므로 정답 후보를 모두
# 알려 주는 셈이라, 이 맥락을 쓰는 팔은 "카탈로그"로 따로 표시하고 비기면 쓰지 않는 팔을 고른다(규칙).
def context_prompt(domain: Domain) -> str:
    """The company name and every catalog value, comma-separated (hotwords for Th, the prompt for Qc)."""
    return ", ".join((domain.company, *catalog(domain)))


# 모든 업종의 맥락 글(context 하위 명령이 JSON으로 저장해 2패스 스크립트가 읽는다).
def contexts() -> dict[str, str]:
    return {key: context_prompt(d) for key, d in DOMAINS.items()}


# --- sources ----------------------------------------------------------------------------------------------


# 팔 이름이 "+I"로 끝나면 식별자 규칙을 쓴다(예: Th+I, L+I). 이름 규칙으로 정해 두어 팔마다 설정을 따로 들고
# 다니지 않는다.
def uses_itn(arm: str) -> bool:
    return arm.endswith("+I")


# 1패스 index.jsonl(발화마다 키, 소리 길이, WAV sha256, T의 글, 같은 키의 v2 캐시 글)을 읽고 분할로 거른다.
def read_index(audio: str | Path = DEFAULT_AUDIO, split: str | None = None) -> list[dict]:
    rows = list(read_jsonl(Path(audio) / "index.jsonl"))
    return [r for r in rows if split is None or r["split"] == split]


# index 행을 (건 id, 발화 번호)로 찾을 수 있게 dict로 바꾼다.
def by_turn(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(r["item_id"], r["turn"]): r for r in rows}


# index 원천: 1패스에서 기준 인식기 T(large-v3-turbo)가 남긴 바이트를 듣고 적은 글.
def from_index(row: dict) -> str:
    """T's text, recognised in pass 1 from the bytes it kept."""
    return row["heard"]


# hyp 원천: 2패스 팔 하나의 결과 파일에서 같은 키의 글을 꺼내는 함수를 만든다. 모든 팔이 같은 키(같은 소리
# 바이트)를 들으므로 팔끼리 소리 차이가 없다.
def from_hyp(hyp: dict[str, dict]) -> HeardOf:
    def heard(row: dict) -> str:
        return hyp[row["key"]]["heard"]

    return heard


# hyp 파일을 키 → 행 dict로 읽는다.
def read_hyp(path: str | Path) -> dict[str, dict]:
    return {r["key"]: r for r in read_jsonl(path)}


# 원천 지정("index" 또는 hyp 파일 경로)을 (글을 꺼내는 함수, 출처 메타)로 바꾼다. hyp 옆의 .meta.json(모델
# 경로, 디코딩, 버전 등)을 데이터 meta에 옮기되, 길고 업종마다 다른 맥락 글(context)은 뺀다.
def source(spec: str) -> tuple[HeardOf, dict]:
    """`index` or a hyp file -> (text of an index row, what made it)."""
    if spec == "index":
        return from_index, {"source": "index", "model": "large-v3-turbo"}
    meta_path = Path(spec + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.pop("context", None)
    return from_hyp(read_hyp(spec)), {"source": Path(spec).as_posix(), **meta}


# 한 팔이 한 발화에서 들은 글. 소리가 없는 발화(wav가 None: 발음형에 한글이 없어 합성하지 않은 발화)는 빈
# 글이고, "+I" 팔이면 식별자 규칙을 씌운다.
def heard_text(arm: str, heard_of: HeardOf, row: dict) -> str:
    text = "" if row["wav"] is None else heard_of(row)
    return id_itn(text) if uses_itn(arm) else text


# --- build ------------------------------------------------------------------------------------------------


# 한 팔의 데이터셋 행을 만든다: turns에는 그 팔의 글(+I면 규칙 뒤), 분할 이름은 "<분할>-asr"(평가가 spoken
# 규칙을 쓰게 함), meta.asr에는 출처·verbalizer v2·식별자 규칙 판·쓴 글(written). 명세·참고 요약·건 순서는
# 원래 그대로라 v2와 짝지어 비교할 수 있다.
# 모든 발화가 index에 있어야 한다(없으면 KeyError로 멈춤). 결과는 단계 A의 data/asr3/<팔>/dev.jsonl과 단계 B의
# datasets/dev-asr-v3-*.jsonl이 됐다.
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


# 동일성 확인: 다시 합성한 소리에서 T가 적은 글이 기록된 v2 전사와 발화마다 같은지 센다. MeloTTS의 GPU 합성은
# 비트 단위로 같다는 보장이 없어서, 모두 같으면 기록된 v2 전사가 그대로 T이고, 하나라도 다르면 새 소리의 T
# 글을 기준으로 삼는다(규칙). 실제로 dev 3,157발화 중 24개가 달라 새 T 글이 기준이 됐다.
# cache_same은 같은 키의 v2 텍스트 캐시 글이 기록된 전사와 같은 수로, 키 계산이 맞는지 확인하는 값이다.
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


# 한 건의 정답 값 중 번호 종류(주문번호, 운송장번호 등)만 (유형, 값)으로.
def _id_values(it: Item) -> list[tuple[str, str]]:
    kinds = {et.label: et.kind for et in DOMAINS[it.domain].entity_types}
    return [(e.type, e.value) for e in it.gold().entities if kinds.get(e.type) == "id"]


# 싼 첫 관문: 번호가 든 발화만 먼저 합성·인식한(--only-ids) 결과로, 팔마다 정답 번호가 몇 개 찾아지는지 센다.
# dev 전체를 합성하기 전에 가망 없는 경우를 걸러 GPU 시간을 아낀다(dev 번호 발화 289개, 약 0.85시간 분량).
# index에 있는 발화만 이어 붙여 전사를 만들고, 채점의 spoken 규칙으로 번호를 찾는다.
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


# 관문 판정: 후보(기준 T와 참고 팔을 뺀 것) 중 하나라도 T보다 많이 찾으면 go, 아니면 stop. 참고 팔(T+I, 규칙
# 없는 Th·L 등)은 후보가 아니라 판정에서 뺀다(커밋 3062500). 실제 결과는 Th+I 110, L+I 113 > T 107로 go였다.
def gate_verdict(got: dict[str, dict], base: str = "T", reference: set[str] = frozenset()) -> str:
    """`stop` when no candidate (an arm that is neither the base nor a reference) finds more gold identifiers
    than the base arm, else `go`."""
    candidates = [g for a, g in got.items() if a != base and a not in reference]
    return "go" if any(g["found"] > got[base]["found"] for g in candidates) else "stop"


# --- stage A: compare built datasets ----------------------------------------------------------------------


# CER 비교용 정규화: NFKC, 숫자·영문·한글만 남기고(공백·문장부호 제거) 영문은 대문자로. 인식기마다 다른
# 띄어쓰기·문장부호 습관이 CER에 섞이지 않게 한다(규칙의 2차 지표 정의).
def _cer_norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", unicodedata.normalize("NFKC", text)).upper()


# 글자 편집 거리(레벤슈타인)와 기준 길이를 돌려준다. 행 하나씩만 들고 계산하는 동적 계획법이라 메모리가 짧은
# 쪽 길이에 비례한다. 발화마다 더해 팔 전체 CER(편집 합 ÷ 길이 합)을 낸다.
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


# 들린 글의 4자리 이상 숫자 덩어리 중 쓴 글에 숫자 덩어리나 금액 값으로 없는 것의 수와 전체 수. 인식기가
# 숫자를 잘못 듣거나 만들어 낸 정도를 보는 2차 지표다. 금액은 값으로 비교해 "36만 1천 원"과 "361,000원"을 같게
# 본다.
def wrong_digit_runs(written: str, heard: str) -> tuple[int, int]:
    """(heard digit runs of 4+ digits that are no digit run or amount of the written turn, heard runs of 4+
    digits). Amounts count by value, so "36만 1천 원" written and "361,000원" heard agree."""
    said = set(digit_runs(written)) | values_in_transcript("amount", written)
    runs = [r for r in digit_runs(heard) if len(r) >= 4]
    return sum(r not in said for r in runs), len(runs)


# 한글 숫자 이름이 네 자 이상 이어진 꼴("오팔구일"). 식별자 규칙 뒤에도 남은 한글 번호를 세는 2차 지표다.
_HANGUL_DIGIT_RUN = re.compile(r"[공영일이삼사오육칠팔구]{4,}")


def hangul_digit_runs(text: str) -> int:
    return len(_HANGUL_DIGIT_RUN.findall(text))


# 글에 한글 음절이 있는지. 쓴 글에 말이 있는데 들린 글이 빈 발화(인식기가 아무것도 적지 않음)를 세는 데 쓴다.
def has_hangul(text: str) -> bool:
    return any("가" <= ch <= "힣" for ch in text)


# 건 하나의 값(모든 정답 값이 남았으면 1.0, 아니면 0.0). paired_bootstrap_diff는 item_id로 짝을 맞추므로
# ItemScore 대신 이 작은 꼴을 넘긴다.
@dataclass(frozen=True)
class _Flag:
    item_id: str
    value: float


# _Flag 목록의 평균(= 모든 값이 남은 건의 비율). 빈 목록이면 NaN.
def _mean(xs) -> float:
    return sum(x.value for x in xs) / len(xs) if xs else math.nan


# 만들어진 팔 데이터셋 하나의 1차 지표(건마다 모든 정답 값이 남았는지)와 2차 지표(유형별 보존율, CER, 쓴 글에
# 없는 숫자열, 남은 한글 숫자 이름, 빈 발화)를 센다. 모두 spoken=True 채점 규칙과 meta.asr.written(쓴 글)을
# 쓴다.
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


# 단계 A: 기준(T) 대비 각 팔의 "모든 값이 남은 건" 비율 차이를 같은 240건 짝지은 부트스트랩으로 내고, 하한 >
# 0이면 통과로 표시한다. 등록 순서(order), 카탈로그 여부, 참고 여부를 함께 적어 ranking이 쓴다. 참고 팔은 같은
# 방식으로 보고만 하고 순위에는 넣지 않는다.
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


# 통과한 후보를 줄 세운다: 하한 높은 순 → 차이 큰 순 → 카탈로그를 안 쓰는 팔 먼저 → 등록 순서. 앞의 둘이 단계
# B로 간다(규칙). 실제로는 Th+I, L+I 순이었다.
def ranking(got: dict[str, dict]) -> list[str]:
    """Passing candidate arms, best first: higher lower bound, higher difference, non-catalog, registration
    order."""
    passed = [a for a, g in got.items() if g.get("passed") and not g.get("reference")]
    return sorted(passed, key=lambda a: (-got[a]["low"], -got[a]["diff"], got[a]["catalog"], got[a]["order"]))


# 비율을 백분율 소수 한 자리로. NaN(x != x)이면 "-".
def _pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.1f}"


# compare 결과를 docs/experiments.md "단계 A 결과" 표 꼴로 만든다: 팔, 모든 값이 남은 건, 팔 − T [95%], 관문,
# 유형별 보존율, CER, 틀린 숫자열, 한글 숫자 이름, 빈 발화.
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


# NAME=VALUE 인자 목록을 dict로. 형식이 틀리면 바로 멈춘다. 넘긴 순서가 dict 순서(= 등록 순서)로 남는다.
def _pairs(specs: list[str]) -> dict[str, str]:
    out = {}
    for spec in specs:
        name, _, value = spec.partition("=")
        if not value:
            raise SystemExit(f"expected NAME=VALUE, got {spec!r}")
        out[name] = value
    return out


# 하위 명령 다섯: context, identity, gate, build, compare. identity·gate·build는 --split과 --audio로 1패스
# index를 읽는다. 모델도 소리도 직접 다루지 않아 CPU만으로 돈다.
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
    g.add_argument("--reference", action="append", default=[], help="arms reported but not candidates")
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
        # --base는 NAME=<파일> 하나만 받는다(튜플 풀기로 하나가 아니면 오류).
        ((base_name, base),) = _pairs([args.base]).items()
        got = compare(base, _pairs(args.arm), set(args.catalog), base_name, set(args.reference))
        print(compare_table(got))
        print("ranking:", " ".join(ranking(got)) or "(none passed)")
    else:
        index = by_turn(read_index(args.audio, args.split))
        if args.cmd == "identity":
            got = identity(index, load_items(args.data))
            # 다른 발화 목록은 길어서 화면에는 개수만 찍고, 전체는 --json 파일에 남긴다.
            print(json.dumps({k: (len(v) if k == "different" else v) for k, v in got.items()}))
        elif args.cmd == "gate":
            arms = {name: source(spec)[0] for name, spec in _pairs(args.arm).items()}
            got = gate(load_items(args.source), index, arms)
            for arm, r in got.items():
                print(f"{arm}: {r['found']}/{r['ids']} {r['types']}")
            print("verdict:", gate_verdict(got, reference=set(args.reference)))
        else:
            heard_of, meta = source(args.src)
            # build: 출처 메타에 팔 이름을 더해 데이터 행마다 남기고, 파일만 쓰고 끝낸다(--json 없음).
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
