"""dev에 남은 오류가 정답 잡음인지 가르는 감사 도구. 기록된 실행만 읽고(새 추론 없음) dev 정답은 바꾸지
않는다.

python -m call_summary.label_audit build --group NAME=RUN[,RUN...] ... --errors-of RUN \\
        [--min-groups 2] --out AUDIT.json --template REVIEW.jsonl
    검토 대상 = 서빙 모델 실행(--errors-of)이 틀린 건 ∪ 서로 다른 학습 묶음 둘 이상이 정답과 다른 같은 구조
    기록을 낸 건. 대상 목록(모델 답 포함)과, 모델 답을 뺀 검토 양식을 쓴다
python -m call_summary.label_audit tally --audit AUDIT.json --review REVIEW.jsonl
    채운 검토 양식을 세어 결론 문장을 찍는다
"""

# [파일 역할] 기록된 평가 실행(reports/<run_id>/items.jsonl)에서 "정답 잡음일 수 있는 dev 건"을 고르고, 사람이
# 대화와 명세를 읽어 나눌 검토 양식을 만들며, 채운 양식을 미리 정한 규칙대로 센다. 모델·GPU를 쓰지 않는다.
# [왜 필요한가] 1단계 사람 점검(docs/experiments.md "사람 점검")에서 train 30건 중 3건(약 10%)의 정답 라벨이
# 대화와 조금 어긋났다. 서빙 모델 dev exact 94.6%에 남은 오류가 이런 정답 잡음 때문인지는 세지 않았다. 여러
# 모델이 정답과 다른 같은 답을 내는 건은 정답 쪽이 틀렸을 가능성이 높아(모델 합의로 라벨 잡음 후보를 찾는
# 방법), 서빙 모델의 오류와 함께 사람이 읽을 대상으로 삼는다. 규칙은 docs/experiments.md "dev 남은 오류의 정답
# 잡음 감사"에 검토 전에 적었다.
# [설계 결정] (1) 합의는 서로 다른 학습 묶음 사이에서만 센다: 같은 설정으로 다시 학습한 학생이나 실행기만 바꾼
# 실행(transformers와 GGUF)은 같은 버릇을 공유해, 그 일치를 합의로 세면 후보가 부푼다. (2) 같은 기록인지는
# 채점 (scoring.score_item)과 같은 기준으로 가린다: 요약은 보지 않고, 핵심 값은 (유형, 정규화 값) 다중 집합,
# 처리 목록과 후속 조치 코드는 순서·중복을 보지 않는 집합이다. 그래서 "정답과 다르다"가 곧 exact가 틀렸다는
# 뜻이 되고, 기록된 exact와 어긋나면 멈춘다. (3) 검토 양식에는 모델 답을 넣지 않는다: 판정자가 대화와 명세를
# 먼저 읽고 정답이 대화로 뒷받침되는지 적은 뒤에 대상 목록에서 모델 답을 보게 한다. (4) 같은 데이터로 채점된
# 실행인지(data_sha)와 지금 데이터 파일이 그 지문과 같은지 확인한다. 데이터가 바뀌었으면 정답도 다를 수 있다.
# [흐름] structure(구조 기록 정규화) → field_diff(필드별 차이) → load_run·build_audit(대상 목록) →
# review_template(검토 양식) → tally·conclusion(채운 양식 세기) → main(명령줄).
# [연결] 데이터셋과 정답은 dataset.Item(정답은 명세에서 다시 만든다), 값 정규화는 values.normalize, 업종별
# 핵심 값 종류는 domains.DOMAINS를 쓴다. 실행 manifest의 데이터 경로는 dataset.manifest_data로 슬래시 꼴로
# 읽는다.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .dataset import Item, load_items, manifest_data, read_jsonl, write_jsonl
from .domains import DOMAINS
from .values import normalize

# 구조 필드(요약을 뺀 정답 기록의 필드). 차이 필드와 검토 칸은 늘 이 순서로 적는다.
FIELDS = ("category", "resolution", "entities", "actions_taken", "follow_up")
# 검토 1단계(모델 답을 보기 전): 대화가 정답 기록의 그 필드를 뒷받침하는가.
READINGS = ("뒷받침", "어긋남", "애매")
# 검토 2단계(모델 답을 본 뒤): 필드 분류. 정답 잡음 = 대화만 읽으면 정답의 그 필드가 틀림, 모델 오류 = 대화가
# 정답을 뒷받침, 애매 = 둘 다로 읽힘(docs/experiments.md의 분류 기준).
CLASSES = ("정답 잡음", "모델 오류", "애매")
# 1단계 판단에서 바로 나오는 분류. 2단계에서 이와 다르게 적은 필드는 "바꾼 필드"로 따로 센다(모델 답을 본
# 영향을 드러내려는 것).
DEFAULT_CLASS = {"뒷받침": "모델 오류", "어긋남": "정답 잡음", "애매": "애매"}
# 답 하나의 분류는 그 답이 정답과 다른 필드 가운데 가장 나쁜 것이다: 모델 오류 > 애매 > 정답 잡음. 모든 차이
# 필드가 정답 잡음일 때만 그 오류를 정답 잡음으로 친다(정답을 고쳐도 다른 필드가 틀리면 오류가 남는다).
_SEVERITY = {"정답 잡음": 0, "애매": 1, "모델 오류": 2}
# 검토 대상이 된 이유: 서빙 모델 실행이 틀린 건, 교차 묶음 합의 후보.
SERVING, CROSS = "serving_error", "cross_group"


# 정렬·비교·출력에 함께 쓰는 고정된 JSON 글(키 정렬, 한글 그대로). 같은 구조면 같은 글이 나온다.
def _dump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


# 기록 dict(정답이나 모델 답의 pred)를 채점과 같은 같음 기준의 구조로 바꾼다. 입력: 업종 키, AfterCallRecord
# 꼴의 dict(summary는 있어도 무시). 출력: 유형·처리 결과(글자 그대로), 핵심 값([유형, 정규화 값] 목록을 정렬한
# 다중 집합. 유형을 모르거나 값을 그 종류로 읽을 수 없으면 [유형, None, 원래 값]으로 남긴다. 채점은 이런 값을
# 오답으로 치므로 정답과 같아질 수 없다), 처리 목록과 후속 조치 코드(중복을 뺀 정렬 목록), 후속 조치 필요
# 여부. 주의: 정답 쪽 값은 명세에서 나와 늘 읽을 수 있다(scoring._entity_keys의 가정과 같다).
def structure(domain: str, record: Mapping) -> dict:
    """요약을 뺀 구조 기록을 채점(scoring.score_item)과 같은 같음 기준으로 정규화한다."""
    kinds = {et.label: et.kind for et in DOMAINS[domain].entity_types}
    entities = []
    for e in record["entities"]:
        kind = kinds.get(e["type"])
        norm = normalize(kind, e["value"]) if kind else None
        entities.append([e["type"], norm] if norm is not None else [e["type"], None, e["value"]])
    follow = record["follow_up"]
    return {
        "category": record["category"],
        "resolution": record["resolution"],
        "entities": sorted(entities, key=_dump),
        "actions_taken": sorted(set(record["actions_taken"])),
        "follow_up": {"required": bool(follow["required"]), "codes": sorted(set(follow["codes"]))},
    }


# 두 구조(structure의 결과)의 필드별 차이. 입력: 정답 구조, 답 구조. 출력: 다른 필드만 FIELDS 순서로 담은
# dict. 핵심 값과 처리는 {"missing": 정답에만 있는 것, "extra": 답에만 있는 것}, 나머지 필드는 {"gold",
# "pred"}.
def field_diff(gold: Mapping, pred: Mapping) -> dict:
    out: dict = {}
    for f in FIELDS:
        if gold[f] == pred[f]:
            continue
        if f in ("entities", "actions_taken"):
            g = Counter(_dump(x) for x in gold[f])
            p = Counter(_dump(x) for x in pred[f])
            out[f] = {
                "missing": [json.loads(k) for k in sorted((g - p).elements())],
                "extra": [json.loads(k) for k in sorted((p - g).elements())],
            }
        else:
            out[f] = {"gold": gold[f], "pred": pred[f]}
    return out


# 사람이 읽을 기록(요약만 뺀 원래 말투 그대로). 대상 목록과 검토 양식의 "gold"·"record" 칸에 쓴다.
def _record(record: Mapping) -> dict:
    return {f: record[f] for f in FIELDS}


# 기록된 평가 실행 하나. name은 실행 폴더 이름(run_id)이고 대상 목록에는 이 이름만 남긴다(로컬 절대 경로를
# 남기지 않으려는 것). data는 manifest의 데이터셋 경로(슬래시 꼴), rows는 item_id → items.jsonl 한 줄.
@dataclass(frozen=True)
class Run:
    name: str
    data: str
    data_sha: str | None
    rows: dict[str, dict]


# 실행 폴더(reports/<run_id> 또는 outputs/runs/<run_id>)의 manifest.json과 items.jsonl을 읽는다. 같은 건이 두
# 번 나오면 어느 답을 셀지 모호하므로 멈춘다.
def load_run(path: str | Path) -> Run:
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    rows: dict[str, dict] = {}
    for row in read_jsonl(path / "items.jsonl"):
        if row["item_id"] in rows:
            raise ValueError(f"{path.name}: {row['item_id']} appears twice")
        rows[row["item_id"]] = row
    return Run(path.name, manifest_data(manifest), manifest.get("data_sha"), rows)


# 데이터 파일의 지문: sha256 앞 12자리. evaluate가 manifest의 data_sha로 남기는 값과 같은 계산이다.
def data_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# 실행들이 같은 데이터로 채점됐는지 확인하고 그 데이터를 읽는다. 입력: 실행 목록, 데이터 경로(None이면 첫
# 실행의 manifest 경로). 출력: (경로, 지문, 건 목록). 실행마다 data_sha가 다르거나 없으면, 지금 파일의 지문이
# 기록과 다르면(정답이 달라졌을 수 있다), 실행에 데이터셋에 없는 건이 있으면 ValueError로 멈춘다.
def _load_data(runs: Sequence[Run], data: str | Path | None) -> tuple[Path, str, list[Item]]:
    shas = sorted({r.data_sha or "" for r in runs})
    if len(shas) != 1 or not shas[0]:
        raise ValueError(f"runs must share one recorded data_sha, got {shas}")
    path = Path(data) if data is not None else Path(runs[0].data)
    actual = data_sha(path)
    if actual != shas[0]:
        raise ValueError(f"{path.as_posix()}: sha256 {actual} is not the runs' data_sha {shas[0]}")
    items = load_items(path)
    known = {it.item_id for it in items}
    for r in runs:
        unknown = sorted(set(r.rows) - known)
        if unknown:
            raise ValueError(f"{r.name}: {unknown[0]} is not in {path.as_posix()}")
    return path, shas[0], items


# 실행 하나가 건 하나에 낸 답이 정답과 다른 구조면 그 구조를, 맞았거나 형식 실패(pred 없음)거나 그 건이 없으면
# None을 돌려준다. 이 도구의 같음 기준과 기록된 score.exact가 어긋나면(정규화 규칙이 채점 뒤에 바뀌었거나
# 기록이 손상됨) 멈춘다. 이 확인 덕분에 "정답과 다른 같은 기록"이 채점의 오답과 늘 같은 뜻이다.
def _wrong_structure(run: Run, item: Item, gold_key: str) -> dict | None:
    row = run.rows.get(item.item_id)
    if row is None:
        return None
    exact = bool(row["score"]["exact"])
    if row["pred"] is None:
        if exact:
            raise ValueError(f"{run.name}: {item.item_id} has no parsed record but exact=True")
        return None
    s = structure(item.domain, row["pred"])
    wrong = _dump(s) != gold_key
    if wrong == exact:
        side = "differs from" if wrong else "equals"
        raise ValueError(
            f"{run.name}: {item.item_id} recorded exact={exact} but the record {side} the gold record"
        )
    return s if wrong else None


# 검토 대상 목록을 만든다. 입력: 묶음 이름 → 실행 폴더 목록(서로 다른 학습 묶음. 같은 설정의 재학습·실행기만
# 바꾼 실행은 한 묶음에 넣는다), 서빙 모델 실행(errors_of, 묶음에 들어 있어도 된다), 합의로 칠 최소 묶음 수,
# 데이터 경로(None이면 manifest의 경로). 묶음이 그 기록을 냈다 = 그 묶음의 실행 하나 이상이 냈다. 출력 dict:
# 데이터·지문·건수, 묶음별 실행 이름, 서빙 실행 이름과 exact [맞은 수, 건수], 수(서빙 오류, 교차 묶음 후보,
# 모든 묶음이 같은 후보, 둘 다인 건, 대상), 대상(데이터 순서): item_id, 업종, 이유, 차이 필드(서빙 모델의 틀린
# 답과 합의된 답이 정답과 다른 필드의 합집합), 정답 기록, 서빙 답(맞았으면 None), 묶음 실행들이 낸 틀린 답
# 전부(묶음 수가 많은 순, 같으면 실행 수가 많은 순. 합의 여부 agreed와 필드별 차이 diff를 함께).
def build_audit(
    groups: Mapping[str, Sequence[str | Path]],
    errors_of: str | Path,
    min_groups: int = 2,
    data: str | Path | None = None,
) -> dict:
    if len(groups) < 2:
        raise ValueError("need at least two groups: agreement inside one group does not count")
    if not 2 <= min_groups <= len(groups):
        raise ValueError(f"--min-groups must be between 2 and the number of groups ({len(groups)})")
    loaded: dict[str, list[Run]] = {}
    owner: dict[str, str] = {}
    for name, paths in groups.items():
        if not paths:
            raise ValueError(f"group {name} has no runs")
        loaded[name] = []
        for p in paths:
            run = load_run(p)
            if run.name in owner:
                raise ValueError(f"{run.name} is in group {owner[run.name]} and group {name}")
            owner[run.name] = name
            loaded[name].append(run)
    serving = load_run(errors_of)
    path, sha, items = _load_data([*(r for runs in loaded.values() for r in runs), serving], data)

    counts = dict.fromkeys(("serving_errors", "cross_group", "all_groups", "both"), 0)
    n_exact = 0
    targets = []
    for item in items:
        gold_record = item.gold().model_dump()
        gold = structure(item.domain, gold_record)
        gold_key = _dump(gold)

        # 묶음 실행들의 틀린 답을 같은 구조끼리 모은다(첫 실행의 원래 말투를 사람이 읽을 기록으로 남긴다).
        outs: dict[str, dict] = {}
        for name, runs in loaded.items():
            for run in runs:
                s = _wrong_structure(run, item, gold_key)
                if s is None:
                    continue
                o = outs.setdefault(
                    _dump(s),
                    {
                        "groups": [],
                        "runs": [],
                        "structure": s,
                        "record": _record(run.rows[item.item_id]["pred"]),
                    },
                )
                if name not in o["groups"]:
                    o["groups"].append(name)
                o["runs"].append(run.name)
        answers = sorted(
            outs.values(), key=lambda o: (-len(o["groups"]), -len(o["runs"]), _dump(o["structure"]))
        )
        agreed = [o for o in answers if len(o["groups"]) >= min_groups]

        # 서빙 모델 답: 틀렸으면 그 기록과 차이, 형식 실패면 기록 없이 빈 차이, 맞았으면 None.
        serving_entry = None
        row = serving.rows.get(item.item_id)
        if row is not None:
            s = _wrong_structure(serving, item, gold_key)
            if s is not None:
                serving_entry = {"record": _record(row["pred"]), "diff": field_diff(gold, s)}
            elif row["pred"] is None:
                serving_entry = {"record": None, "diff": {}}
            else:
                n_exact += 1

        reasons = ([SERVING] if serving_entry is not None else []) + ([CROSS] if agreed else [])
        if not reasons:
            continue
        counts["serving_errors"] += serving_entry is not None
        counts["cross_group"] += bool(agreed)
        counts["all_groups"] += any(len(o["groups"]) == len(groups) for o in agreed)
        counts["both"] += len(reasons) == 2
        fields = set(serving_entry["diff"]) if serving_entry is not None else set()
        for o in agreed:
            fields |= set(field_diff(gold, o["structure"]))
        targets.append(
            {
                "item_id": item.item_id,
                "domain": item.domain,
                "reasons": reasons,
                "fields": [f for f in FIELDS if f in fields],
                "gold": _record(gold_record),
                "serving": serving_entry,
                "answers": [
                    {
                        "groups": o["groups"],
                        "runs": o["runs"],
                        "agreed": len(o["groups"]) >= min_groups,
                        "record": o["record"],
                        "diff": field_diff(gold, o["structure"]),
                    }
                    for o in answers
                ],
            }
        )
    return {
        "data": path.as_posix(),
        "data_sha": sha,
        "n_items": len(items),
        "groups": {name: [r.name for r in runs] for name, runs in loaded.items()},
        "errors_of": serving.name,
        "errors_of_exact": [n_exact, len(serving.rows)],
        "min_groups": min_groups,
        "counts": {**counts, "targets": len(targets)},
        "targets": targets,
    }


# 검토 양식(건마다 한 줄). 입력: build_audit의 결과, 데이터 경로(None이면 대상 목록에 적힌 경로. 지문이 다르면
# 멈춘다). 출력 dict 목록(대상 순서): item_id, 업종, 이유, 차이 필드 이름, 명세 전체, 요약을 뺀 정답 기록,
# 번호 붙은 발화("번호 화자: 내용"), 관련 발화(정답 핵심 값과 말실수 값이 글자 그대로 나온 발화 번호), 빈 판정
# 칸 (필드마다 reading·class·turns·note). 모델 답은 넣지 않는다: 판정자가 대화와 명세로 정답을 먼저 판단하게
# 한다.
def review_template(audit: Mapping, data: str | Path | None = None) -> list[dict]:
    path = Path(data) if data is not None else Path(audit["data"])
    if data_sha(path) != audit["data_sha"]:
        raise ValueError(f"{path.as_posix()}: sha256 {data_sha(path)} is not the audit's data_sha")
    items = {it.item_id: it for it in load_items(path)}
    rows = []
    for t in audit["targets"]:
        item = items[t["item_id"]]
        spec = item.spec
        values = [(f"{sv.type} {sv.value}", sv.value) for sv in spec.slots]
        if spec.distractor is not None:
            d = spec.distractor
            values.append((f"말실수 값 {d.type} {d.value}", d.value))
        rows.append(
            {
                "item_id": item.item_id,
                "domain": item.domain,
                "reasons": t["reasons"],
                "fields": t["fields"],
                "spec": json.loads(json.dumps(spec.to_dict(), ensure_ascii=False)),
                "gold": t["gold"],
                "turns": [f"{i} {turn['speaker']}: {turn['text']}" for i, turn in enumerate(item.turns, 1)],
                "value_turns": {
                    label: [i for i, turn in enumerate(item.turns, 1) if value in turn["text"]]
                    for label, value in values
                },
                "review": {f: {"reading": None, "class": None, "turns": [], "note": ""} for f in t["fields"]},
            }
        )
    return rows


# 필드 분류 목록 가운데 가장 나쁜 것(모델 오류 > 애매 > 정답 잡음).
def _worst(classes: Sequence[str]) -> str:
    return max(classes, key=_SEVERITY.__getitem__)


# 채운 검토 양식을 센다. 입력: 대상 목록, 채운 양식의 줄들. 대상과 양식의 건이 맞지 않거나, 차이 필드 가운데
# 판정이 비었거나 정한 말이 아니면 ValueError로 멈춘다(빠진 판정을 건너뛰면 비율이 틀린다). 셈: 서빙 모델 답의
# 분류 = 그 답이 다른 필드 가운데 가장 나쁜 것(형식 실패는 모델 오류), 합의된 답의 분류 = 합의된 답들이 다른
# 필드 가운데 가장 나쁜 것, 검토 대상 한 건의 분류 = 서빙 모델이 틀린 건이면 서빙 답의 분류, 아니면 합의된
# 답의 분류. 가설 지표 = 후보 가운데 합의된 답이 정답 잡음인 비율(점 추정). 천장(참고) = (서빙 모델이 맞힌 수
# + 정답 잡음인 서빙 오류 수) / 서빙 실행 건수. 정답을 고쳤을 때 모델 답이 맞을지는 보지 않으므로 위로 잡은
# 값이다.
def tally(audit: Mapping, review: Sequence[Mapping]) -> dict:
    by_id: dict[str, Mapping] = {}
    for row in review:
        if row["item_id"] in by_id:
            raise ValueError(f"{row['item_id']} appears twice in the review")
        by_id[row["item_id"]] = row
    ids = [t["item_id"] for t in audit["targets"]]
    missing = [i for i in ids if i not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError(f"review does not match the audit targets: missing {missing}, extra {extra}")
    unfinished = []
    for t in audit["targets"]:
        cells = by_id[t["item_id"]].get("review") or {}
        for f in t["fields"]:
            cell = cells.get(f) or {}
            if cell.get("reading") not in READINGS or cell.get("class") not in CLASSES:
                unfinished.append(f"{t['item_id']} {f}")
    if unfinished:
        raise ValueError("unfinished or unknown labels: " + ", ".join(unfinished))

    labels = dict.fromkeys(CLASSES, 0)
    serving_labels = dict.fromkeys(CLASSES, 0)
    candidate_labels = dict.fromkeys(CLASSES, 0)
    n_fields = n_changed = 0
    per_item = []
    for t in audit["targets"]:
        cells = by_id[t["item_id"]]["review"]
        cls = {f: cells[f]["class"] for f in t["fields"]}
        n_fields += len(t["fields"])
        n_changed += sum(cells[f]["class"] != DEFAULT_CLASS[cells[f]["reading"]] for f in t["fields"])
        serving_label = agreed_label = None
        if t["serving"] is not None:
            diff = t["serving"]["diff"]
            serving_label = _worst([cls[f] for f in diff]) if diff else "모델 오류"
            serving_labels[serving_label] += 1
        agreed_fields = {f for a in t["answers"] if a["agreed"] for f in a["diff"]}
        if agreed_fields:
            agreed_label = _worst([cls[f] for f in agreed_fields])
            candidate_labels[agreed_label] += 1
        label = serving_label if serving_label is not None else agreed_label
        labels[label] += 1
        per_item.append(
            {"item_id": t["item_id"], "label": label, "serving": serving_label, "agreed": agreed_label}
        )
    n_candidates = sum(candidate_labels.values())
    share = candidate_labels["정답 잡음"] / n_candidates if n_candidates else None
    ok, n_run = audit["errors_of_exact"]
    return {
        "targets": len(audit["targets"]),
        "labels": labels,
        "serving_errors": sum(serving_labels.values()),
        "serving_labels": serving_labels,
        "candidates": n_candidates,
        "candidate_labels": candidate_labels,
        "candidate_noise_share": share,
        "hypothesis_met": None if share is None else share >= 0.5,
        "fields": n_fields,
        "changed": n_changed,
        "exact": [ok, n_run],
        "ceiling": (ok + serving_labels["정답 잡음"]) / n_run if n_run else None,
        "items": per_item,
    }


# tally 결과를 docs/experiments.md에 검토 전에 정한 결론 문장 틀로 쓴다. 첫 줄이 결론 문장이고, 이어서 가설
# 지표, 천장(참고), 모델 답을 본 뒤 바꾼 필드 수를 적는다.
def conclusion(t: Mapping) -> str:
    lab, srv, cand = t["labels"], t["serving_labels"], t["candidate_labels"]
    lines = [
        f"검토 대상 {t['targets']}건 중 정답 잡음 {lab['정답 잡음']}, 모델 오류 {lab['모델 오류']}, "
        f"애매 {lab['애매']}. 서빙 모델 dev 오류 {t['serving_errors']}건 중 "
        f"{srv['정답 잡음']}건이 정답 잡음이다."
    ]
    if t["candidates"]:
        verdict = "맞음" if t["hypothesis_met"] else "맞지 않음"
        share = 100 * t["candidate_noise_share"]
        lines.append(
            f"교차 묶음 후보 {t['candidates']}건 중 정답 잡음 {cand['정답 잡음']}, "
            f"모델 오류 {cand['모델 오류']}, 애매 {cand['애매']} "
            f"(정답 잡음 비율 {share:.1f}%, 가설(절반 이상) {verdict})."
        )
    ok, n = t["exact"]
    if n:
        lines.append(
            f"참고: 정답 잡음을 맞은 것으로 치면 서빙 모델 exact 점 추정 {100 * t['ceiling']:.1f}% "
            f"(기록 {100 * ok / n:.1f}%, {ok}/{n})."
        )
    lines.append(f"모델 답을 본 뒤 1단계 판단과 다르게 분류한 필드 {t['changed']}/{t['fields']}.")
    return "\n".join(lines)


# --group 인수 "NAME=RUN[,RUN...]"을 (이름, 실행 경로 목록)으로 읽는다. 꼴이 틀리면 argparse 사용법 오류가
# 된다.
def _group(text: str) -> tuple[str, list[str]]:
    name, sep, runs = text.partition("=")
    paths = [r for r in runs.split(",") if r]
    if not sep or not name or not paths:
        raise argparse.ArgumentTypeError(f"expected NAME=RUN[,RUN...], got {text!r}")
    return name, paths


# 명령줄 입구. build는 대상 목록 JSON과 검토 양식 JSONL을 쓰고 수와 대상 id·차이 필드 이름만 찍는다(모델 답은
# 찍지 않는다. 판정자가 검토 전에 답을 보지 않게). tally는 채운 양식을 세어 결론 문장과 건별 분류를 찍는다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="검토 대상 목록과 빈 검토 양식을 만든다")
    b.add_argument(
        "--group",
        action="append",
        type=_group,
        required=True,
        metavar="NAME=RUN[,RUN...]",
        help="학습 묶음 하나와 그 실행 폴더들(쉼표로 구분). 같은 설정으로 다시 학습했거나 실행기만 바꾼 "
        "실행은 한 묶음에 넣는다. 묶음 안의 일치는 합의로 세지 않는다. 두 번 이상 준다",
    )
    b.add_argument(
        "--errors-of",
        required=True,
        metavar="RUN",
        help="exact가 틀린 건을 모두 검토 대상에 넣을 실행(서빙 모델). 묶음에 들어 있어도 된다",
    )
    b.add_argument(
        "--min-groups",
        type=int,
        default=2,
        help="정답과 다른 같은 기록을 낸 묶음이 이 수 이상이면 후보 (기본 2)",
    )
    b.add_argument("--data", help="데이터셋 파일 (기본: 실행 manifest의 경로. 지문이 data_sha와 같아야 한다)")
    b.add_argument("--out", required=True, help="대상 목록 JSON (모델 답 포함, 검토 1단계 뒤에 연다)")
    b.add_argument("--template", required=True, help="검토 양식 JSONL (모델 답 없음, 판정 칸을 채운다)")
    t = sub.add_parser("tally", help="채운 검토 양식을 세어 결론 문장을 찍는다")
    t.add_argument("--audit", required=True, help="build가 쓴 대상 목록 JSON")
    t.add_argument("--review", required=True, help="판정 칸을 채운 검토 양식 JSONL")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        groups = dict(args.group)
        if len(groups) != len(args.group):
            ap.error("group names must be unique")
        audit = build_audit(groups, args.errors_of, args.min_groups, args.data)
        rows = review_template(audit, args.data)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(audit, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        n_rows = write_jsonl(args.template, rows)
        c = audit["counts"]
        ok, n = audit["errors_of_exact"]
        print(f"데이터 {audit['data']} ({audit['data_sha']}) {audit['n_items']}건, 묶음 {len(groups)}개")
        print(
            f"서빙 모델 오류 {c['serving_errors']} (exact {ok}/{n}), 교차 묶음 후보 {c['cross_group']} "
            f"(모든 묶음 {c['all_groups']}), 겹침 {c['both']}, 검토 대상 {c['targets']}"
        )
        for target in audit["targets"]:
            print(f"  {target['item_id']}  {'+'.join(target['reasons'])}  {','.join(target['fields'])}")
        print(f"대상 목록 {out.as_posix()}, 검토 양식 {Path(args.template).as_posix()} ({n_rows}건)")
        return 0

    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    result = tally(audit, list(read_jsonl(args.review)))
    print(conclusion(result))
    for it in result["items"]:
        serving_label, agreed_label = it["serving"] or "-", it["agreed"] or "-"
        print(f"  {it['item_id']}  {it['label']}  (서빙 답 {serving_label}, 합의된 답 {agreed_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
