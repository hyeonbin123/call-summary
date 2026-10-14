"""누군가 처리하기 전에 확인이 필요한 번호.

기록 안의 번호(주문번호, 운송장번호, 접수번호, 카드 끝자리)는 정규화한 꼴(values.normalize)이 배포
형식(domains.EntityType.id_format)과 맞지 않거나, 전사에서 찾을 수 없으면(values.occurs_in_transcript,
spoken=True라 인식기가 쉼표나 공백으로 나눠 적은 숫자도 찾은 것으로 친다) 표시한다. 기록 자체는 절대
바꾸지 않는다.

표시는 정정도 아니고 정확도 이득도 아니다: 형식이 맞고 전사에 들어 있는 틀린 번호(인식기의 오류를 그대로
옮긴 것, 길이가 같은 숫자 혼동)는 두 검사를 모두 통과한다.

python -m call_summary.verify RUN...   기록된 평가 실행(reports/<run_id>)에 검사를 대 보고, 표시된 맞은
                                       번호와 틀린 번호를 센다(모델이 필요 없다)
"""

# [파일 역할] 기록의 번호(주문번호, 운송장번호, 접수번호, 카드 끝자리)가 배포 형식에 맞는지와 전사에 있는지를
# 검사해 "확인이 필요한 번호"를 고른다. 같은 검사를 기록된 평가 실행에 대 보는 명령(python -m
# call_summary.verify)도 둔다.
# [왜 필요한가] 음성 인식을 거친 전사에서 학습한 모델이 빠진 번호의 숫자를 짐작해 채우는 일이 있었다(README
# 4단계). 서비스는 스키마와 라벨 목록만 보면 틀린 번호도 그대로 내보내므로, 기록은 그대로 두고 다시 확인할
# 번호를 따로 알려 주는 장치를 만들었다(docs/experiments.md "확인이 필요한 번호 표시").
# [설계 결정] 형식은 값 그대로가 아니라 정규화한 값(공백·하이픈을 뺀 꼴)에서 본다. 값 그대로 보면 "5892 6890
# 2287"처럼 띄어 쓴 맞은 번호도 표시하는 오검출이 생긴다(같은 문서 "브리프와 다른 점"). 전사 포함은 4단계
# 채점과 같은 넓은 규칙(spoken=True)이다.
# [흐름] check_identifier(번호 하나) → verify_entities(기록 하나, service.py가 부름) → guard_report(기록된
# 실행 하나의 집계) → guard_table·guard_detail(문서 표) → main.

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal

# pydantic: IdentifierFlag를 서비스 응답(SummarizeResponse)의 필드로 그대로 쓰려고 BaseModel로 정의한다.
from pydantic import BaseModel

# dataset: 평가 실행이 가리키는 데이터 파일과 건별 기록(items.jsonl)을 읽는다. domains: 번호 유형과 배포
# 형식(id_format). values: 채점과 같은 정규화와 전사 찾기.
from .dataset import load_items, read_jsonl
from .domains import DOMAINS, EntityType
from .schema import AfterCallRecord
from .values import normalize, occurs_in_transcript

# 표시 사유: format(정규화한 값이 배포 형식과 맞지 않음), not_in_transcript(전사에서 찾을 수 없음). 둘 다일 수
# 있다.
Reason = Literal["format", "not_in_transcript"]


# 표시한 번호 하나. 응답의 needs_confirmation 목록 원소이며 {type, value, reasons} 꼴이다(README 서비스 문단).
class IdentifierFlag(BaseModel):
    type: str
    value: str
    reasons: list[Reason]


# 번호 하나를 두 검사에 대 보고 걸린 사유 목록을 돌려준다. 빈 목록이면 통과다.
# 형식: 정규화한 값 전체가 id_format 정규식과 맞아야 한다(fullmatch). 읽을 수 없는 값(정규화 결과가 비어
# None)도 형식 오류다. id_format이 없는 유형은 형식을 보지 않는다.
# 전사 포함: values.occurs_in_transcript(spoken=True). 서비스는 전사가 음성 인식 결과인지 알 수 없으므로 늘
# 넓은 규칙을 쓴다(인식기가 쉼표·공백으로 나눠 적은 숫자도 찾은 것으로 본다).
def check_identifier(et: EntityType, value: str, transcript: str) -> list[Reason]:
    """이 번호를 확인해야 하는 사유들(비어 있으면 두 검사를 모두 통과한 것)."""
    reasons: list[Reason] = []
    norm = normalize("id", value)
    if et.id_format is not None and (norm is None or re.fullmatch(et.id_format, norm) is None):
        reasons.append("format")
    if not occurs_in_transcript("id", value, transcript, spoken=True):
        reasons.append("not_in_transcript")
    return reasons


# 업종의 번호 종류(kind == "id") 유형만 이름 → 유형으로 모은다.
def _id_types(domain_key: str) -> dict[str, EntityType]:
    return {et.label: et for et in DOMAINS[domain_key].entity_types if et.kind == "id"}


# 기록의 핵심 값 중 번호 유형만 검사해 걸린 것을 기록 순서대로 돌려준다. 이름·금액·날짜 등 다른 유형과 목록에
# 없는 유형은 건너뛴다. 기록 자체는 바꾸지 않는다.
# 호출 위치: service.py /summarize(검증을 통과한 기록에만), tests/test_verify.py.
def verify_entities(domain_key: str, record: AfterCallRecord, transcript: str) -> list[IdentifierFlag]:
    """기록의 번호들에 대한 표시(기록 순서대로). 번호가 아닌 종류의 핵심 값은 검사하지 않는다."""
    types = _id_types(domain_key)
    flags = []
    for e in record.entities:
        et = types.get(e.type)
        if et is None:
            continue
        reasons = check_identifier(et, e.value, transcript)
        if reasons:
            flags.append(IdentifierFlag(type=e.type, value=e.value, reasons=reasons))
    return flags


# ---------------------------------------------------------------------------------------------------------
# 기록된 실행에 검사를 대 보기(docs/experiments.md "확인이 필요한 번호 표시").

# 아래 WRITTEN_FORMATS와 _written_form_ok는 보고용이다. 처음 제안된 "값 그대로" 운송장번호
# 형식(docs/experiments.md "브리프와 다른 점")이었다면 더 표시됐을 번호 수를 세어, 정규화한 값에서 형식을
# 보기로 한 결정이 수치에 영향을 줬는지 보인다(결과: 모든 실행에서 0, docs/experiments.md "확인이 필요한 번호
# 표시 결과").
# 운송장번호 형식은 브리프가 적은 그대로이고 값 그대로에 적용한다. 보고용일 뿐이며, 값 그대로에 형식 규칙을
# 걸었다면 어떤 맞은 번호(하이픈 없이 쓰거나 공백을 넣은 것)가 표시됐을지 보여 준다.
WRITTEN_FORMATS = {"운송장번호": r"\d{4}-\d{4}-\d{4}"}


# 값 그대로(앞뒤 공백만 뗀 꼴) 형식 검사. 운송장번호는 처음 제안된 하이픈 꼴, 나머지는 배포 형식을 쓴다.
def _written_form_ok(et: EntityType, value: str) -> bool:
    pattern = WRITTEN_FORMATS.get(et.label, et.id_format)
    return pattern is None or re.fullmatch(pattern, value.strip()) is not None


# 표시되지 않고 나가는 틀린 번호가 그 건의 같은 유형 정답 번호와 어떻게 다른지 분류한다: 그 유형 정답이 없음 /
# 읽을 수 없음 / 길이가 다름 / 길이가 같고 한 글자만 다름(어느 숫자가 어느 숫자로 바뀌었는지 "1>2") / 길이가
# 같고 여러 글자 다름.
# 정답이 여럿이면 길이가 같은 정답 중 다른 글자 수가 가장 적은 것과 비교한다. 결과 절의 "1 → 2가 8개" 같은
# 집계가 여기서 나온다.
def _shape(norm: str | None, golds: list[str]) -> tuple[str, str]:
    """틀린 번호가 그 건의 같은 유형 정답 값과 어떻게 다른지, 그리고 한 글자만 다르면 어느 글자가 어느
    글자로 바뀌었는지("1>2": 정답의 1이 2로 나왔다)."""
    if not golds:
        return "no_gold_of_type", ""
    if norm is None:
        return "unreadable", ""
    same_len = [g for g in golds if len(g) == len(norm)]
    if not same_len:
        return "other_length", ""
    best = min(same_len, key=lambda g: sum(a != b for a, b in zip(norm, g, strict=True)))
    diffs = [(g, p) for p, g in zip(norm, best, strict=True) if p != g]
    return ("one_char", f"{diffs[0][0]}>{diffs[0][1]}") if len(diffs) == 1 else ("same_length", "")


# 집계 칸: 예측한 번호, 맞음, 틀림, 표시된 맞은 번호(오검출), 표시된 틀린 번호.
def _empty_counts() -> dict:
    return {"predicted": 0, "correct": 0, "wrong": 0, "flagged_correct": 0, "flagged_wrong": 0}


# 기록된 평가 실행 하나(reports/<run_id>)에 검사를 대 보고 지표(docs/experiments.md "지표")에 필요한 수를
# 센다. 새 추론 없이 기록된 답과 전사만 쓰는 결정적 계산이다.
# 데이터 파일은 manifest의 data 경로를 쓰고(Windows 경로의 역슬래시를 바꿔 읽음), data를 넘기면 그것을
# 쓴다(테스트용).
# 돌려주는 것: 전체·유형별 수, 표시된 틀린 번호의 사유별 수, 조용히 나가는 틀린 번호의 모양·바뀐 숫자, 값
# 그대로 규칙이었다면 더 표시됐을 수, 답에 없는 정답 번호 수, 기록이 없는 행 수, 그리고 --show용 목록.
def guard_report(run: str | Path, data: str | Path | None = None) -> dict:
    """평가 실행 하나를 그 데이터셋에 대 본 표시 수(data를 주지 않으면 manifest의 `data`)."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(data or manifest["data"].replace("\\", "/"))}
    total = _empty_counts()
    by_type: dict[str, dict] = {}
    reasons_wrong: Counter = Counter()
    shapes_silent: Counter = Counter()
    changes_silent: Counter = Counter()
    written_rule_extra = {"correct": 0, "wrong": 0}
    missed = no_record = 0
    silent: list[list] = []
    false_flags: list[list] = []
    for row in read_jsonl(run / "items.jsonl"):
        # 스키마를 통과하지 못한 답은 서비스가 502로 거절해 번호가 나가지 않으므로 따로 센다("지표" 규칙).
        if not row.get("pred"):
            no_record += 1  # 서비스는 502로 답하므로 번호가 하나도 나가지 않는다
            continue
        it = items[row["item_id"]]
        types = _id_types(it.domain)
        pred = AfterCallRecord.model_validate(row["pred"])
        # 맞음·틀림은 채점과 같은 다중 집합 대응으로 정한다: 정규화한 (유형, 값)이 남은 정답과 같으면 맞음으로
        # 치고 그 정답을 하나 지운다. 같은 번호를 두 번 적으면 두 번째는 틀림이다.
        gold_ids = [(e.type, normalize("id", e.value)) for e in it.gold().entities if e.type in types]
        gold_left: Counter = Counter(gold_ids)
        transcript = it.transcript
        for e in pred.entities:
            et = types.get(e.type)
            if et is None:
                continue
            norm = normalize("id", e.value)
            ok = gold_left[(e.type, norm)] > 0
            if ok:
                gold_left[(e.type, norm)] -= 1
            # 서비스와 같은 검사(check_identifier)를 그대로 부른다.
            reasons = check_identifier(et, e.value, transcript)
            flagged = bool(reasons)
            cell = by_type.setdefault(e.type, _empty_counts())
            for c in (total, cell):
                c["predicted"] += 1
                c["correct" if ok else "wrong"] += 1
                if flagged:
                    c["flagged_correct" if ok else "flagged_wrong"] += 1
            if not ok and flagged:
                reasons_wrong["+".join(reasons)] += 1
            # 배포 형식(정규화한 값)은 통과했지만 값 그대로 형식에는 걸리는 번호: 값 그대로 규칙이었다면 더
            # 표시됐을 번호로 센다.
            if "format" not in reasons and not _written_form_ok(et, e.value):
                written_rule_extra["correct" if ok else "wrong"] += 1
            if ok and flagged:
                false_flags.append([it.item_id, e.type, e.value, reasons])
            # 조용히 나가는 틀린 번호: 형식도 맞고 전사에도 있어 표시되지 않은 틀린 번호. 이 검사가 잡지
            # 못하는 한계(대부분 한 글자 바뀜)를 보이려고 모양을 분류해 둔다.
            if not ok and not flagged:
                golds = sorted({g for t, g in gold_ids if t == e.type and g is not None})
                shape, change = _shape(norm, golds)
                shapes_silent[shape] += 1
                if change:
                    changes_silent[change] += 1
                silent.append([it.item_id, e.type, e.value, golds, shape, change])
        # 대응하지 못하고 남은 정답 번호 = 답에 없는 정답 번호. 이 표시가 다룰 수 없는 경우다(빠진 번호는
        # 표시할 값이 없다).
        missed += sum(gold_left.values())
    return {
        **total,
        "by_type": dict(sorted(by_type.items())),
        "flagged_wrong_reasons": dict(reasons_wrong.most_common()),
        "silent_wrong_shapes": dict(shapes_silent.most_common()),
        "silent_wrong_one_char_changes": dict(changes_silent.most_common()),
        "written_format_rule_would_also_flag": written_rule_extra,
        "gold_ids_not_predicted": missed,
        "rows_without_record": no_record,
        "silent_wrong": silent,
        "false_flags": false_flags,
    }


# 백분율 문자열(분모가 0이면 "-").
def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}" if b else "-"


# 실행마다 한 줄인 마크다운 표. docs/experiments.md 결과 절의 표와 같은 열이다: 오검출률 = 표시된 맞은 번호 /
# 맞은 번호, 틀린 번호 표시율 = 표시된 틀린 번호 / 틀린 번호, 조용히 나가는 틀린 번호 = (틀린 번호 − 표시된
# 틀린 번호) / 예측한 번호를 표시 전과 후로.
def guard_table(got: dict[str, dict]) -> str:
    lines = [
        "| 실행 | 예측한 번호 | 맞음 | 틀림 | 표시된 맞은 번호 (오검출률) | 표시된 틀린 번호 "
        "| 조용히 나가는 틀린 번호: 표시 전 → 후 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, g in got.items():
        n, ok, bad = g["predicted"], g["correct"], g["wrong"]
        fc, fw = g["flagged_correct"], g["flagged_wrong"]
        lines.append(
            f"| `{name}` | {n} | {ok} | {bad} | {fc} ({_pct(fc, ok)}%) | {fw}/{bad} ({_pct(fw, bad)}%) "
            f"| {bad}/{n} ({_pct(bad, n)}%) → {bad - fw}/{n} ({_pct(bad - fw, n)}%) |"
        )
    return "\n".join(lines)


# 표 아래에 적는 세부: 유형별 수, 사유별 수, 조용히 나가는 틀린 번호의 모양과 바뀐 숫자, 값 그대로 규칙의 추가
# 표시, 답에 없는 정답 번호.
def guard_detail(got: dict[str, dict]) -> str:
    lines = []
    for name, g in got.items():
        lines.append(f"\n{name}")
        for t, c in g["by_type"].items():
            lines.append(
                f"  {t}: predicted {c['predicted']}, correct {c['correct']} "
                f"(flagged {c['flagged_correct']}), wrong {c['wrong']} (flagged {c['flagged_wrong']})"
            )
        lines.append(f"  flagged wrong by reason: {g['flagged_wrong_reasons']}")
        lines.append(f"  silent wrong by shape: {g['silent_wrong_shapes']}")
        lines.append(f"  silent one-character changes (gold>predicted): {g['silent_wrong_one_char_changes']}")
        lines.append(f"  written-format rule would also flag: {g['written_format_rule_would_also_flag']}")
        lines.append(
            f"  gold ids not predicted: {g['gold_ids_not_predicted']}, rows without record: "
            f"{g['rows_without_record']}"
        )
    return "\n".join(lines)


# 명령줄 입구: 실행 폴더 여럿을 받아 표와 세부를 찍는다. --show는 조용히 나간 틀린 번호와 오검출 목록을 건마다
# 찍고, --json은 수치 전체를 파일로 남긴다(결과 절의 수치 출처).
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="기록된 평가 실행에 번호 확인 검사를 대 본다(모델이 필요 없다).")
    ap.add_argument("runs", nargs="+", help="reports/<run_id> 실행 폴더들")
    ap.add_argument("--show", action="store_true", help="표시 없이 나간 틀린 번호와 오검출 목록도 찍는다")
    ap.add_argument("--json", help="수치 전체를 이 파일에도 JSON으로 쓴다")
    args = ap.parse_args(argv)
    got = {str(r): guard_report(r) for r in args.runs}
    print(guard_table(got))
    print(guard_detail(got))
    if args.show:
        for name, g in got.items():
            print(f"\n{name}")
            for item_id, t, value, golds, shape, change in g["silent_wrong"]:
                print(f"  silent {item_id} {t} {value!r} gold {golds} ({shape} {change})")
            for item_id, t, value, reasons in g["false_flags"]:
                print(f"  false flag {item_id} {t} {value!r} {reasons}")
    if args.json:
        Path(args.json).write_text(json.dumps(got, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
