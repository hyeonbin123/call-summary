"""모델 없이 만드는 음성 조건 보고서.

asr_report survival DATA...   값 유형별로, 전사 파일마다 아직 찾을 수 있는 정답 값의 비율
                              (입력 분석: 어떤 모델도 데이터를 보기 전에 돌린다)
asr_report unfound RUN...     전사에서 찾을 수 없는 예측 값을 복원(정답과 같음)과 지어낸 값으로 나눠
                              유형별로 센다(4단계 규칙). 그중 빈 값과 자리 표시 값("없음", "알 수 없음" ...)도
                              함께 센다
asr_report tracking-forms DATA...
                              쓴 발화의 운송장번호를 인식기가 각각 어떤 꼴로 적었는지
                              (음성 조건 v2의 입력 분석)

값은 채점과 같은 방식으로 찾는다: `values.occurs_in_transcript`, `*-asr` 분할은 spoken=True.
"""

# [파일 역할] 음성 조건(4단계, v2, v3)의 분석 가운데 모델 없이 돌릴 수 있는 세 가지를 모은 명령줄 도구다.
# survival은 전사 파일에 정답 값이 얼마나 남았는지, unfound는 평가 실행에서 전사에 없는 예측 값을 복원/지어낸
# 값으로 나눈 수, tracking-forms는 인식기가 운송장번호를 어떤 꼴로 적었는지를 센다.
# [왜 필요한가] 음성 인식을 거치면 정답 값이 전사에서 깨질 수 있어, 모델 점수가 떨어진 것이 모델 탓인지 입력
# 탓인지 가려야 한다. 그래서 모델을 돌리기 전에 입력(전사)에 값이 얼마나 남았는지를 먼저 적고("입력 분석"),
# 모델이 전사에 없는 값을 적었을 때 학습에서 본 이름으로 바로잡은 것(복원)인지 지어낸 것인지를 나눈다(4단계
# 규칙 "복원과 환각을 나눠 센다").
# [설계 이유] 값 찾기는 채점과 같은 함수(values.occurs_in_transcript, -asr 분할이면 spoken=True)를 써서, 분석
# 수치와 채점 수치가 같은 기준이 되게 한다. 결과를 마크다운 표로 찍어 docs/experiments.md에 그대로 붙인다.

from __future__ import annotations

# SequenceMatcher: 쓴 글과 들린 글을 글자 단위로 맞춰, 잃은 운송장번호가 들린 글에서 어떻게 적혔는지 그 부분을
# 찾는 데 쓴다(heard_span).
import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

# value_survival: 정답 값마다 전사에서 찾아지는지. unfound_split: 전사에 없는 예측 값을 복원/지어낸 값으로
# 나눈다. 둘 다 채점기(scoring.py)에 있어 채점과 같은 정규화를 쓴다.
from .dataset import load_items, read_jsonl
from .domains import DOMAINS
from .schema import AfterCallRecord
from .scoring import unfound_split, value_survival
from .values import occurs_in_transcript


# 전사 파일마다 값 유형별 [찾은 수, 전체 수]와 모든 정답 값이 남은 건의 수를 센다. 음성 조건 파일(분할 이름이
# -asr로 끝남)은 번호를 쪼개 적은 꼴도 찾는 spoken 규칙을 쓴다.
# "모든 값이 남은 건"은 학생이 완벽히 맞힐 수 있는 상한에 가깝다. 예: test-d-v2 81.7%인데 서빙 학생 exact가
# 79.3이라 학생이 인식기의 천장 가까이 있다는 근거가 됐다(docs/experiments.md "음성 조건 v3" 왜).
def survival(paths: list[str | Path]) -> dict[str, dict]:
    """파일 경로별 {path: {"items", "all_found", "types": {type: [found, n]}}}."""
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
            # 건의 모든 유형, 모든 값이 찾아질 때만 1을 더한다.
            all_found += all(all(v) for v in got.values())
        out[str(path)] = {"items": len(items), "all_found": all_found, "types": dict(sorted(types.items()))}
    return out


# 비율을 백분율 소수 한 자리로. 분모가 0이면 "-".
def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}" if b else "-"


# survival 결과를 유형 × 파일 표로 만든다. 칸은 "보존율 (찾은 수/전체)", 마지막 줄은 모든 값이 남은 건의 비율.
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


# PLACEHOLDERS: 값 대신 적힌 자리 표시 낱말(없음, 알 수 없음, 미정, null, n/a 등). 공백을 지우고 소문자로 바꿔
# 비교한다. 2026년 비교 대상이 빠진 값 자리에 "없음"을 적는 일이 있어, 빈 값과 함께 따로 셌다(커밋
# afd83a2·e9dc490). 자리 표시 값은 빈 값을 채운 것의 글 형태다(3단계 14B는 빈 값을 채웠다). 둘 다 지어낸 값에
# 들어가므로 따로 세어 보인다.
PLACEHOLDERS = frozenset(
    {
        "없음",
        "해당없음",
        "알수없음",
        "모름",
        "미상",
        "불명",
        "미정",
        "미확인",
        "미확정",
        "미제공",
        "미입력",
        "미지정",
        "미언급",
        "null",
        "none",
        "n/a",
        "-",
    }
)


# 값이 자리 표시 낱말인지(공백 무시, 대소문자 무시).
def is_placeholder(value: str) -> bool:
    return "".join(value.split()).lower() in PLACEHOLDERS


# 평가 실행 하나에서 예측한 핵심 값 수, 전사에 없는 값의 복원/지어낸 값 유형별 수, 그중 빈 값과 자리 표시 값의
# 수를 센다.
# 데이터는 실행 manifest에 적힌 파일을 쓴다(또는 data 인자). 스키마를 통과하지 못해 예측 기록이 없는 건은
# 건너뛴다.
def unfound(run: str | Path, data: str | Path | None = None) -> dict:
    """평가 실행 하나(reports/<run_id>)의 복원 값과 지어낸 값. 그 실행의 데이터셋에 대 본다."""
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    items = {it.item_id: it for it in load_items(data or manifest["data"])}
    predicted = empty = placeholder = 0
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
        # 빈 값·자리 표시 값은 "전사에 없는 값" 가운데서만 센다. 전사에서 찾아지는 값은 건너뛴다. 업종 목록에
        # 없는 유형은 text 종류로 보고 찾는다.
        kinds = {et.label: et.kind for et in DOMAINS[it.domain].entity_types}
        for e in pred.entities:
            if occurs_in_transcript(kinds.get(e.type, "text"), e.value, it.transcript, spoken):
                continue
            if not e.value.strip():
                empty += 1
            elif is_placeholder(e.value):
                placeholder += 1
    return {
        "predicted": predicted,
        "restored": dict(restored.most_common()),
        "invented": dict(invented.most_common()),
        "empty": empty,  # 찾지 못한 값 중 빈 문자열(모두 지어낸 값). 3단계와 같은 방식으로 센다
        "placeholder": placeholder,  # 찾지 못한 값 중 자리 표시 낱말(모두 지어낸 값)
    }


# unfound 결과를 실행마다 한 줄로: 예측한 값, 전사에 없는 값(비율), 복원, 지어낸 값, 그중 빈 값, 빈 값을 뺀
# 비율, 자리 표시 값, 둘 다 뺀 비율. 빈 값을 뺀 열은 3단계 14B처럼 빈 값을 채워 넣은 모델의 환각을 같은
# 기준으로 견주려고 둔다(3단계 14B는 test-c 환각 123건 중 80건이 빈 값이었다, docs/experiments.md).
def unfound_table(got: dict[str, dict]) -> str:
    lines = [
        "| 실행 | 예측한 값 | 전사에 없는 값 | 복원 (정답과 같음) | 지어낸 값 | 그중 빈 값 "
        "| 빈 값을 뺀 전사에 없는 값 | 그중 자리 표시 값 | 빈 값과 자리 표시 값을 뺀 것 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in got.items():
        n = g["predicted"]
        r, i = sum(g["restored"].values()), sum(g["invented"].values())
        empty, placeholder = g.get("empty", 0), g.get("placeholder", 0)

        def detail(c: dict) -> str:
            return ", ".join(f"{t} {k}" for t, k in c.items())

        lines.append(
            f"| `{name}` | {n} | {_pct(r + i, n)}% ({r + i}) | {_pct(r, n)}% ({r}: {detail(g['restored'])}) "
            f"| {_pct(i, n)}% ({i}: {detail(g['invented'])}) | {empty} "
            f"| {_pct(r + i - empty, n)}% ({r + i - empty}) | {placeholder} "
            f"| {_pct(r + i - empty - placeholder, n)}% ({r + i - empty - placeholder}) |"
        )
    return "\n".join(lines)


# _TRACKING: 쓴 글의 운송장번호 꼴(숫자 4-4-4). _HANGUL_DIGIT_NAMES: 한글 숫자 이름이 네 자 이상 이어진
# 꼴("오팔구일"). _HANGUL: 한글 음절.
# _RANGE_WORD: v1 조건에서 운송장번호 뒤 두 묶음이 "A에서 B" 범위로 읽혀 생긴 가짜 "에서". v2에서 이것이
# 사라졌는지 따로 센다(dev 112 → 1, docs/experiments.md "음성 조건 v2").
_TRACKING = re.compile(r"\d{4}-\d{4}-\d{4}")
_HANGUL_DIGIT_NAMES = re.compile(r"[공영일이삼사오육칠팔구]{4,}")
_HANGUL = re.compile(r"[가-힣]")
_RANGE_WORD = "에서"  # v1의 범위 읽기("A에서 B"). 표의 따로 된 행에서 센다


# 쓴 글의 [start, end) 구간(번호)에 맞춰지는 들린 글의 부분을 찾는다. difflib의 편집 연산(opcodes)에서 시작·끝
# 위치가 들어 있는 연산을 찾아 들린 글의 위치로 옮기고(같은 구간이면 오프셋 그대로, 다르면 연산 경계), 양 끝에
# 붙은 숫자까지 넓혀 번호 가장자리에 더해진 숫자도 놓치지 않는다.
def heard_span(written: str, heard: str, start: int, end: int) -> str:
    """written[start:end]에 (difflib로 글자 단위로) 맞춰지는 `heard`의 부분. 양쪽에 붙은 숫자까지
    넓혀, 가장자리에 끼어든 숫자도 남긴다."""
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


# 운송장번호 하나가 들린 꼴을 분류한다(앞의 것이 맞으면 거기서 끝난다): 그대로 있음 → 채점의 spoken 규칙으로
# 찾아짐(쉼표·공백으로 나뉜 묶음) → 들린 발화에 한글 숫자 이름이 네 자 이상 → 못 찾음. 못 찾은 것은 맞춰진
# 부분에 한글 음절이 있는지("에서"는 빼고 봄), 숫자가 적은지·많은지·수는 같고 다른지로 나눈다.
# 2026-10-04 검토에서 처음 표가 "651일 5502"처럼 일부만 한글로 적힌 번호를 세지 못한 것을 찾아, 이 세부 분류를
# 커밋한 도구로 다시 셌다(docs/experiments.md 바로잡음).
def tracking_form(number: str, written: str, heard: str, at: int = 0) -> tuple[str, str]:
    """`written`(위치 `at`)에서 말하고 `heard`로 들린 운송장번호 하나의 (꼴, 들린 부분).

    꼴: "as_written"; "split_found"(채점의 spoken 규칙으로 찾음, 예: 쉼표로 나뉜 묶음);
    "hangul_digit_names"(못 찾음, 들린 발화에 한글 숫자 이름 4자 이상); 그 밖은 못 찾음이며, 번호에
    맞춰진 들린 부분으로 나눈다: "lost_hangul"(한글 음절이 있음, "에서"는 빼고 봄, 예: "651일 5502"),
    "lost_fewer" / "lost_more"(번호보다 숫자가 적음 / 많음), "lost_changed"(숫자 수는 같음).
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


# 분류 이름의 고정 순서. 표의 행 순서와 0으로 시작하는 카운터에 쓴다.
TRACKING_FORMS = (
    "as_written",
    "split_found",
    "hangul_digit_names",
    "lost_hangul",
    "lost_fewer",
    "lost_more",
    "lost_changed",
)


# 파일의 모든 건에서, 발화마다 쓴 글(meta.asr.written)의 운송장번호를 찾아 들린 글(turns의 text)과 견주어
# 분류를 센다. 못 찾은 번호는 [건 id, 번호, 분류, 들린 부분]으로 모아 --show-lost로 볼 수 있다.
# meta.asr.written이 있어야 하므로 asr_condition.py나 asr_arms build가 만든 음성 조건 파일에만 쓸 수 있다.
def tracking_forms(paths: list[str | Path]) -> dict[str, dict]:
    """{path: {"numbers", <form>: count..., "heard_has_에서", "lost": [[item_id, number, form, span]]}}.

    발화마다 쓴 글(meta.asr.written)의 모든 운송장번호를 들린 글과 견준다."""
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
                    # 번호마다 들린 발화에 "에서"가 있는지 센다(bool을 더해 개수로).
                    in_range_word += _RANGE_WORD in heard
                    if form.startswith("lost_"):
                        lost.append([it.item_id, m.group(), form, span])
        out[str(path)] = {"numbers": numbers, **n, "heard_has_에서": in_range_word, "lost": lost}
    return out


# 표의 행 이름(한국어). 앞에 공백을 둔 행은 "찾지 못함"의 세부 분류다.
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


# tracking-forms 결과를 행 × 파일 표로 만든다. "찾지 못함"은 lost 목록의 길이로 낸다.
def tracking_forms_table(got: dict[str, dict]) -> str:
    lines = ["| | " + " | ".join(f"`{p}`" for p in got) + " |", "|---|" + "---|" * len(got)]
    for key, label in _TRACKING_ROWS.items():
        cells = [str(len(g["lost"]) if key == "lost" else g[key]) for g in got.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 하위 명령 셋(survival, unfound, tracking-forms)을 고른다. 모두 표를 찍고, --json을 주면 원래 숫자를 파일로도
# 남긴다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survival", help="전사 파일에 아직 찾을 수 있는 정답 값")
    s.add_argument("data", nargs="+")
    u = sub.add_parser("unfound", help="평가 실행의 복원 값과 지어낸 값")
    u.add_argument("runs", nargs="+")
    t = sub.add_parser("tracking-forms", help="인식기가 운송장번호를 적은 꼴")
    t.add_argument("data", nargs="+")
    t.add_argument("--show-lost", action="store_true", help="못 찾은 번호와 그 들린 부분도 모두 찍는다")
    for p in (s, u, t):
        p.add_argument("--json", help="원래 숫자를 이 파일에도 쓴다")
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
