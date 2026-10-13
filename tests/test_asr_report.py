# [테스트 묶음] 모델 없이 도는 음성 조건 분석 도구 call_summary.asr_report와 그 바탕인 scoring.unfound_split을
# 검증한다. survival(정답 값 보존율), unfound(전사에 없는 예측 값을 복원·지어낸 값으로 나누고 빈 값·자리 표시
# 값을 따로 셈), tracking-forms(인식기가 운송장번호를 어떤 꼴로 적었는지 분류)를 다룬다.
# [왜 중요한가] 이 도구의 수치가 문서의 표(4단계 입력 분석, 음성 조건 v2의 운송장번호 표, 2026 비교 대상의
# "없음" 자리 표시 값)로 그대로 들어간다. v2 입력 분석의 운송장번호 표는 처음에 저장소 밖 스크립트로 셌다가 이
# 도구로 옮겨 커밋된 데이터만으로 다시 나오게 했다(커밋 58520e3).

import json

# _item(tests/test_scoring.py): 정답 값을 모두 말하는 짧은 가짜 건. 발화를 바꿔 "인식 뒤 값이 사라진 전사"를
# 만든다.
from call_summary import asr_report
from call_summary.dataset import write_jsonl
from call_summary.schema import Entity, ParseResult
from call_summary.scoring import score_item, unfound_split
from tests.test_scoring import _item


# 정답 기록에 예측 값(extra)을 더한 예측 기록을 만든다.
def _pred(it, extra=()):
    gold = it.gold()
    return gold.model_copy(update={"entities": [*gold.entities, *extra]})


# 전사에 없는 예측 값이 정답과 같으면 복원, 아니면 지어낸 값으로 나뉘는지 확인한다(4단계 규칙 "복원과 환각을
# 나눠 센다").
# - 값이 모두 전사에 있으면 정답 값은 어느 쪽에도 들지 않고, 정답에 없는 금액과 업종에 없는 유형만 지어낸
#   값이다.
# - 값이 사라진 전사(silent)에서는 정답을 그대로 옮긴 값이 모두 복원으로 센다(학습한 이름으로 인식 오류를
#   바로잡는 경우).
# - 복원 + 지어낸 값의 합이 채점기의 환각 수(entity_hallucinated)와 같다. 두 지표가 같은 "전사에 없는 값"을
#   나눈 것임을 보장한다.
def test_unfound_values_split_into_restored_and_invented():
    it = _item(0)  # shop: every gold value is said in the transcript
    gold = it.gold()
    invented = Entity(type="금액", value="999,900원")
    unknown = Entity(type="좌석번호", value="12A")  # not a type of this domain
    pred = _pred(it, [invented, unknown])

    restored, made_up = unfound_split(it.domain, gold, pred, it.transcript)
    assert not restored and made_up == {"금액": 1, "좌석번호": 1}

    silent = "상담원: 네 도토리마켓입니다."  # the values did not survive: copying gold back is a restoration
    restored, made_up = unfound_split(it.domain, gold, pred, silent)
    assert sum(restored.values()) == len(gold.entities)
    assert made_up == {"금액": 1, "좌석번호": 1}
    parsed = ParseResult(json_ok=True, schema_ok=True, record=pred)
    score = score_item(it.item_id, it.domain, gold, parsed, silent, spoken=True)
    assert score.entity_hallucinated == sum(restored.values()) + sum(made_up.values())


# survival이 파일마다 유형별 [찾은 값, 전체]와 모든 값이 남은 건 수를 세고, 표에 백분율로 나오는지 본다. 값을
# 모두 말한 파일은 100%, 값이 하나도 없는 파일은 0%다. 이 표는 모델을 돌리기 전에 입력 조건을 적는 "입력
# 분석"에 쓴다.
def test_survival_table_per_file(tmp_path):
    it = _item(0)
    heard = _item(0)
    heard.turns = [{"speaker": "상담원", "text": "네 도토리마켓입니다."}]
    write_jsonl(tmp_path / "v1.jsonl", [it.to_dict()])
    write_jsonl(tmp_path / "v2.jsonl", [heard.to_dict()])
    got = asr_report.survival([tmp_path / "v1.jsonl", tmp_path / "v2.jsonl"])
    types = {e.type for e in it.gold().entities}
    v1, v2 = got[str(tmp_path / "v1.jsonl")], got[str(tmp_path / "v2.jsonl")]
    assert v1["items"] == 1 and v1["all_found"] == 1
    assert v2["all_found"] == 0
    assert set(v1["types"]) == types and all(found == n for found, n in v1["types"].values())
    assert all(found == 0 for found, _ in v2["types"].values())
    table = asr_report.survival_table(got)
    assert "모든 값이 남은 건" in table and "100.0" in table and "0.0" in table


# 기록된 실행 하나(manifest + items.jsonl)를 읽어 unfound가 예측한 값 수, 복원, 지어낸 값을 세는지 확인한다.
# 분할이 "-asr"라 넓은 번호 규칙으로 찾는다.
def test_unfound_report_of_a_run(tmp_path):
    it = _item(0)
    it.split = "dev-asr"
    it.turns = [{"speaker": "상담원", "text": "네 도토리마켓입니다."}]
    data = tmp_path / "dev-asr.jsonl"
    write_jsonl(data, [it.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    pred = _pred(it, [Entity(type="금액", value="999,900원")])
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "pred": pred.model_dump()}])

    got = asr_report.unfound(run)
    n_gold = len(it.gold().entities)
    assert got["predicted"] == n_gold + 1
    assert sum(got["restored"].values()) == n_gold and got["invented"] == {"금액": 1}
    assert "복원" in asr_report.unfound_table({"run": got})


# 빈 값("", 공백만)은 지어낸 값으로 들어가면서 empty로도 따로 센다. 3단계가 환각을 빈 값 포함·제외 두 가지로
# 보고했으므로(docs/experiments.md 3단계) 표에 빈 값을 뺀 비율이 함께 나와야 한다.
def test_unfound_report_counts_empty_values(tmp_path):
    # Stage 3 reported hallucination with and without empty values ("" filled into a type the call never
    # mentioned): empty values are unfound and invented, and the table gives the rate without them too.
    it = _item(0)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    extra = [
        Entity(type="금액", value=""),
        Entity(type="금액", value=" "),
        Entity(type="금액", value="999,900원"),
    ]
    pred = _pred(it, extra)
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "pred": pred.model_dump()}])

    got = asr_report.unfound(run)
    n = len(it.gold().entities) + 3
    assert got["predicted"] == n and got["invented"] == {"금액": 3} and got["empty"] == 2
    table = asr_report.unfound_table({"run": got})
    assert "빈 값" in table and f"{100 * 1 / n:.1f}% (1)" in table


# "없음", "알 수 없음", "미정" 같은 자리 표시 낱말은 placeholder로 따로 센다(띄어쓰기·앞뒤 공백 무시, "알수
# 없음"도 같음). 낱말 전체가 자리 표시일 때만이라 "없음 아님"은 그냥 지어낸 값이다. empty는 3단계와 같은
# 셈으로 그대로 두고, 표에는 빈 값만 뺀 비율과 둘 다 뺀 비율을 함께 낸다(커밋 e9dc490).
def test_unfound_report_counts_placeholder_values(tmp_path):
    # The 2026 comparators wrote a word for "no value" ("없음", "알 수 없음") where stage 3's 14B wrote "":
    # the same failure in text form. Placeholders are counted on their own so `empty` stays as stage 3
    # counted it, and the table gives the rate without both.
    it = _item(0)
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict()])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data)}), encoding="utf-8")
    extra = [
        Entity(type="금액", value="없음"),
        Entity(type="주문번호", value="알 수 없음"),
        Entity(type="금액", value=" 미정 "),
        Entity(type="금액", value=""),
        Entity(type="금액", value="999,900원"),
        Entity(type="금액", value="없음 아님"),  # not a placeholder word: an invented value
    ]
    pred = _pred(it, extra)
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "pred": pred.model_dump()}])

    got = asr_report.unfound(run)
    n = len(it.gold().entities) + 6
    assert got["predicted"] == n and got["invented"] == {"금액": 5, "주문번호": 1}
    assert got["empty"] == 1 and got["placeholder"] == 3
    assert asr_report.is_placeholder("알수 없음") and not asr_report.is_placeholder("")
    table = asr_report.unfound_table({"run": got})
    assert "자리 표시 값" in table
    assert f"{100 * 5 / n:.1f}% (5)" in table  # without empty values only, as before
    assert f"{100 * 2 / n:.1f}% (2)" in table  # without empty and placeholder values


# 운송장번호 하나가 든 쓴 글. 아래 테스트가 같은 문장을 인식기가 여러 꼴로 적은 경우를 흉내 낸다.
WRITTEN = "운송장 번호 6291-7877-5168로 받은 건이에요."


# tracking_form의 분류를 꼴마다 확인한다: 원래 꼴 그대로, 묶음이 쉼표로 나뉘었지만 채점 규칙으로 찾음, 한글
# 숫자 이름 네 자 이상, 그리고 찾지 못한 번호를 번호와 정렬한 구간으로 나눈 네 가지(한글 음절이 섞임, 숫자가
# 적음, 많음, 수는 같고 다름).
# 가장자리에 붙은 숫자는 구간에 넣고("52685"), 같은 발화의 다른 곳에 있는 날짜는 넣지 않는다. v1의 범위
# 읽기("에서")는 따로 세므로 "에서"만으로 한글 음절 꼴이 되지 않는다(커밋 58520e3).
def test_tracking_form_classes():
    def form(heard):
        return asr_report.tracking_form("6291-7877-5168", WRITTEN, heard, WRITTEN.index("6291"))

    assert form(WRITTEN) == ("as_written", "6291-7877-5168")
    assert form("운송장 번호 6291, 7877, 5168로 받은 건이에요.")[0] == "split_found"
    assert form("운송장 번호 육이구일 칠팔칠칠 오일육팔로 받은 건이에요.")[0] == "hangul_digit_names"
    # one digit heard as a Hangul syllable inside the number
    assert form("운송장 번호 629일 7877 5168로 받은 건이에요.") == ("lost_hangul", "629일 7877 5168")
    assert form("운송장 번호 629-7877-5168로 받은 건이에요.")[0] == "lost_fewer"
    # an inserted digit at the edge stays in the span; a date elsewhere in the utterance does not count
    assert form("운송장 번호 6291-7877-52685로 받은 건이에요.") == ("lost_more", "6291-7877-52685")
    dated = "운송장 번호 6291-7877-5268로 10월 20일에 받은 건이에요."
    assert form(dated) == ("lost_changed", "6291-7877-5268")
    # the v1 range reading has its own row: "에서" alone does not make the hangul class
    assert form("운송장 번호 6291 78717에서 5168로 받은 건이에요.") == ("lost_more", "6291 78717에서 5168")


# tracking_forms가 파일의 모든 발화를 쓴 글(meta.asr.written)과 들린 글로 짝지어 번호마다 분류하고, 찾지 못한
# 번호 목록과 "에서"가 들린 발화 수를 남기는지, 표에 행 이름대로 나오는지 확인한다.
def test_tracking_forms_per_file(tmp_path):
    it = _item(0, domain="parcel", category=None)
    it.split = "dev-asr"
    it.meta = {"asr": {"written": [WRITTEN, "네 확인했습니다.", WRITTEN]}}
    it.turns = [
        {"speaker": "고객", "text": "운송장 번호 6291-7877-5168로 받은 건이에요."},
        {"speaker": "상담원", "text": "네 확인했습니다."},
        {"speaker": "고객", "text": "운송장 번호 6291 78717에서 5168로 받은 건이에요."},
    ]
    path = tmp_path / "dev-asr.jsonl"
    write_jsonl(path, [it.to_dict()])
    got = asr_report.tracking_forms([path])[str(path)]
    assert got["numbers"] == 2 and got["as_written"] == 1 and got["lost_more"] == 1
    assert got["heard_has_에서"] == 1 and got["lost_hangul"] == 0
    assert got["lost"] == [[it.item_id, "6291-7877-5168", "lost_more", "6291 78717에서 5168"]]
    table = asr_report.tracking_forms_table({str(path): got})
    assert "| 찾지 못함 | 1 |" in table and "| 번호 수 | 2 |" in table
