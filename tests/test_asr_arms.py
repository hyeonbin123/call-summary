"""call_summary.asr_arms: the CPU side of speech condition v3 (no model, no audio)."""

# [테스트 묶음] 음성 조건 v3(인식기 바꾸기, docs/experiments.md "음성 조건 v3")의 CPU 쪽 도구
# call_summary.asr_arms를 모델·소리 없이 검증한다. 업종 맥락 글(context), 팔마다 학생이 받을 데이터
# 만들기(build), 새 소리의 기준 인식기 글과 기록된 전사의 동일성(identity), 번호 발화만 보는 싼 관문(gate),
# 단계 A 비교·순위(compare, ranking), CER·숫자 세기 도우미를 다룬다.
# [왜 중요한가] 이 도구의 판정(go/stop, 통과, 순위)이 GPU 몇 시간짜리 다음 단계를 할지 정한다. 규칙을 측정
# 전에 고정했으므로(규칙 커밋 f5b2cce), 코드가 문서의 규칙과 같게 세는지 작은 손 데이터로 미리 확인한다.

import json

import pytest

# _item(tests/test_scoring.py): 명세로 만든 짧은 가짜 건. 여기서는 택배 건의 발화를 바꿔 인식기 글을 흉내
# 낸다.
from call_summary import asr_arms
from call_summary.dataset import Item, load_items, write_jsonl
from call_summary.domains import DOMAINS
from tests.test_scoring import _item

# TRACKING_SAID: 운송장번호 6291-7877-5168을 v2 조건처럼 숫자 하나씩 한글 이름으로 읽은 꼴(인식기가 이렇게
# 적으면 채점이 번호를 찾지 못한다). NAMES: i번째 글자가 숫자 i의 한글 이름(공=0 … 구=9)이라 숫자를 한글로
# 바꾸는 데 쓴다.
TRACKING_SAID = "육이구일, 칠팔칠칠, 오일육팔"
NAMES = "공일이삼사오육칠팔구"


# 업종 맥락(ctx-1)이 회사 이름 + 카탈로그 값 전부를 정해진 순서로 쉼표로 이은 글인지, 네 업종 모두 있는지, 두
# 번 불러도 같은지 확인한다. 이 글이 Th 팔의 hotwords와 Qc 팔의 프롬프트로 들어가므로, 내용이 바뀌면 측정
# 조건이 바뀐다(문서의 택배 예시와 같은 글을 그대로 단언한다).
def test_context_lists_the_company_and_the_whole_catalog():
    ctx = asr_arms.contexts()
    assert set(ctx) == set(DOMAINS)
    assert ctx["shop"].startswith("도토리마켓, 무선 이어폰 소리방울 2, 접이식 캠핑 의자")
    assert ctx["shop"].endswith("러닝화 250")
    assert ctx["telecom"].count(", ") == 16  # company + 6 plans + 5 devices + 5 add-ons
    assert "은하 탭 9" in ctx["telecom"] and "해외 로밍 패스" in ctx["telecom"]
    assert (
        ctx["parcel"] == "새싹택배, 전자레인지, 노트북, 김치 10kg, 도자기 그릇 세트, 의류 상자, "
        "책 상자, 전기 자전거 배터리, 화분"
    )
    assert asr_arms.contexts() == ctx


# 택배 "배송 조회" 건 하나를 만든다: 발화 1에 운송장번호, 발화 2는 "..."(소리 없는 발화로 쓴다). 건, 정답
# 운송장번호, 읽은 꼴을 튜플로 돌려준다.
def _parcel(i: int, said: str) -> tuple[Item, str, str]:
    it = _item(i, "parcel", category="배송 조회")
    tracking = next(e.value for e in it.gold().entities if e.type == "운송장번호")
    others = " ".join(f"{e.type} {e.value}" for e in it.gold().entities if e.type != "운송장번호")
    it.turns = [
        {"speaker": "상담원", "text": f"새싹택배입니다. {others}"},
        {"speaker": "고객", "text": f"운송장 {tracking}로 조회해 주세요."},
        {"speaker": "고객", "text": "..."},
    ]
    return it, tracking, said


# 1패스가 남기는 index.jsonl 행을 흉내 낸다: 발화마다 키, 기준 인식기 T의 글(heard, 여기서는 쓴 글 그대로),
# 같은 키의 v2 캐시 글, WAV 경로. "..." 발화는 소리가 없는 발화라 wav가 None이다.
def _index(items, split="dev"):
    rows = []
    for it in items:
        for i, t in enumerate(it.turns):
            silent = t["text"] == "..."
            rows.append(
                {
                    "key": f"{it.item_id}:{i}",
                    "split": split,
                    "item_id": it.item_id,
                    "turn": i,
                    "domain": it.domain,
                    "heard": "" if silent else t["text"],
                    "cached_heard": "" if silent else t["text"],
                    "wav": None if silent else f"wav/{i}.wav",
                    "audio_seconds": 0.0 if silent else 2.0,
                }
            )
    return rows


# 2패스가 팔마다 남기는 가설 파일(키 → 들린 글)을 흉내 낸다. text_of로 발화마다 들린 글을 바꿔 인식 오류를
# 만든다. 소리 없는 발화는 넣지 않는다.
def _hyp(items, text_of):
    return {
        f"{it.item_id}:{i}": {"key": f"{it.item_id}:{i}", "heard": text_of(it, i, t["text"])}
        for it in items
        for i, t in enumerate(it.turns)
        if t["text"] != "..."
    }


# build가 팔의 글을 학생이 받는 dev-asr 형식으로 만드는지 확인한다.
# - 이름이 +I가 아닌 팔(Q)은 인식기 글 그대로, +I 팔(Q+I)은 식별자 규칙(id-itn-1)을 거친 글("6291, 7877,
#   5168")을 받는다. 판정을 학생이 실제로 받는 글로 하기 때문이다(규칙: 모든 후보 팔은 같은 식별자 규칙을
#   거친다).
# - 소리 없는 발화는 빈 글, 분할 이름에 "-asr"가 붙고, meta.asr에 쓴 글·규칙 판·팔 이름이 남는다.
# - 기준 팔 T는 1패스 index의 글을 그대로 쓴다.
# - 가설 파일에 없는 발화가 있으면 조용히 넘어가지 않고 KeyError로 멈춘다(빠진 발화가 빈 글로 채점되는 일을
#   막는다).
def test_build_gives_the_arm_text_after_the_identifier_rule_only_for_plus_i_arms():
    it, tracking, _ = _parcel(0, TRACKING_SAID)
    index = asr_arms.by_turn(_index([it]))

    def hangul(_it, i, text):
        return text.replace(tracking, TRACKING_SAID)

    hyp = _hyp([it], hangul)
    raw = asr_arms.build([it], index, asr_arms.from_hyp(hyp), itn=False, meta={"arm": "Q"})[0]
    itn = asr_arms.build([it], index, asr_arms.from_hyp(hyp), itn=True, meta={"arm": "Q+I"})[0]
    assert raw["turns"][1]["text"] == f"운송장 {TRACKING_SAID}로 조회해 주세요."
    assert itn["turns"][1]["text"] == "운송장 6291, 7877, 5168로 조회해 주세요."
    assert raw["turns"][2]["text"] == "" and raw["split"] == "dev-asr"
    assert itn["meta"]["asr"]["written"] == [t["text"] for t in it.turns]
    assert itn["meta"]["asr"]["itn"] == "id-itn-1" and raw["meta"]["asr"]["itn"] is None
    assert itn["meta"]["asr"]["arm"] == "Q+I"
    base = asr_arms.build([it], index, asr_arms.from_index, itn=False, meta={"arm": "T"})[0]
    assert [t["text"] for t in base["turns"]] == [t["text"] for t in it.turns[:2]] + [""]
    with pytest.raises(KeyError):
        asr_arms.build([it], index, asr_arms.from_hyp({}), itn=False, meta={})


# 동일성 확인: 다시 합성한 소리에서 T가 적은 글이 기록된 v2 전사와 발화마다 같은지 센다. 모두 같으면 기록된 v2
# 전사를 T로 그대로 쓰고, 하나라도 다르면 새 T 글이 기준이 된다(문서 "소리" 절). 한 발화를 다르게 바꾸면 그 건
# id와 발화 번호가 different에 남는지도 본다. cache_same은 소리가 있는 발화만 세므로 2다.
def test_identity_compares_the_base_text_of_new_audio_with_the_recorded_transcripts():
    it, _, _ = _parcel(0, TRACKING_SAID)
    recorded = Item.from_dict(
        {**it.to_dict(), "split": "dev-asr", "turns": [*it.turns[:2], {**it.turns[2], "text": ""}]}
    )
    rows = _index([it])
    got = asr_arms.identity(asr_arms.by_turn(rows), [recorded])
    assert got == {"turns": 3, "same": 3, "different": [], "missing": 0, "cache_same": 2}
    rows[1]["heard"] = "운송장 다르게 들림"
    got = asr_arms.identity(asr_arms.by_turn(rows), [recorded])
    assert got["same"] == 2 and got["different"][0][:2] == [it.item_id, 1]


# 싼 첫 관문: 번호 발화만 남긴 index(1패스 --only-ids)에서 팔마다 정답 번호를 몇 개 찾는지 세고 go/stop을
# 정한다.
# - 운송장번호를 한글 숫자 이름으로 적은 팔(Q)은 채점 규칙으로 찾지 못하고, 같은 글에 식별자 규칙을 거친
#   팔(Q+I)은 3개 모두 찾는다.
# - 후보 하나라도 T보다 많이 찾으면 go, 아니면 stop이다. 참고 팔(reference)은 후보가 아니므로 그 팔만 T를
#   넘어도 stop이다(참고 팔은 T+I와 식별자 규칙 없는 인식기 글이다. 규칙상 후보 다섯만 관문을 연다, 커밋
#   3062500).
def test_the_cheap_gate_counts_gold_identifiers_found_in_the_identifier_turns():
    pairs = [_parcel(i, TRACKING_SAID) for i in range(3)]
    items = [p[0] for p in pairs]
    rows = [r for r in _index(items) if r["turn"] == 1]  # pass 1 with --only-ids keeps only these
    index = asr_arms.by_turn(rows)

    def hangul(it, i, text):
        tracking = next(e.value for e in it.gold().entities if e.type == "운송장번호")
        return text.replace(
            tracking, ", ".join("".join(NAMES[int(c)] for c in g) for g in tracking.split("-"))
        )

    base = {r["key"]: {**r, "heard": "운송장 번호 몰라요"} for r in rows}  # T lost every number
    arms = {
        "T": asr_arms.from_hyp(base),
        "Q": asr_arms.from_hyp(_hyp(items, hangul)),
        "Q+I": asr_arms.from_hyp(_hyp(items, hangul)),
    }
    got = asr_arms.gate(items, index, arms)
    assert got["T"]["found"] == 0 and got["Q"]["found"] == 0  # Hangul digit names are not found
    assert got["Q+I"]["found"] == 3 and got["Q+I"]["ids"] == 3
    assert got["Q+I"]["types"] == {"운송장번호": [3, 3]}
    assert asr_arms.gate_verdict(got) == "go"
    assert asr_arms.gate_verdict({"T": got["T"], "Q": got["Q"]}) == "stop"
    assert asr_arms.gate_verdict(got, reference={"Q+I"}) == "stop"  # a reference arm is no candidate


# build 결과와 같은 꼴의 데이터 파일을 만든다. lost에 든 건은 운송장번호 발화를 "운송장 번호 몰라요"로 바꿔
# 번호를 잃게 한다. 단계 A가 건마다 "모든 값이 남았는지"를 세므로 이것으로 팔마다 남는 건 수를 정해 둔다.
def _dataset(tmp_path, name, items, lost=()):
    """A built dataset where the items in `lost` lost their tracking number."""
    out = []
    for it in items:
        d = it.to_dict()
        d["split"] = "dev-asr"
        d["meta"] = {"asr": {"written": [t["text"] for t in it.turns]}}
        if it.item_id in lost:
            d["turns"] = [
                {**t, "text": "운송장 번호 몰라요"} if i == 1 else t for i, t in enumerate(it.turns)
            ]
        out.append(d)
    path = tmp_path / f"{name}.jsonl"
    write_jsonl(path, out)
    return path


# 단계 A 비교와 순위를 확인한다(T 28/40건 남음 기준).
# - 잃는 건 없이 10건을 얻은 팔(L+I)은 차이 +25%p, 짝지은 부트스트랩 하한 > 0이라 통과한다. 6건 얻고 6건 잃은
#   팔(N+I)은 차이 0이라 못 넘는다(규칙: 팔 − T 하한 > 0이면 통과).
# - 참고 팔(L)은 통과해도 순위에 들지 않는다.
# - 같은 결과의 카탈로그 팔(Qc+I)은 카탈로그를 쓰지 않는 팔 뒤에 선다(규칙: 하한 → 차이 → 카탈로그 아님 → 등록
#   순서).
# - CER은 잃은 발화가 많은 쪽이 크고, 표 출력에 팔 줄이 있으며, 만든 파일은 보통 데이터셋으로 읽힌다.
def test_compare_gates_each_arm_on_the_paired_survival_difference_and_ranks_the_passing_ones(tmp_path):
    items = [_parcel(i, TRACKING_SAID)[0] for i in range(40)]
    ids = [it.item_id for it in items]
    base = _dataset(tmp_path, "T", items, lost=ids[:12])  # 28/40 survive
    good = _dataset(tmp_path, "L+I", items, lost=ids[:2])  # gains 10, loses none
    catalog = _dataset(tmp_path, "Qc+I", items, lost=ids[:2])  # the same, but catalog-biased
    mixed = _dataset(tmp_path, "N+I", items, lost=ids[6:18])  # gains 6, loses 6
    reference = _dataset(tmp_path, "L", items, lost=ids[:1])  # passes, but only reported
    arms = {"L+I": good, "Qc+I": catalog, "N+I": mixed, "L": reference}
    got = asr_arms.compare(base, arms, catalog={"Qc+I"}, reference={"L"})
    assert got["L"]["passed"] and got["L"]["reference"]
    assert got["T"]["all_found"] == 28 and got["L+I"]["all_found"] == 38
    assert got["L+I"]["diff"] == pytest.approx(0.25) and got["L+I"]["low"] > 0 and got["L+I"]["passed"]
    assert got["N+I"]["diff"] == 0 and not got["N+I"]["passed"]
    assert got["L+I"]["types"]["운송장번호"] == [38, 40]
    assert asr_arms.ranking(got) == ["L+I", "Qc+I"]  # same bounds: the non-catalog arm first
    assert got["T"]["cer"] > got["L+I"]["cer"] > 0 and got["T"]["wrong_digit_runs"] == 0
    assert "| L+I |" in asr_arms.compare_table(got)
    load_items(good)  # the built files are ordinary datasets


# 2차 지표 도우미를 확인한다. CER은 공백·문장부호를 빼고 세므로 "6291-7877-5168로"와 "6291, 7877, 5168 로"는
# 편집 0이다. wrong_digit_runs는 들린 글의 4자리 이상 숫자 덩어리 중 쓴 글에 없는 것(5268)만 센다.
# hangul_digit_runs는 한글 숫자 이름 4자 이상 연속만 세어 "일이" 같은 낱말은 넣지 않는다.
def test_cer_and_the_number_counters():
    assert asr_arms.cer_counts("운송장 6291-7877-5168로", "운송장 6291, 7877, 5168 로") == (0, 16)
    assert asr_arms.cer_counts("가나다", "가다") == (1, 3)
    assert asr_arms.cer_counts("", "") == (0, 0)
    written, heard = "6291-7877-5168로 조회하고 주문은 D4477838", "6291 7877 5268로 조회하고 주문은 D4477838"
    assert asr_arms.wrong_digit_runs(written, heard) == (1, 2)
    assert asr_arms.hangul_digit_runs(f"운송장 {TRACKING_SAID}, 일이 있어요") == 3


# 팔 이름 규약: "+I"로 끝나는 팔만 식별자 규칙을 거친다(모듈 머리말).
def test_arm_names_say_whether_the_identifier_rule_is_applied():
    assert asr_arms.uses_itn("Q+I") and not asr_arms.uses_itn("Q") and not asr_arms.uses_itn("T")


# 명령줄 context 하위 명령이 contexts()와 같은 내용을 JSON 파일로 쓰는지 확인한다(outputs/asr3/context.json을
# 만드는 명령).
def test_cli_context_writes_json(tmp_path):
    out = tmp_path / "ctx.json"
    assert asr_arms.main(["context", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == asr_arms.contexts()
