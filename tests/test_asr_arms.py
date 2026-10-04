"""call_summary.asr_arms: the CPU side of speech condition v3 (no model, no audio)."""

import json

import pytest

from call_summary import asr_arms
from call_summary.dataset import Item, load_items, write_jsonl
from call_summary.domains import DOMAINS
from tests.test_scoring import _item

TRACKING_SAID = "육이구일, 칠팔칠칠, 오일육팔"
NAMES = "공일이삼사오육칠팔구"


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


def _parcel(i: int, said: str) -> Item:
    it = _item(i, "parcel", category="배송 조회")
    tracking = next(e.value for e in it.gold().entities if e.type == "운송장번호")
    others = " ".join(f"{e.type} {e.value}" for e in it.gold().entities if e.type != "운송장번호")
    it.turns = [
        {"speaker": "상담원", "text": f"새싹택배입니다. {others}"},
        {"speaker": "고객", "text": f"운송장 {tracking}로 조회해 주세요."},
        {"speaker": "고객", "text": "..."},
    ]
    return it, tracking, said


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


def _hyp(items, text_of):
    return {
        f"{it.item_id}:{i}": {"key": f"{it.item_id}:{i}", "heard": text_of(it, i, t["text"])}
        for it in items
        for i, t in enumerate(it.turns)
        if t["text"] != "..."
    }


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


def test_cer_and_the_number_counters():
    assert asr_arms.cer_counts("운송장 6291-7877-5168로", "운송장 6291, 7877, 5168 로") == (0, 16)
    assert asr_arms.cer_counts("가나다", "가다") == (1, 3)
    assert asr_arms.cer_counts("", "") == (0, 0)
    written, heard = "6291-7877-5168로 조회하고 주문은 D4477838", "6291 7877 5268로 조회하고 주문은 D4477838"
    assert asr_arms.wrong_digit_runs(written, heard) == (1, 2)
    assert asr_arms.hangul_digit_runs(f"운송장 {TRACKING_SAID}, 일이 있어요") == 3


def test_arm_names_say_whether_the_identifier_rule_is_applied():
    assert asr_arms.uses_itn("Q+I") and not asr_arms.uses_itn("Q") and not asr_arms.uses_itn("T")


def test_cli_context_writes_json(tmp_path):
    out = tmp_path / "ctx.json"
    assert asr_arms.main(["context", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == asr_arms.contexts()
