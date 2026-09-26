from call_summary.dataset import Item
from call_summary.gen import check_dialogue, check_summary, generate_item, writer_prompt
from call_summary.providers import ScriptedProvider
from call_summary.specs import EV_CORRECTION, make_spec, make_specs
from call_summary.train import IGNORE, encode, take_balanced


def _spec_with_correction():
    for i in range(200):
        s = make_spec("dev", "shop", i)
        if s.distractor is not None:
            return s
    raise AssertionError("no spec with a correction event")


def _turns(spec, extra_text="", verify=None):
    values = [sv.value for sv in spec.slots]
    turns = [{"speaker": "상담원", "text": "안녕하세요, 도토리마켓 고객센터입니다."}]
    if verify if verify is not None else "verify" in spec.events:
        turns.append({"speaker": "고객", "text": "문의가 있어요."})
        turns.append({"speaker": "상담원", "text": "성함과 생년월일 여섯 자리 부탁드립니다."})
        turns.append({"speaker": "고객", "text": "김철수, 900101이요."})
    if spec.distractor is not None:
        turns.append({"speaker": "고객", "text": f"번호가 {spec.distractor.value}예요."})
        turns.append({"speaker": "상담원", "text": "조회가 안 되는데요, 다시 불러 주시겠어요?"})
    for v in values:
        turns.append({"speaker": "고객", "text": f"{v} 입니다."})
        turns.append({"speaker": "상담원", "text": "네, 확인했습니다."})
    while len(turns) < 10:
        turns.append({"speaker": "고객" if turns[-1]["speaker"] == "상담원" else "상담원", "text": "네."})
    if extra_text:
        turns.append({"speaker": "고객", "text": extra_text})
    return turns


def test_writer_prompt_contains_values_and_correction():
    s = _spec_with_correction()
    p = writer_prompt(s)
    for sv in s.slots:
        assert sv.value in p
    assert s.distractor.value in p
    assert EV_CORRECTION in s.events


def test_check_dialogue_accepts_clean_and_rejects_problems():
    s = _spec_with_correction()
    assert check_dialogue(s, _turns(s)) == []
    assert any("extra amount" in p for p in check_dialogue(s, _turns(s, "그리고 99,900원도 냈어요.")))
    assert any("extra date" in p for p in check_dialogue(s, _turns(s, "12월 25일에요.")))
    assert any("extra time" in p for p in check_dialogue(s, _turns(s, "오후 7시에 전화 주세요.")))
    assert any("extra id" in p for p in check_dialogue(s, _turns(s, "다른 주문은 D1234567이에요.")))
    missing = [t for t in _turns(s) if s.slots[0].value not in t["text"]]
    assert any("missing" in p for p in check_dialogue(s, missing))
    swapped = [{"speaker": "고객", "text": "여보세요"}] + _turns(s)
    assert "does not start with the agent" in check_dialogue(s, swapped)


def test_birth_date_digits_are_not_ids():
    s = next(
        make_spec("dev", "telecom", i) for i in range(50) if "verify" in make_spec("dev", "telecom", i).events
    )
    assert check_dialogue(s, _turns(s, "생년월일은 950302입니다.")) == []


def test_check_summary():
    s = make_spec("dev", "parcel", 3)
    full = (
        "고객은 "
        + ", ".join(sv.value for sv in s.slots)
        + " 건으로 문의함. 상담원은 확인 후 안내함. 고객은 알겠다고 함."
    )
    assert check_summary(s, full) == []
    assert any(
        "misses" in p
        for p in check_summary(s, "고객은 문의함. 상담원은 확인 후 안내함. 고객은 알겠다고 함. 끝.")
    )


def test_generate_item_retries_then_accepts():
    import json

    s = make_spec("dev", "shop", 5)
    good = json.dumps({"turns": _turns(s)}, ensure_ascii=False)
    summary = json.dumps(
        {
            "summary": "고객은 "
            + ", ".join(sv.value for sv in s.slots)
            + " 관련 문의함. 상담원이 처리함. 고객은 감사하다고 함."
        },
        ensure_ascii=False,
    )
    writer = ScriptedProvider(["not json", good])
    summarizer = ScriptedProvider([summary])
    item, problems, stats = generate_item(s, writer, summarizer)
    assert item is not None and problems == []
    assert stats["dialogue_tries"] == 2 and stats["summary_tries"] == 1
    assert item.gold().summary.startswith("고객은")


def test_take_balanced():
    specs = make_specs("train", ("shop", "telecom", "parcel"), 10)
    items = [Item(item_id=s.spec_id, split="train", domain=s.domain, spec=s, turns=[]) for s in specs]
    got = take_balanced(items, 7)
    doms = [it.domain for it in got]
    assert len(got) == 7 and doms.count("parcel") >= 2 and doms.count("shop") >= 2
    assert take_balanced(items, None) == items


class _FakeTok:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(m["content"] for m in messages) + "<gen>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 for c in text]}


def test_encode_masks_prompt():
    s = make_spec("dev", "shop", 2)
    it = Item(item_id=s.spec_id, split="dev", domain="shop", spec=s, turns=_turns(s), summary="요약.")
    ex = encode(_FakeTok(), it, max_len=100_000)
    n_prompt = sum(1 for x in ex["labels"] if x == IGNORE)
    assert n_prompt > 0 and len(ex["labels"]) == len(ex["input_ids"])
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    assert encode(_FakeTok(), it, max_len=10) is None


def test_judge_parse_and_metrics():
    from call_summary.judge import (
        SummaryScore,
        agreement,
        fact_recall,
        judge_summary,
        parse_judge,
        wrong_rate,
    )

    assert parse_judge('{"facts": ["포함", "누락"], "wrong_statements": 0}', 2) == (("포함", "누락"), 0)
    assert parse_judge('{"facts": ["포함"], "wrong_statements": 0}', 2) is None
    assert parse_judge('{"facts": ["예", "누락"], "wrong_statements": 0}', 2) is None
    a = [SummaryScore("x", ("포함", "누락"), 0, True), SummaryScore("y", ("포함", "틀림"), 1, True)]
    b = [SummaryScore("x", ("포함", "포함"), 0, True), SummaryScore("y", ("포함", "틀림"), 0, True)]
    assert fact_recall(a) == 0.5
    assert wrong_rate(a) == 0.5
    assert agreement(a, b) == 0.75
    s = make_spec("dev", "shop", 1)
    judge = ScriptedProvider(
        ['{"facts": ' + str(["포함"] * len(s.facts)).replace("'", '"') + ', "wrong_statements": 0}']
    )
    got = judge_summary(judge, s.spec_id, s, "대화", "요약입니다")
    assert got.ok and got.recall == 1.0
    assert judge_summary(judge, s.spec_id, s, "대화", "  ").recall == 0.0


def test_foreign_script_is_rejected():
    s = make_spec("dev", "telecom", 1)
    assert "non-Korean script (Han/kana)" in check_dialogue(s, _turns(s, "明白了您的需求"))


def test_take_first_keeps_index_order_per_domain():
    from call_summary.finalize import id_values, take_first

    specs = make_specs("dev", ("shop", "parcel"), 12)
    items = [Item(item_id=s.spec_id, split="dev", domain=s.domain, spec=s, turns=[]) for s in specs]
    items = [it for it in items if not it.item_id.endswith("00003")]  # a rejected spec
    got = take_first(list(reversed(items)), 5)
    assert [it.item_id for it in got if it.domain == "shop"] == [f"dev-shop-{i:05d}" for i in (0, 1, 2, 4, 5)]
    assert len(got) == 10
    parcel = next(it for it in items if it.domain == "parcel")
    assert id_values(parcel) and all("-" not in v for v in id_values(parcel))


def test_generate_item_survives_request_errors():
    class Boom:
        name = "boom"

        def generate(self, messages, json_schema=None):
            raise TimeoutError("slow")

    s = make_spec("dev", "shop", 5)
    item, problems, stats = generate_item(s, Boom(), Boom(), max_tries=2)
    assert item is None and problems == ["request failed: TimeoutError"]
    assert stats["dialogue_tries"] == 2 and stats["failed_requests"] == 2


def test_retries_use_a_different_seed_and_keep_the_rejected_text():
    from call_summary.providers import OllamaProvider

    class Seeded(OllamaProvider):
        def __init__(self):
            super().__init__(model="x")
            self.seen = []

        def generate(self, messages, json_schema=None):
            self.seen.append(self.seed)
            from call_summary.providers import Reply

            return Reply(text='{"turns": [{"speaker": "고객", "text": "짧다"}]}', latency_s=0)

    s = make_spec("dev", "shop", 5)
    w = Seeded()
    item, problems, stats = generate_item(s, w, w, max_tries=3, seed=1000)
    assert item is None and w.seen == [1000, 1001, 1002]
    assert stats["last_rejected"][0]["text"] == "짧다"


def test_verification_must_match_spec():
    with_v = next(
        make_spec("dev", "shop", i) for i in range(50) if "verify" in make_spec("dev", "shop", i).events
    )
    without = next(
        make_spec("dev", "parcel", i)
        for i in range(50)
        if "verify" not in make_spec("dev", "parcel", i).events
    )
    assert "spec verification missing" in check_dialogue(with_v, _turns(with_v, verify=False))
    assert check_dialogue(with_v, _turns(with_v)) == []
    assert check_dialogue(without, _turns(without)) == []
    assert "verification not in spec" in check_dialogue(without, _turns(without, verify=True))
    assert "본인 확인 절차" in writer_prompt(without) and "본인 확인 절차" not in writer_prompt(with_v)
