# [테스트 묶음] 1단계 데이터 만들기의 코드 검사를 모델 없이 확인한다. 주로 gen.py(교사 지시문, 생성한
# 대화·요약의 자동 검사, 재시도)를 보고, 같은 파일에 train.py(업종 고르게 고르기, 프롬프트 가리기),
# judge.py(판정 답 읽기와 지표), finalize.py(분할 확정)의 작은 테스트도 함께 있다.
# [왜 중요한가] 정답은 명세에서 나오고 대화는 교사 모델이 쓴다. 대화가 명세와 어긋나면(값이 빠짐, 명세에 없는
# 금액·번호가 끼어듦, 명세에 없는 본인 확인) 정답 라벨에 잡음이 생기므로, 자동 검사가 이런 대화를 빠짐없이
# 버려야 한다(docs/experiments.md 1단계 규칙 "자동 검사").

from call_summary.dataset import Item
from call_summary.gen import check_dialogue, check_summary, generate_item, writer_prompt
from call_summary.providers import ScriptedProvider
from call_summary.specs import EV_CORRECTION, make_spec, make_specs
from call_summary.train import IGNORE, encode, take_balanced


# 말실수 사건(고객이 틀린 번호를 먼저 말함)이 든 명세를 dev 쇼핑몰 번호 순서로 찾는다. 명세는 결정적이라 매번
# 같은 것이 나온다.
def _spec_with_correction():
    for i in range(200):
        s = make_spec("dev", "shop", i)
        if s.distractor is not None:
            return s
    raise AssertionError("no spec with a correction event")


# 검사를 통과하도록 손으로 만든 대화: 상담원으로 시작하고, 명세에 본인 확인이 있으면(또는 verify=True)
# 생년월일 확인을 넣고, 말실수 값 뒤에 다시 묻기, 모든 슬롯 값을 글자 그대로 말하기, 최소 10턴 채우기.
# extra_text로 문제가 되는 문장을 끝에 더해 검사를 시험한다.
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


# 교사에게 주는 지시문에 모든 슬롯 값과 말실수 값이 그대로 들어가는지 확인한다. 지시문에 없는 값은 교사가 쓸
# 수 없어 대화가 검사에서 버려진다.
def test_writer_prompt_contains_values_and_correction():
    s = _spec_with_correction()
    p = writer_prompt(s)
    for sv in s.slots:
        assert sv.value in p
    assert s.distractor.value in p
    assert EV_CORRECTION in s.events


# 깨끗한 대화는 문제 0개이고, 명세에 없는 금액·날짜·시각·번호가 끼어들거나, 슬롯 값이 빠지거나, 고객이 먼저
# 말하면 각각의 문제 이름이 나오는지 확인한다(규칙: 명세에 없는 금액·날짜·시각·번호가 없음, 명세의 값이 모두
# 나옴, 화자 형식).
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


# 본인 확인에서 말한 생년월일 여섯 자리(950302)는 번호 꼴 검사에 걸리지 않아야 한다. 걸리면 본인 확인이 있는
# 대화가 모두 "명세에 없는 번호"로 버려진다.
def test_birth_date_digits_are_not_ids():
    s = next(
        make_spec("dev", "telecom", i) for i in range(50) if "verify" in make_spec("dev", "telecom", i).events
    )
    assert check_dialogue(s, _turns(s, "생년월일은 950302입니다.")) == []


# 참고 요약 검사: 모든 핵심 값을 담은 요약은 통과하고, 값이 빠진 요약은 "misses" 문제가 나온다(규칙: 요약이
# 모든 핵심 값을 담음). 값은 채점과 같은 정규화로 찾는다.
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


# 교사의 첫 답이 JSON이 아니면 다시 시도하고, 두 번째 대화와 첫 요약이 통과하면 건을 받아들이는지, 시도 횟수가
# 통계에 남는지 본다(규칙: 대화·요약 각각 최대 3번 시도). ScriptedProvider가 정해 둔 답을 차례로 낸다.
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


# train.take_balanced: 7건을 고르면 업종을 돌아가며 골라 쇼핑몰·택배가 각각 2건 이상이고, n이 없으면 전체를
# 그대로 쓴다(데이터 크기 곡선 500건 학습에 쓴 함수).
def test_take_balanced():
    specs = make_specs("train", ("shop", "telecom", "parcel"), 10)
    items = [Item(item_id=s.spec_id, split="train", domain=s.domain, spec=s, turns=[]) for s in specs]
    got = take_balanced(items, 7)
    doms = [it.domain for it in got]
    assert len(got) == 7 and doms.count("parcel") >= 2 and doms.count("shop") >= 2
    assert take_balanced(items, None) == items


# 토크나이저 대용: 채팅 템플릿은 메시지를 잇고 "<gen>"을 붙이며, 글자마다 id 하나를 준다. 진짜 Qwen3
# 토크나이저 없이 encode의 가리기 규칙만 본다.
class _FakeTok:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(m["content"] for m in messages) + "<gen>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 for c in text]}


# train.encode: 프롬프트 위치의 라벨은 모두 IGNORE이고 그 뒤(정답 JSON + 끝 토큰)는 입력 id와 같아 답에서만
# 손실이 생기는지, max_len을 넘으면 None으로 빠지는지 확인한다(3단계 규칙: 손실은 답 부분만).
def test_encode_masks_prompt():
    s = make_spec("dev", "shop", 2)
    it = Item(item_id=s.spec_id, split="dev", domain="shop", spec=s, turns=_turns(s), summary="요약.")
    ex = encode(_FakeTok(), it, max_len=100_000)
    n_prompt = sum(1 for x in ex["labels"] if x == IGNORE)
    assert n_prompt > 0 and len(ex["labels"]) == len(ex["input_ids"])
    assert ex["labels"][n_prompt:] == ex["input_ids"][n_prompt:]
    assert encode(_FakeTok(), it, max_len=10) is None


# judge.py: 판정 답에서 사실 수만큼의 판정과 어긋난 문장 수를 읽고, 개수가 다르거나 정해진
# 낱말(포함·누락·틀림)이 아니면 실패(None)로 친다. 사실 재현율(micro), wrong_rate(틀린 사실이나 어긋난 문장이
# 있는 요약 비율), 두 채점자의 사실 단위 일치율을 손 계산 값과 맞춰 본다. 요약이 비면 판정 모델을 부르지 않고
# 재현율 0이다.
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


# 교사가 대화 중간에 중국어로 바뀐 일이 있어(파일럿, docs/experiments.md 1단계) 한자·가나가 든 대화를 버린다.
def test_foreign_script_is_rejected():
    s = make_spec("dev", "telecom", 1)
    assert "non-Korean script (Han/kana)" in check_dialogue(s, _turns(s, "明白了您的需求"))


# finalize.take_first: 입력 순서와 상관없이 업종마다 명세 번호 순서로 통과한 앞의 N건을 고른다(거절된 00003은
# 건너뛰고 00005까지). 규칙이 "업종마다 번호 순서로 자동 검사를 통과한 앞의 N건"이라 고르는 데 결과를 보지
# 않는다. id_values는 번호를 정규화해(하이픈 제거) train과 test의 번호 겹침을 비교할 수 있게 한다.
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


# 요청이 시간 초과 등으로 실패해도 생성 전체가 죽지 않고 시도 한 번으로 세어 다음 시도로
# 간다(docs/experiments.md 1단계 진행 기록: 요청 실패는 시도 1회로 센다). 모두 실패하면 건 없이 이유를
# 돌려준다.
def test_generate_item_survives_request_errors():
    class Boom:
        name = "boom"

        def generate(self, messages, json_schema=None):
            raise TimeoutError("slow")

    s = make_spec("dev", "shop", 5)
    item, problems, stats = generate_item(s, Boom(), Boom(), max_tries=2)
    assert item is None and problems == ["request failed: TimeoutError"]
    assert stats["dialogue_tries"] == 2 and stats["failed_requests"] == 2


# 재시도마다 seed를 하나씩 바꿔(1000, 1001, 1002) 같은 출력이 되풀이되지 않게 하고, 거절한 마지막 출력을
# 통계에 남겨 거절 이유를 살펴볼 수 있게 한다. 처음에는 seed가 같아 재시도가 무의미했던 버그를 고친
# 것이다(docs/experiments.md 1단계 진행 기록 2026-09-26 20:20).
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


# 본인 확인은 정답 처리('본인 확인')이므로 대화의 본인 확인 여부가 명세와 정확히 같아야 한다. 명세에 있는데
# 대화에 없거나, 없는데 대화에 있으면 버린다. 본인 확인이 없는 명세의 지시문에는 "본인 확인 절차는 하지
# 않는다"를 넣는다(docs/experiments.md 1단계: 라벨 잡음을 찾아 검사와 지시문 w2를 더함, 커밋 f4d0efe).
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
