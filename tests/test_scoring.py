# [테스트 묶음] 채점의 바탕을 검증한다: 답 읽기(schema.parse_reply), 건별 채점(scoring.score_item), 부트스트랩
# 구간과 짝지은 차이, 업종별 시스템 프롬프트, 평가 루프(evaluate.run_items)와 배치 순서, 프롬프트 지문 고정,
# 음성 조건의 넓은 번호 찾기, 값 보존율.
# [왜 중요한가] 모든 결론이 이 채점기의 exact와 구간으로 나온다(docs/experiments.md "공통"). 다른 테스트
# 파일들도 여기의 _item을 가져다 가짜 건을 만든다.

import json
import math

# ScriptedProvider: 정해 둔 답(또는 함수)을 내는 가짜 모델. GPU·Ollama 없이 평가 루프를 끝까지 돌린다.
from call_summary.dataset import Item
from call_summary.evaluate import pick_shots, run_items
from call_summary.prompts import build_messages, record_to_json, system_prompt
from call_summary.providers import ScriptedProvider
from call_summary.schema import parse_reply
from call_summary.scoring import (
    METRICS,
    bootstrap_ci,
    exact_rate,
    paired_bootstrap_diff,
    score_item,
    summarize,
)
from call_summary.specs import make_spec


# 명세 하나로 만든 두 발화짜리 가짜 건. 고객 발화에 모든 슬롯 값을 "유형 값"으로 말하므로 정답 값이 전부
# 전사에 있다(환각 0이 기대값). category=None이면 유형도 명세 생성기가 고른다. 여러 테스트 파일이 이 함수를
# 가져다 쓴다.
def _item(i=0, domain="shop", category: str | None = "주문 취소"):
    spec = make_spec("dev", domain, i, category=category)
    values = " ".join(f"{sv.type} {sv.value}" for sv in spec.slots)
    turns = [
        {"speaker": "상담원", "text": "네 도토리마켓입니다."},
        {"speaker": "고객", "text": f"문의드려요. {values} 입니다."},
    ]
    return Item(
        item_id=spec.spec_id, split="dev", domain=domain, spec=spec, turns=turns, summary="요약입니다."
    )


# parse_reply가 ```json 울타리, 앞뒤 설명이 붙은 답에서도 JSON을 꺼내 통과시키고, JSON이 없으면
# json_ok·schema_ok 모두 거짓인지 본다. 정의에 없는 키(confidence)는 JSON이지만 스키마 실패(extra="forbid"),
# 허용 밖 처리 결과("완료")도 실패다. 요약 문자열 안의 중괄호는 JSON 구간 찾기를 헷갈리게 하지 않아야 한다.
def test_parse_reply_variants():
    it = _item()
    good = record_to_json(it.gold())
    assert parse_reply(good).schema_ok
    assert parse_reply(f"```json\n{good}\n```").schema_ok
    assert parse_reply("설명: " + good + " 끝").schema_ok
    bad = parse_reply("그런 건 모르겠어요")
    assert not bad.json_ok and not bad.schema_ok
    extra = json.loads(good)
    extra["confidence"] = 0.9
    r = parse_reply(json.dumps(extra, ensure_ascii=False))
    assert r.json_ok and not r.schema_ok
    wrong_enum = json.loads(good)
    wrong_enum["resolution"] = "완료"
    assert not parse_reply(json.dumps(wrong_enum, ensure_ascii=False)).schema_ok
    brace_in_string = json.loads(good)
    brace_in_string["summary"] = "고객이 {괄호}를 말함"
    assert parse_reply(json.dumps(brace_in_string, ensure_ascii=False)).schema_ok


# 정답 그대로 답하면 exact이고 fp·fn·환각·목록 밖 라벨이 모두 0이다(채점의 기준점).
def test_perfect_prediction_is_exact():
    it = _item()
    s = score_item(it.item_id, it.domain, it.gold(), parse_reply(record_to_json(it.gold())), it.transcript)
    assert (
        s.exact and s.entity_fp == 0 and s.entity_fn == 0 and s.entity_hallucinated == 0 and s.off_list == 0
    )


# 형식 실패 답은 모든 필드가 오답이고, 정답 값·처리는 모두 fn으로 들어가 F1에 반영된다(규칙: 스키마를 통과하지
# 못한 건은 모든 필드를 오답으로 친다).
def test_format_failure_scores_zero():
    it = _item()
    s = score_item(it.item_id, it.domain, it.gold(), parse_reply("nope"), it.transcript)
    assert not s.schema_ok and not s.exact
    assert s.entity_fn == len(it.gold().entities)
    assert s.action_fn == len(it.gold().actions_taken)


# 금액을 다른 말투("3만 2천 원" ↔ "32,000원")로 적어도 정규화한 값이 같으면 맞은 값이다. 명세가 일부러 말투를
# 바꿔 뽑으므로 이 규칙이 없으면 맞은 답이 틀린 것으로 채점된다.
def test_equivalent_amount_surface_matches():
    it = _item()
    gold = it.gold()
    pred = gold.model_copy(deep=True)
    for e in pred.entities:
        if e.type == "금액":
            # rewrite in the other surface style
            from call_summary.values import normalize

            e.value = f"{int(normalize('amount', e.value)):,}원"
    s = score_item(it.item_id, it.domain, gold, parse_reply(record_to_json(pred)), it.transcript)
    assert s.entity_fp == 0 and s.entity_fn == 0


# 번호 하나를 틀리게 바꾸고 전사에 없는 날짜를 하나 더하면: tp가 하나 줄고, fp 2(틀린 번호 + 더한 날짜), fn
# 1(빠진 정답 번호), 환각 2(둘 다 전사에 없음), exact 아님.
def test_wrong_and_invented_entities():
    it = _item()
    gold = it.gold()
    pred = gold.model_copy(deep=True)
    pred.entities[0].value = "999-9999-99"  # wrong order id, not in the transcript
    pred.entities.append(type(pred.entities[0])(type="날짜", value="11월 30일"))
    s = score_item(it.item_id, it.domain, gold, parse_reply(record_to_json(pred)), it.transcript)
    assert s.entity_tp == len(gold.entities) - 1
    assert s.entity_fp == 2 and s.entity_fn == 1
    assert s.entity_hallucinated == 2
    assert not s.exact


# 업종 목록 밖의 유형("기타"), 처리("마음대로 처리"), 핵심 값 유형("고객명")을 하나씩 쓰면 목록 밖 라벨 수가
# 3이다. 서비스의 재시도 판단(service.off_list_labels)과 같은 기준이다.
def test_off_list_labels_are_counted():
    it = _item()
    pred = it.gold().model_copy(deep=True)
    pred.category = "기타"
    pred.actions_taken.append("마음대로 처리")
    pred.entities.append(type(pred.entities[0])(type="고객명", value="홍길동"))
    s = score_item(it.item_id, it.domain, it.gold(), parse_reply(record_to_json(pred)), it.transcript)
    assert s.off_list == 3
    assert not s.category_ok


# 모두 맞은 실행의 구간은 [1, 1], 모두 틀린 실행과의 짝지은 차이는 +1(하한도 1)이다. 반반 섞인 실행은 점
# 추정이 구간 안쪽에 있다. 예측 값이 하나도 없으면 환각률은 0이 아니라 NaN(정의되지 않음)이고, summarize가
# METRICS의 모든 지표 이름을 돌려주는지도 본다.
def test_bootstrap_and_paired_diff():
    items = [_item(i) for i in range(20)]
    good = [
        score_item(it.item_id, it.domain, it.gold(), parse_reply(record_to_json(it.gold())), it.transcript)
        for it in items
    ]
    bad = [score_item(it.item_id, it.domain, it.gold(), parse_reply("x"), it.transcript) for it in items]
    point, lo, hi = bootstrap_ci(good, exact_rate, n_boot=200)
    assert point == lo == hi == 1.0
    d, dlo, dhi = paired_bootstrap_diff(bad, good, exact_rate, n_boot=200)
    assert d == 1.0 and dlo == 1.0
    mixed = good[:10] + bad[10:]
    p, lo, hi = bootstrap_ci(mixed, exact_rate, n_boot=500)
    assert lo < p < hi
    summary = summarize(bad)
    assert set(summary) == set(METRICS)
    assert math.isnan(summary["hallucination"])  # nothing predicted


# 시스템 프롬프트에 그 업종의 회사 이름, 유형, 핵심 값 유형이 들어가고 다른 업종의 회사 이름은 없는지
# 본다(test-c 카드 업종은 학습에 없어도 프롬프트의 라벨 목록으로만 답한다).
def test_system_prompt_lists_domain_labels():
    p = system_prompt("card")
    assert "구름카드" in p and "카드 분실" in p and "가맹점명" in p
    assert "도토리마켓" not in p


# evaluate.run_items가 건마다 프롬프트를 만들어 provider에 보내고 답을 채점하는지, 2-shot이면 메시지가 시스템
# 1 + 예시 2쌍 + 사용자 1 = 6개이고 build_messages와 같은지 확인한다. 가짜 provider는 사용자 메시지에서 전사를
# 꺼내 그 건의 정답을 돌려준다.
def test_run_items_with_scripted_provider():
    items = [_item(i, category=None) for i in range(6)]
    pool = [_item(100 + i, category=None) for i in range(12)]
    by_transcript = {it.transcript: record_to_json(it.gold()) for it in items}

    def reply(messages):
        text = messages[-1]["content"]
        body = text.split("상담 대화:\n", 1)[1].rsplit("\n\n상담 기록 JSON:", 1)[0]
        return by_transcript[body]

    provider = ScriptedProvider(reply)
    rows, scores = run_items(provider, items, lambda d: pick_shots(pool, d, 2))
    assert all(s.exact for s in scores)
    assert len(rows) == 6
    # 1 system + 2 shots * 2 + 1 user
    assert len(provider.calls[0]) == 6
    assert provider.calls[0] == build_messages("shop", items[0].transcript, pick_shots(pool, "shop", 2))


# 배치로 생성하는 provider(transformers 경로)에서 7건을 3·3·1로 나눠 보내도 결과가 입력 순서 그대로인지
# 확인한다. 순서가 바뀌면 답이 다른 건에 채점된다.
def test_run_items_batched_keeps_order():
    from call_summary.providers import Reply

    items = [_item(i, category=None) for i in range(7)]
    by_transcript = {it.transcript: record_to_json(it.gold()) for it in items}

    class Batcher(ScriptedProvider):
        def generate_batch(self, batch):
            self.calls.append(batch)
            out = []
            for m in batch:
                body = m[-1]["content"].split("상담 대화:\n", 1)[1].rsplit("\n\n상담 기록 JSON:", 1)[0]
                out.append(Reply(text=by_transcript[body], latency_s=0.0))
            return out

    p = Batcher(replies=[])
    rows, scores = run_items(p, items, batch_size=3)
    assert [r["item_id"] for r in rows] == [it.item_id for it in items]
    assert all(s.exact for s in scores)
    assert [len(b) for b in p.calls] == [3, 3, 1]


# p1 프롬프트의 지문은 "4cead4cf6510"으로 고정되어야 한다. p2를 더한 뒤에도 p1 지문이 그대로여야 예전 실행
# 기록과 서비스 /health의 지문이 맞는다. p2는 p1 뒤에 [기록 기준]을 덧붙인 것이고 지문이
# 다르다(docs/experiments.md 3단계 p2 참고치).
def test_prompt_p2_extends_p1_and_keeps_p1_hash():
    from call_summary.prompts import prompt_hash

    assert prompt_hash() == "4cead4cf6510"
    p1, p2 = system_prompt("shop"), system_prompt("shop", "p2")
    assert p2.startswith(p1) and "[기록 기준]" in p2 and "[기록 기준]" not in p1
    assert prompt_hash("p2") != prompt_hash()
    msgs = build_messages("card", "상담원: 안녕하세요", version="p2")
    assert "[기록 기준]" in msgs[0]["content"]


# 음성 조건의 넓은 번호 규칙(4단계 채점): 쉼표와 "에서"로 나뉜 숫자도 한 덩어리로 이어 읽어 번호를 찾는다. 쓴
# 글(spoken=False)에서는 찾지 않고, 숫자가 하나라도 다르면 찾지 못하며, 4자리 미만은 넓은 규칙을 쓰지 않는다.
# "디 3,928,174"처럼 글자와 숫자가 떨어진 주문번호도 숫자로 찾는다.
def test_spoken_ids_match_digit_runs():
    from call_summary.values import digit_runs, occurs_in_transcript

    heard = "운송장 번호는 5,816, 9,659에서 8,958입니다. 10월 19일 오후 4시요."
    assert "581696598958" in digit_runs(heard)
    assert occurs_in_transcript("id", "5816-9659-8958", heard, spoken=True)
    assert not occurs_in_transcript("id", "5816-9659-8958", heard)  # written transcripts stay strict
    assert not occurs_in_transcript("id", "5816-9659-8959", heard, spoken=True)
    assert not occurs_in_transcript("id", "123", "번호 1,2,3", spoken=True)  # too short to count
    assert occurs_in_transcript("id", "D3928174", "주문번호 디 3,928,174요", spoken=True)


# 값 보존율: 값을 모두 말한 전사에서는 모든 정답 값이 남고, 값이 하나도 없는 글에서는 유형마다 모두 거짓이다.
def test_value_survival():
    from call_summary.scoring import value_survival

    it = _item(0)
    gold = it.gold()
    got = value_survival(it.domain, gold, it.transcript)
    assert all(all(v) for v in got.values())
    assert value_survival(it.domain, gold, "아무 값도 없는 대화") == {
        k: [False] * len(v) for k, v in got.items()
    }
