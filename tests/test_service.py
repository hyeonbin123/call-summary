# [테스트 묶음] 5단계 서비스(call_summary.service)를 모델 없이 FastAPI TestClient로 검증한다: 정상 응답,
# 확인이 필요한 번호 표시, 형식·라벨 실패 시 한 번 재시도 후 502, 재시도가 첫 답의 되풀이가 아님, 입력
# 검사(422)의 정확한 범위, 보안 스캔이 처리기까지 닿는지(OpenAPI 설명, ISO-8859-1 한글, HAR 씨앗), 413·503,
# 보안 헤더, /stats, 요청별 로그 한 줄(개인정보 없음), 스키마 강제는 켤 때만.
# [왜 중요한가] 서비스는 학습한 모델의 답을 그대로 내보내지 않고 검증·재시도·표시를 거친다(README 5단계). 이
# 동작과 로그의 개인정보 규칙, 보안 스캔이 처리기 안쪽을 시험할 수 있는 조건이 코드가 바뀌어도 유지되는지
# 지킨다.

import json
import logging
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from call_summary.dataset import load_items
from call_summary.domains import DOMAINS
from call_summary.prompts import record_to_json
from call_summary.providers import OllamaProvider, Reply, ScriptedProvider
from call_summary.service import EXAMPLE_REQUEST, MAX_CHARS, create_app, create_offline_app
from call_summary.specs import make_spec

# 번호를 하나도 말하지 않는 짧은 대화(쇼핑몰).
TURNS = [
    {"speaker": "상담원", "text": "도토리마켓입니다."},
    {"speaker": "고객", "text": "주문 취소하려고요."},
]


# 검증을 통과하는 쇼핑몰 기록 JSON(명세 dev-shop-00003의 정답 + 요약). 이 기록의 주문번호는 TURNS에 없다.
def _good():
    return record_to_json(make_spec("dev", "shop", 3, category="주문 취소").gold("고객이 취소를 요청함."))


# 정상 경로: 첫 답이 통과하면 200, 기록 그대로, 시도 1번.
def test_summarize_ok():
    p = ScriptedProvider([_good()])
    c = TestClient(create_app(p))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["record"]["category"] == "주문 취소" and body["attempts"] == 1


# 전사에 없는 주문번호는 needs_confirmation에 not_in_transcript로 표시되고, 기록 자체는 모델이 쓴
# 그대로다(표시만 하고 고치지 않음). 같은 번호를 띄어 말한 전사("D 203 1905")에서는 넓은 번호 규칙으로 찾아
# 표시하지 않는다. 로그에는 표시한 개수(1, 0)만 남고 번호 값은 남지 않는다(번호는 개인정보).
def test_identifiers_that_need_confirmation_are_listed(caplog):
    good = _good()  # its order number D2031905 is never said in TURNS
    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(ScriptedProvider([good, good])))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["needs_confirmation"] == [
        {"type": "주문번호", "value": "D2031905", "reasons": ["not_in_transcript"]}
    ]
    assert body["record"] == json.loads(good)  # returned as the model wrote it
    said = TURNS + [{"speaker": "고객", "text": "주문번호는 D 203 1905요."}]
    assert c.post("/summarize", json={"domain": "shop", "turns": said}).json()["needs_confirmation"] == []
    assert [x["needs_confirmation"] for x in _log_lines(caplog)] == [1, 0]
    # The log keeps the count only: the flagged value is personal data.
    assert "D2031905" not in " ".join(rec.getMessage() for rec in caplog.records)


# 오프라인 모드의 고정 답에는 업종마다 번호 하나가 있어, 스캔 요청도 번호 검사 코드를 지난다. 전사에 그 번호가
# 없으면 표시되고, 말하면 표시되지 않는다. OpenAPI 응답 스키마에도 needs_confirmation이 있어야 한다.
def test_offline_app_runs_the_identifier_checks():
    c = TestClient(create_offline_app())
    for domain in DOMAINS:
        body = c.post("/summarize", json={"domain": domain, "turns": TURNS}).json()
        (entity,) = body["record"]["entities"]
        assert body["needs_confirmation"] == [{**entity, "reasons": ["not_in_transcript"]}]
        said = TURNS + [{"speaker": "고객", "text": f"번호는 {entity['value']}입니다."}]
        assert c.post("/summarize", json={"domain": domain, "turns": said}).json()["needs_confirmation"] == []
    response = _openapi()["components"]["schemas"]["SummarizeResponse"]
    assert "needs_confirmation" in response["properties"]


# 첫 답이 JSON이 아니고 두 번째 답이 목록 밖 유형("기타")이면 모델을 정확히 두 번 부른 뒤 502를 주고, 응답
# 본문에 마지막 문제(category)가 들어간다. 반쯤 맞는 기록을 내보내지 않는다.
def test_summarize_retries_once_then_502():
    bad_label = json.loads(_good())
    bad_label["category"] = "기타"
    p = ScriptedProvider(["not json", json.dumps(bad_label, ensure_ascii=False)])
    c = TestClient(create_app(p))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 502
    assert "category" in json.dumps(r.json(), ensure_ascii=False)
    assert len(p.calls) == 2


# 첫 답이 깨졌어도 두 번째 답이 통과하면 200이고 attempts가 2다.
def test_retry_recovers():
    p = ScriptedProvider(["{", _good()])
    r = TestClient(create_app(p)).post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 200 and r.json()["attempts"] == 2


# 진짜 OllamaProvider로 보내는 요청 본문을 가로채 본다. 첫 요청은 temperature 0, 재시도는 temperature > 0이고
# seed가 다르며, 메시지와 컨텍스트 길이는 같다(컨텍스트가 바뀌면 Ollama가 모델을 다시 올린다). 탐욕 디코딩의
# 재시도는 같은 답을 되풀이할 뿐이라 샘플링해야 한다(커밋 1344e49).
def test_retry_is_not_a_replay(monkeypatch):
    # A greedy model answers the same request the same way, so the second try must sample.
    off_list = json.loads(_good())
    off_list["category"] = "기타"
    bodies = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": json.dumps(off_list, ensure_ascii=False)}}

    def fake_post(url, **kw):
        bodies.append(kw["json"])
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    c = TestClient(create_app(OllamaProvider(model="m", use_schema=True, num_ctx=4096)))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 502 and len(bodies) == 2
    first, second = (b["options"] for b in bodies)
    assert first["temperature"] == 0 and second["temperature"] > 0
    assert second["seed"] != first["seed"]
    assert second["num_ctx"] == first["num_ctx"]  # a different context would reload the model
    assert bodies[0]["messages"] == bodies[1]["messages"]


# 목록 밖 업종, 빈 발화 목록, 허용되지 않은 발화자는 422다. /health와 /domains도 응답한다.
def test_input_validation():
    c = TestClient(create_app(ScriptedProvider([])))
    assert c.post("/summarize", json={"domain": "bank", "turns": TURNS}).status_code == 422
    assert c.post("/summarize", json={"domain": "shop", "turns": []}).status_code == 422
    bad = [{"speaker": "로봇", "text": "안녕"}]
    assert c.post("/summarize", json={"domain": "shop", "turns": bad}).status_code == 422
    assert c.get("/health").json()["status"] == "ok"
    assert "card" in c.get("/domains").json()


# OpenAPI에 enum을 적은 뒤에도 받는 값이 느슨해지지 않았는지 확인한다: 공백·줄바꿈이 붙거나 일부만 같거나
# 타입이 다른 발화자, 대소문자·공백이 다르거나 너무 긴 업종, 2,001자 발화, 200개를 넘는 발화, turns가 없는
# 본문은 모두 422다(docs/experiments.md "보안 스캔 기록" 2026-10-03: 받는 값은 바뀌지 않는다).
def test_input_validation_is_exact():
    # The OpenAPI enums must not loosen what the service accepts.
    c = TestClient(create_offline_app())
    for speaker in ("상담원 ", "고객\n", "상담", "agent", 1, None):
        body = {"domain": "shop", "turns": [{"speaker": speaker, "text": "안녕하세요"}]}
        assert c.post("/summarize", json=body).status_code == 422, speaker
    for domain in ("SHOP", "shop ", "x" * 33, "", 1):
        assert c.post("/summarize", json={"domain": domain, "turns": TURNS}).status_code == 422, domain
    too_long_turn = [{"speaker": "고객", "text": "가" * 2001}]
    assert c.post("/summarize", json={"domain": "shop", "turns": too_long_turn}).status_code == 422
    assert c.post("/summarize", json={"domain": "shop", "turns": TURNS * 101}).status_code == 422
    assert c.post("/summarize", json={"domain": "shop"}).status_code == 422


# 오프라인 앱이 공개하는 OpenAPI 문서.
def _openapi() -> dict:
    return TestClient(create_offline_app()).get("/openapi.json").json()


# 보안 스캐너처럼 OpenAPI 스키마만 보고 요청 본문을 만든다: $ref를 따라가고, enum이면 첫 값, 문자열이면 자리
# 표시 값("John Doe"), 배열은 최소 개수만큼, 객체는 필수 키만.
def _from_schema(schema: dict, spec: dict):
    """A request built from the schema alone, as an API scanner builds one: the first enum value, else a
    placeholder; as many array items as the minimum asks for (at least one)."""
    if "$ref" in schema:
        return _from_schema(spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]], spec)
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {k: _from_schema(v, spec) for k, v in schema["properties"].items() if k in schema["required"]}
    if kind == "array":
        return [_from_schema(schema["items"], spec) for _ in range(max(1, schema.get("minItems", 1)))]
    assert kind == "string", schema
    return "John Doe"


# OpenAPI가 발화자(상담원, 고객)와 업종 목록을 enum으로 알린다. 그래야 스캐너가 입력 검사를 통과하는 값을 고를
# 수 있다(커밋 a13fcb6).
def test_openapi_lists_the_accepted_speakers_and_domains():
    schemas = _openapi()["components"]["schemas"]
    assert schemas["Turn"]["properties"]["speaker"]["enum"] == ["상담원", "고객"]
    assert schemas["SummarizeRequest"]["properties"]["domain"]["enum"] == list(DOMAINS)


# OpenAPI만 보고 만든 요청이 422에서 멈추지 않고 처리기에 닿아 200과 로그 줄(ok)을 남기는지 확인한다. 예전
# 스캔은 모든 /summarize 요청이 입력 검사에서 걸려 처리기 안쪽을 시험하지 못했다(docs/experiments.md "보안
# 스캔 기록").
def test_request_built_from_openapi_reaches_the_handler(caplog):
    # HawkScan builds /summarize bodies from the OpenAPI description; with a free-form speaker and
    # domain every one of them stopped at validation (422), so the handler was never scanned.
    spec = _openapi()
    op = spec["paths"]["/summarize"]["post"]
    body = _from_schema(op["requestBody"]["content"]["application/json"]["schema"], spec)
    caplog.set_level(logging.INFO, logger="call_summary.service")
    r = TestClient(create_offline_app()).post("/summarize", json=body)
    assert r.status_code == 200, r.text
    assert [x["outcome"] for x in _log_lines(caplog)] == ["ok"]


# 2026-10-03 18:38 스캔을 재현한다: 맞는 본문이라도 ISO-8859-1로 보내면 한글 발화자가 "???"가 되어 speaker
# 검사(422)에서 걸린다. 스캔이 처리기에 닿지 못한 원인이 서비스가 아니라 본문 인코딩이었음을 보이는 테스트다.
def test_korean_body_sent_as_iso_8859_1_stops_at_validation():
    # The 2026-10-03 18:38 scan: a valid body, sent in ISO-8859-1, reached the service with speaker "???".
    body = json.dumps(EXAMPLE_REQUEST, ensure_ascii=False).encode("iso-8859-1", errors="replace")
    r = TestClient(create_offline_app()).post(
        "/summarize", content=body, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "speaker"


# stackhawk.yml이 HAR 씨앗(hawk/summarize.har)을 쓰고, 그 요청이 POST /summarize이며 본문이 ASCII(한글을 JSON
# \u 이스케이프로 적음)이고 EXAMPLE_REQUEST와 같은지 확인한다. ISO-8859-1로 보내도 깨지지 않아 처리기에
# 닿는다(커밋 b4e69ac).
def test_hawkscan_seed_request_is_ascii_and_reaches_the_handler(caplog):
    # HawkScan sends the bodies it builds from OpenAPI in ISO-8859-1, so 상담원 arrives as "???" (422 on
    # 2026-10-03). The HAR seed in stackhawk.yml keeps its body ASCII with JSON \u escapes.
    root = Path(__file__).resolve().parents[1]
    assert "hawk/summarize.har" in (root / "stackhawk.yml").read_text(encoding="utf-8")
    har = json.loads((root / "hawk" / "summarize.har").read_text(encoding="utf-8"))
    (entry,) = har["log"]["entries"]
    req = entry["request"]
    assert (req["method"], req["url"]) == ("POST", "http://127.0.0.1:8072/summarize")
    body = req["postData"]["text"]
    assert body.isascii() and json.loads(body) == EXAMPLE_REQUEST
    headers = {h["name"].lower(): h["value"] for h in req["headers"]}
    caplog.set_level(logging.INFO, logger="call_summary.service")
    r = TestClient(create_offline_app()).post(
        "/summarize", content=body.encode("iso-8859-1"), headers={"Content-Type": headers["content-type"]}
    )
    assert r.status_code == 200, r.text
    assert [x["outcome"] for x in _log_lines(caplog)] == ["ok"]


# OpenAPI에 붙인 요청 예시가 실제로 200을 받는 요청이어야 한다(예시가 틀리면 스캐너와 사용자를 잘못 안내한다).
def test_openapi_request_examples_are_valid():
    examples = _openapi()["components"]["schemas"]["SummarizeRequest"]["examples"]
    assert examples
    c = TestClient(create_offline_app())
    for ex in examples:
        assert c.post("/summarize", json=ex).status_code == 200, ex


# 조립한 전사가 MAX_CHARS(4,000자)를 넘으면 모델을 부르지 않고 413과 상한 값을 돌려준다(README 서비스 문단,
# 커밋 fdc1b9c).
def test_transcript_beyond_model_context_is_413():
    # ~4,500 chars is ~3,300 prompt tokens: with the reply budget it no longer fits the 4,096 context.
    p = ScriptedProvider([_good()])
    long_turns = [{"speaker": "상담원" if i % 2 == 0 else "고객", "text": "가" * 1500} for i in range(3)]
    r = TestClient(create_app(p)).post("/summarize", json={"domain": "shop", "turns": long_turns})
    assert r.status_code == 413 and p.calls == []
    assert str(MAX_CHARS) in r.json()["detail"]


# 커밋된 모든 데이터 분할의 가장 긴 전사가 서비스 상한 안에 있는지 확인한다. 상한 때문에 측정 데이터가 413으로
# 잘리는 일이 없음을 보장한다(docs/experiments.md "보안 스캔 기록" 2026-09-29 항목의 바로잡음: 가장 긴 전사
# 1,485자).
def test_dataset_transcripts_fit_service_limit():
    files = sorted((Path(__file__).resolve().parents[1] / "datasets").glob("*.jsonl"))
    assert files
    for f in files:
        assert max(len(it.transcript) for it in load_items(f)) <= MAX_CHARS, f.name


# 모델 서버가 연결을 거부하면 503을 주고, 응답에 내부 오류 문구("refused")를 내보내지 않으며, /stats에
# model_unavailable로 센다.
def test_model_server_down_is_503_and_counted():
    class Down:
        name = "down"

        def generate(self, messages, json_schema=None):
            raise ConnectionError("refused")

    c = TestClient(create_app(Down()))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 503 and "refused" not in r.text
    assert c.get("/stats").json()["requests"]["model_unavailable"] == 1


# 응답에 보안 헤더(nosniff, no-store)가 붙고, 기본 앱은 문서 페이지와 OpenAPI를 숨긴다(404).
def test_security_headers_and_hidden_docs():
    c = TestClient(create_app(ScriptedProvider([_good()])))
    r = c.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "no-store"
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


# /stats가 결과별 수(재시도로 통과 1, 바로 통과 1)와 지연 표본 수를 맞게 세는지 본다.
def test_stats_after_requests():
    c = TestClient(create_app(ScriptedProvider(["{", _good(), _good()])))
    for _ in range(2):
        assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 200
    st = c.get("/stats").json()
    assert st["requests"]["retried_ok"] == 1 and st["requests"]["ok"] == 1
    assert st["latency_s"]["n"] == 2


# 오프라인 모드는 네 업종 모두 200을 주고 모델 이름이 offline:canned이며 OpenAPI를 공개한다(스캔용).
def test_offline_app_answers_every_domain():
    c = TestClient(create_offline_app())
    for domain in ("shop", "telecom", "parcel", "card"):
        r = c.post("/summarize", json={"domain": domain, "turns": TURNS})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "offline:canned"
    assert c.get("/openapi.json").status_code == 200


# 음성 인식이 빈 글을 낸 발화도 받아들인다. test-d에서 빈 발화 하나가 입력 검사에 걸려 422가 난 것을
# 고쳤다(docs/experiments.md 5단계 발견 1).
def test_empty_recognised_turns_are_accepted():
    c = TestClient(create_app(ScriptedProvider([_good()])))
    turns = TURNS + [{"speaker": "고객", "text": ""}]
    assert c.post("/summarize", json={"domain": "shop", "turns": turns}).status_code == 200


# Ollama처럼 토큰 수(프롬프트 900, 생성 200)를 붙여 답하는 가짜 provider. 로그의 토큰 합을 확인하는 데 쓴다.
class TokenProvider:
    """Replies in order, with token counts as Ollama reports them."""

    name = "tokens"

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, json_schema=None):
        return Reply(text=self.replies.pop(0), latency_s=0.0, prompt_tokens=900, completion_tokens=200)


# caplog에서 서비스 로거의 JSON 줄만 꺼내 dict로 읽는다.
def _log_lines(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "call_summary.service"]


# 처리기에 닿은 요청마다 JSON 한 줄: 요청 id(응답과 같음), 결과, 상태, 시도, 토큰 수, 업종, 발화 수, 지연.
# 대화 내용과 기록 요약 글은 로그에 없어야 한다(개인정보, README 서비스 문단).
def test_summarize_logs_one_line_with_tokens(caplog):
    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(TokenProvider([_good()])))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 200
    (line,) = _log_lines(caplog)
    assert line["event"] == "summarize" and line["outcome"] == "ok" and line["status"] == 200
    assert line["request_id"] == r.json()["request_id"]
    assert (line["attempts"], line["prompt_tokens"], line["completion_tokens"]) == (1, 900, 200)
    assert line["domain"] == "shop" and line["turns"] == 2 and line["latency_s"] >= 0
    # No call content in the log: transcripts and records hold personal data.
    raw = " ".join(rec.getMessage() for rec in caplog.records)
    assert "주문 취소하려고요" not in raw and "고객이 취소를 요청함" not in raw


# 두 번 모두 실패하면 rejected·502·시도 2로 남고, 토큰 수는 두 시도를 더한 값(1800, 400)이다.
def test_log_sums_tokens_over_attempts_and_records_rejection(caplog):
    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(TokenProvider(["not json", "{"])))
    assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 502
    (line,) = _log_lines(caplog)
    assert (line["outcome"], line["status"], line["attempts"]) == ("rejected", 502, 2)
    assert (line["prompt_tokens"], line["completion_tokens"]) == (1800, 400)


# 토큰 수를 주지 않는 provider면 로그의 토큰 칸은 0이 아니라 null이다(모름과 0을 구분).
def test_log_without_model_counts(caplog):
    # Providers that report no token counts log null, not zero.
    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(ScriptedProvider(["{", _good()])))
    assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 200
    (line,) = _log_lines(caplog)
    assert (line["outcome"], line["attempts"]) == ("retried_ok", 2)
    assert line["prompt_tokens"] is None and line["completion_tokens"] is None


# 모델에 닿지 않고 끝난 요청도 로그 줄을 남긴다: 모델 서버 오류(503, 시도 1), 목록 밖 업종(422, 시도 0), 너무
# 긴 전사(413, 시도 0). 이때 확인 표시 수는 null이고, 예외 문구는 로그에도 남지 않는다.
def test_log_lines_for_requests_that_never_reach_the_model(caplog):
    class Down:
        name = "down"

        def generate(self, messages, json_schema=None):
            raise ConnectionError("refused")

    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(Down()))
    long_turns = [{"speaker": "고객", "text": "가" * 1500} for _ in range(3)]
    assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 503
    assert c.post("/summarize", json={"domain": "bank", "turns": TURNS}).status_code == 422
    assert c.post("/summarize", json={"domain": "shop", "turns": long_turns}).status_code == 413
    lines = _log_lines(caplog)
    assert [(x["outcome"], x["status"], x["attempts"]) for x in lines] == [
        ("model_unavailable", 503, 1),
        ("unknown_domain", 422, 0),
        ("too_long", 413, 0),
    ]
    assert all(x["needs_confirmation"] is None for x in lines)
    assert "refused" not in json.dumps(lines)


# 스키마 강제 디코딩은 CALL_SUMMARY_SCHEMA=1일 때만 켜진다. 키 순서를 바꿔 학습한 모델이 actions_taken을
# 빠뜨렸기 때문이다(docs/experiments.md 5단계 발견 1, 커밋 34086df). /health의 모델 이름 끝 ":schema"로
# 확인한다.
def test_schema_decoding_is_opt_in(monkeypatch):
    # /health reports the provider name, which ends in ":schema" when constrained decoding is on.
    monkeypatch.delenv("CALL_SUMMARY_SCHEMA", raising=False)
    assert not TestClient(create_app()).get("/health").json()["model"].endswith(":schema")
    monkeypatch.setenv("CALL_SUMMARY_SCHEMA", "1")
    assert TestClient(create_app()).get("/health").json()["model"].endswith(":schema")
