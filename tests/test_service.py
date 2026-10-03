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

TURNS = [
    {"speaker": "상담원", "text": "도토리마켓입니다."},
    {"speaker": "고객", "text": "주문 취소하려고요."},
]


def _good():
    return record_to_json(make_spec("dev", "shop", 3, category="주문 취소").gold("고객이 취소를 요청함."))


def test_summarize_ok():
    p = ScriptedProvider([_good()])
    c = TestClient(create_app(p))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["record"]["category"] == "주문 취소" and body["attempts"] == 1


def test_summarize_retries_once_then_502():
    bad_label = json.loads(_good())
    bad_label["category"] = "기타"
    p = ScriptedProvider(["not json", json.dumps(bad_label, ensure_ascii=False)])
    c = TestClient(create_app(p))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 502
    assert "category" in json.dumps(r.json(), ensure_ascii=False)
    assert len(p.calls) == 2


def test_retry_recovers():
    p = ScriptedProvider(["{", _good()])
    r = TestClient(create_app(p)).post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 200 and r.json()["attempts"] == 2


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


def test_input_validation():
    c = TestClient(create_app(ScriptedProvider([])))
    assert c.post("/summarize", json={"domain": "bank", "turns": TURNS}).status_code == 422
    assert c.post("/summarize", json={"domain": "shop", "turns": []}).status_code == 422
    bad = [{"speaker": "로봇", "text": "안녕"}]
    assert c.post("/summarize", json={"domain": "shop", "turns": bad}).status_code == 422
    assert c.get("/health").json()["status"] == "ok"
    assert "card" in c.get("/domains").json()


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


def _openapi() -> dict:
    return TestClient(create_offline_app()).get("/openapi.json").json()


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


def test_openapi_lists_the_accepted_speakers_and_domains():
    schemas = _openapi()["components"]["schemas"]
    assert schemas["Turn"]["properties"]["speaker"]["enum"] == ["상담원", "고객"]
    assert schemas["SummarizeRequest"]["properties"]["domain"]["enum"] == list(DOMAINS)


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


def test_korean_body_sent_as_iso_8859_1_stops_at_validation():
    # The 2026-10-03 18:38 scan: a valid body, sent in ISO-8859-1, reached the service with speaker "???".
    body = json.dumps(EXAMPLE_REQUEST, ensure_ascii=False).encode("iso-8859-1", errors="replace")
    r = TestClient(create_offline_app()).post(
        "/summarize", content=body, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "speaker"


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


def test_openapi_request_examples_are_valid():
    examples = _openapi()["components"]["schemas"]["SummarizeRequest"]["examples"]
    assert examples
    c = TestClient(create_offline_app())
    for ex in examples:
        assert c.post("/summarize", json=ex).status_code == 200, ex


def test_transcript_beyond_model_context_is_413():
    # ~4,500 chars is ~3,300 prompt tokens: with the reply budget it no longer fits the 4,096 context.
    p = ScriptedProvider([_good()])
    long_turns = [{"speaker": "상담원" if i % 2 == 0 else "고객", "text": "가" * 1500} for i in range(3)]
    r = TestClient(create_app(p)).post("/summarize", json={"domain": "shop", "turns": long_turns})
    assert r.status_code == 413 and p.calls == []
    assert str(MAX_CHARS) in r.json()["detail"]


def test_dataset_transcripts_fit_service_limit():
    files = sorted((Path(__file__).resolve().parents[1] / "datasets").glob("*.jsonl"))
    assert files
    for f in files:
        assert max(len(it.transcript) for it in load_items(f)) <= MAX_CHARS, f.name


def test_model_server_down_is_503_and_counted():
    class Down:
        name = "down"

        def generate(self, messages, json_schema=None):
            raise ConnectionError("refused")

    c = TestClient(create_app(Down()))
    r = c.post("/summarize", json={"domain": "shop", "turns": TURNS})
    assert r.status_code == 503 and "refused" not in r.text
    assert c.get("/stats").json()["requests"]["model_unavailable"] == 1


def test_security_headers_and_hidden_docs():
    c = TestClient(create_app(ScriptedProvider([_good()])))
    r = c.get("/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "no-store"
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_stats_after_requests():
    c = TestClient(create_app(ScriptedProvider(["{", _good(), _good()])))
    for _ in range(2):
        assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 200
    st = c.get("/stats").json()
    assert st["requests"]["retried_ok"] == 1 and st["requests"]["ok"] == 1
    assert st["latency_s"]["n"] == 2


def test_offline_app_answers_every_domain():
    c = TestClient(create_offline_app())
    for domain in ("shop", "telecom", "parcel", "card"):
        r = c.post("/summarize", json={"domain": domain, "turns": TURNS})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "offline:canned"
    assert c.get("/openapi.json").status_code == 200


def test_empty_recognised_turns_are_accepted():
    c = TestClient(create_app(ScriptedProvider([_good()])))
    turns = TURNS + [{"speaker": "고객", "text": ""}]
    assert c.post("/summarize", json={"domain": "shop", "turns": turns}).status_code == 200


class TokenProvider:
    """Replies in order, with token counts as Ollama reports them."""

    name = "tokens"

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, json_schema=None):
        return Reply(text=self.replies.pop(0), latency_s=0.0, prompt_tokens=900, completion_tokens=200)


def _log_lines(caplog):
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == "call_summary.service"]


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


def test_log_sums_tokens_over_attempts_and_records_rejection(caplog):
    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(TokenProvider(["not json", "{"])))
    assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 502
    (line,) = _log_lines(caplog)
    assert (line["outcome"], line["status"], line["attempts"]) == ("rejected", 502, 2)
    assert (line["prompt_tokens"], line["completion_tokens"]) == (1800, 400)


def test_log_without_model_counts(caplog):
    # Providers that report no token counts log null, not zero.
    caplog.set_level(logging.INFO, logger="call_summary.service")
    c = TestClient(create_app(ScriptedProvider(["{", _good()])))
    assert c.post("/summarize", json={"domain": "shop", "turns": TURNS}).status_code == 200
    (line,) = _log_lines(caplog)
    assert (line["outcome"], line["attempts"]) == ("retried_ok", 2)
    assert line["prompt_tokens"] is None and line["completion_tokens"] is None


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
    assert "refused" not in json.dumps(lines)


def test_schema_decoding_is_opt_in(monkeypatch):
    # /health reports the provider name, which ends in ":schema" when constrained decoding is on.
    monkeypatch.delenv("CALL_SUMMARY_SCHEMA", raising=False)
    assert not TestClient(create_app()).get("/health").json()["model"].endswith(":schema")
    monkeypatch.setenv("CALL_SUMMARY_SCHEMA", "1")
    assert TestClient(create_app()).get("/health").json()["model"].endswith(":schema")
