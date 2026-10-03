"""HTTP service: POST /summarize turns a call transcript into an after-call record.

The model reply must validate as an AfterCallRecord and use the domain's label lists; otherwise the
service retries once, sampled (a greedy retry would replay the rejected reply), and then answers 502 with
the reason, never a half-valid record. An unreachable model server gives 503.

Each /summarize call that reaches the handler logs one JSON line (logger `call_summary.service`, INFO):
request id, domain, turn and character counts, outcome, HTTP status, attempts, latency and the prompt and
completion tokens summed over attempts (null when the model server reports none). Never the transcript or
the record: both hold personal data.

Run with a model:   CALL_SUMMARY_MODEL=<ollama model> uvicorn call_summary.service:create_app --factory
Run without one:    uvicorn call_summary.service:create_offline_app --factory   (canned replies, for scans)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import statistics
import sys
import threading
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .domains import DOMAINS
from .prompts import build_messages, format_transcript, prompt_hash, record_to_json
from .providers import OllamaProvider, Provider, Reply
from .schema import AfterCallRecord, FollowUp, parse_reply, record_json_schema

MAX_TURNS = 200
NUM_CTX = 4096  # Ollama context the service asks for
# Prompt tokens ~= 571 + 0.605 * transcript chars (dev runs, Qwen3 and qwen2.5 tokenizers). 4,000 chars is
# ~3,000 prompt tokens, leaving ~1,000 for the reply; longer input would be truncated by the model server.
# Dataset transcripts are at most 1,485 chars.
MAX_CHARS = 4_000
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}
log = logging.getLogger("call_summary.service")


def _ensure_log_handler() -> None:
    """uvicorn configures only its own loggers, so without a handler these INFO lines would be dropped.
    Left alone when the application (or pytest) already handles logging at the root."""
    if log.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def _token_sum(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


EXAMPLE_REQUEST = {
    "domain": "shop",
    "turns": [
        {"speaker": "상담원", "text": "도토리마켓입니다. 무엇을 도와드릴까요?"},
        {"speaker": "고객", "text": "어제 한 주문을 취소하려고요."},
    ],
}


class Turn(BaseModel):
    # A Literal, not a regex: the OpenAPI enum tells clients and scanners which values pass.
    speaker: Literal["상담원", "고객"]
    # Empty text is allowed: a speech recogniser returns nothing for some turns (test-d has such turns).
    text: str = Field(max_length=2000)


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    # The enum is only documented here; the handler checks it, so an unknown domain keeps its own 422
    # ("unknown domain") and its log line.
    domain: str = Field(max_length=32, json_schema_extra={"enum": list(DOMAINS)})
    turns: list[Turn] = Field(min_length=1, max_length=MAX_TURNS)


class SummarizeResponse(BaseModel):
    request_id: str
    record: AfterCallRecord
    model: str
    attempts: int
    latency_s: float


def off_list_labels(domain_key: str, record: AfterCallRecord) -> list[str]:
    d = DOMAINS[domain_key]
    bad = []
    if record.category not in d.categories:
        bad.append(f"category {record.category!r}")
    bad += [f"action {a!r}" for a in record.actions_taken if a not in d.actions]
    bad += [f"follow-up {c!r}" for c in record.follow_up.codes if c not in d.follow_up_codes]
    types = {et.label for et in d.entity_types}
    bad += [f"entity type {e.type!r}" for e in record.entities if e.type not in types]
    return bad


def _retry_provider(p: Provider) -> Provider:
    """The second try must not replay the rejected greedy decode: same model and context, sampled."""
    if isinstance(p, OllamaProvider):
        return dataclasses.replace(p, temperature=0.3, seed=p.seed + 1)  # keep num_ctx/num_gpu: no reload
    return p


class Stats:
    """Counts and latencies of this process's /summarize calls (reset on restart)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self.counts = {"ok": 0, "retried_ok": 0, "rejected": 0, "model_unavailable": 0}

    def add(self, outcome: str, latency: float | None = None) -> None:
        with self._lock:
            self.counts[outcome] += 1
            if latency is not None:
                self.latencies.append(latency)

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies)
        out: dict = {"requests": dict(self.counts), "latency_s": None}
        if lat:
            out["latency_s"] = {
                "n": len(lat),
                "p50": round(statistics.median(lat), 3),
                "p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 3),
                "max": round(lat[-1], 3),
            }
        return out


def create_app(provider: Provider | None = None, expose_openapi: bool | None = None) -> FastAPI:
    if expose_openapi is None:
        expose_openapi = os.environ.get("CALL_SUMMARY_EXPOSE_OPENAPI") == "1"
    app = FastAPI(
        title="call-summary",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if expose_openapi else None,
    )
    stats = Stats()
    _ensure_log_handler()
    base: Provider = provider or OllamaProvider(
        model=os.environ.get("CALL_SUMMARY_MODEL", "qwen2.5:7b-instruct"),
        base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        # Off by default. Ollama's schema grammar writes required keys first in alphabetical order, and the
        # fine-tuned model, trained on one key order, then drops `actions_taken` (stage 5 in experiments.md).
        use_schema=os.environ.get("CALL_SUMMARY_SCHEMA") == "1",
        num_ctx=NUM_CTX,
        num_gpu=int(os.environ["CALL_SUMMARY_NUM_GPU"]) if os.environ.get("CALL_SUMMARY_NUM_GPU") else None,
    )
    # Built once: requests run concurrently in a threadpool, so the shared provider is never mutated.
    state: dict = {"provider": base, "retry_provider": _retry_provider(base)}

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": state["provider"].name, "prompt_hash": prompt_hash()}

    @app.get("/stats")
    def get_stats() -> dict:
        return stats.snapshot()

    @app.get("/domains")
    def domains() -> dict:
        return {
            k: {
                "label": d.label,
                "categories": list(d.categories),
                "actions": list(d.actions),
                "follow_up_codes": list(d.follow_up_codes),
                "entity_types": [et.label for et in d.entity_types],
            }
            for k, d in DOMAINS.items()
        }

    @app.post("/summarize", response_model=SummarizeResponse)
    def summarize(req: SummarizeRequest) -> SummarizeResponse:
        p: Provider = state["provider"]
        request_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        line: dict = {
            "event": "summarize",
            "request_id": request_id,
            "domain": req.domain if req.domain in DOMAINS else None,
            "turns": len(req.turns),
            "chars": None,
            "outcome": None,
            "status": None,
            "attempts": 0,
            "latency_s": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": p.name,
        }
        prompt_tokens: list[int | None] = []
        completion_tokens: list[int | None] = []

        def finish(outcome: str, status: int, latency: float | None = None) -> None:
            line.update(
                outcome=outcome,
                status=status,
                latency_s=latency if latency is not None else round(time.perf_counter() - t0, 3),
                prompt_tokens=_token_sum(prompt_tokens),
                completion_tokens=_token_sum(completion_tokens),
            )
            log.info(json.dumps(line, ensure_ascii=False))

        if req.domain not in DOMAINS:
            finish("unknown_domain", 422)
            raise HTTPException(422, "unknown domain")
        transcript = format_transcript([t.model_dump() for t in req.turns])
        line["chars"] = len(transcript)
        if len(transcript) > MAX_CHARS:
            finish("too_long", 413)
            raise HTTPException(413, f"transcript too long (max {MAX_CHARS} characters)")
        messages = build_messages(req.domain, transcript)
        problems: list[str] = []
        for attempt in (1, 2):
            gen: Provider = p if attempt == 1 else state["retry_provider"]
            line["attempts"] = attempt
            try:
                reply = gen.generate(messages, json_schema=record_json_schema())
            except Exception as exc:  # noqa: BLE001 - any model-server failure is a 503, not a traceback
                stats.add("model_unavailable")
                finish("model_unavailable", 503)
                raise HTTPException(503, "model server unavailable") from exc
            prompt_tokens.append(reply.prompt_tokens)
            completion_tokens.append(reply.completion_tokens)
            parsed = parse_reply(reply.text)
            if parsed.record is None:
                problems = [parsed.error or "invalid"]
                continue
            problems = off_list_labels(req.domain, parsed.record)
            if not problems:
                latency = round(time.perf_counter() - t0, 3)
                outcome = "ok" if attempt == 1 else "retried_ok"
                stats.add(outcome, latency)
                finish(outcome, 200, latency)
                return SummarizeResponse(
                    request_id=request_id,
                    record=parsed.record,
                    model=p.name,
                    attempts=attempt,
                    latency_s=latency,
                )
        stats.add("rejected")
        finish("rejected", 502)
        raise HTTPException(502, {"error": "model output rejected", "problems": problems})

    return app


class CannedProvider:
    """No model: a valid record for whichever domain the system prompt names. For scans and demos."""

    name = "offline:canned"

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        system = messages[0]["content"]
        domain = next((d for d in DOMAINS.values() if d.company in system), DOMAINS["shop"])
        record = AfterCallRecord(
            category=domain.categories[0],
            resolution="해결",
            entities=[],
            actions_taken=[],
            follow_up=FollowUp(required=False, codes=[]),
            summary="오프라인 모드의 고정 응답입니다.",
        )
        return Reply(text=record_to_json(record), latency_s=0.0)


def create_offline_app() -> FastAPI:
    return create_app(provider=CannedProvider(), expose_openapi=True)
