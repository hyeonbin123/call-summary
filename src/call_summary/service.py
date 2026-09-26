"""HTTP service: POST /summarize turns a call transcript into an after-call record.

The model reply must validate as an AfterCallRecord and use the domain's label lists; otherwise the
service retries once and then answers 502 with the reason, never a half-valid record.
"""

from __future__ import annotations

import os
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .domains import DOMAINS
from .prompts import build_messages, format_transcript, prompt_hash
from .providers import OllamaProvider, Provider
from .schema import AfterCallRecord, parse_reply, record_json_schema

MAX_TURNS = 200
MAX_CHARS = 20_000


class Turn(BaseModel):
    speaker: str = Field(pattern="^(상담원|고객)$")
    text: str = Field(min_length=1, max_length=2000)


class SummarizeRequest(BaseModel):
    domain: str
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


def create_app(provider: Provider | None = None) -> FastAPI:
    app = FastAPI(title="call-summary", version="0.1.0")
    state: dict = {
        "provider": provider
        or OllamaProvider(
            model=os.environ.get("CALL_SUMMARY_MODEL", "qwen2.5:7b-instruct"),
            base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
            use_schema=True,
        )
    }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": state["provider"].name, "prompt_hash": prompt_hash()}

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
        if req.domain not in DOMAINS:
            raise HTTPException(422, f"unknown domain {req.domain!r}")
        transcript = format_transcript([t.model_dump() for t in req.turns])
        if len(transcript) > MAX_CHARS:
            raise HTTPException(413, "transcript too long")
        messages = build_messages(req.domain, transcript)
        t0 = time.perf_counter()
        problems: list[str] = []
        for attempt in (1, 2):
            reply = p.generate(messages, json_schema=record_json_schema())
            parsed = parse_reply(reply.text)
            if parsed.record is None:
                problems = [parsed.error or "invalid"]
                continue
            problems = off_list_labels(req.domain, parsed.record)
            if not problems:
                return SummarizeResponse(
                    request_id=uuid.uuid4().hex,
                    record=parsed.record,
                    model=p.name,
                    attempts=attempt,
                    latency_s=round(time.perf_counter() - t0, 3),
                )
        raise HTTPException(502, {"error": "model output rejected", "problems": problems})

    return app
