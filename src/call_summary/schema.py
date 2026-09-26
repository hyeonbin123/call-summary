"""The after-call record: the one output format every model is asked for and every score is computed on."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

RESOLUTIONS: tuple[str, ...] = ("해결", "부분 해결", "미해결", "이관")
Resolution = Literal["해결", "부분 해결", "미해결", "이관"]


class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Entity type label from the domain's list, e.g. 주문번호")
    value: str = Field(description="The value as it was said in the call")


class FollowUp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool
    codes: list[str] = Field(default_factory=list)


class AfterCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    resolution: Resolution
    entities: list[Entity] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    follow_up: FollowUp
    summary: str


def record_json_schema() -> dict:
    """JSON Schema handed to constrained decoders (Ollama `format`)."""
    return AfterCallRecord.model_json_schema()


class ParseResult(BaseModel):
    """What came out of one raw model reply."""

    json_ok: bool  # the reply contained a JSON object
    schema_ok: bool  # ...and it validated as an AfterCallRecord
    record: AfterCallRecord | None = None
    error: str | None = None


def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} span, ignoring braces inside strings. Tolerates ``` fences."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_reply(text: str) -> ParseResult:
    span = _first_json_object(text)
    if span is None:
        return ParseResult(json_ok=False, schema_ok=False, error="no JSON object")
    try:
        data = json.loads(span)
    except json.JSONDecodeError as exc:
        return ParseResult(json_ok=False, schema_ok=False, error=f"json: {exc.msg}")
    if not isinstance(data, dict):
        return ParseResult(json_ok=False, schema_ok=False, error="not an object")
    try:
        record = AfterCallRecord.model_validate(data)
    except ValidationError as exc:
        return ParseResult(json_ok=True, schema_ok=False, error=f"schema: {exc.error_count()} errors")
    return ParseResult(json_ok=True, schema_ok=True, record=record)
