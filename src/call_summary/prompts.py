"""The summarizer prompt: the same text for every model, trained or not. Changing it after measuring starts
must be recorded in docs/experiments.md (its hash goes into every run manifest)."""

from __future__ import annotations

import hashlib
import json

from .domains import DOMAINS
from .schema import RESOLUTIONS, AfterCallRecord

PROMPT_VERSION = "p1"

_SYSTEM = """당신은 {company}({label}) 고객센터의 상담 후처리 담당자입니다. 상담 대화 전사를 읽고 상담 기록을 JSON 하나로만 작성하세요. JSON 밖에는 아무것도 쓰지 마세요.

필드:
- category: 문의 유형. 다음 중 하나: {categories}
- resolution: 처리 결과. 다음 중 하나: {resolutions}
  (해결: 요청이 상담 중에 처리됨 / 부분 해결: 일부만 처리되었거나 확인이 더 필요함 / 미해결: 요청을 들어줄 수 없었음 / 이관: 다른 부서로 넘김)
- entities: 상담에서 확인된 핵심 값 목록. 각 항목은 {{"type": 유형, "value": 대화에 나온 값}}. 유형은 다음 중에서: {entity_types}
  고객이 잘못 말했다가 고친 값은 고친 값만 적고, 대화에 나오지 않은 값은 절대 지어내지 마세요.
- actions_taken: 상담원이 실제로 한 처리 목록. 다음 중에서 고르고 하지 않은 처리는 넣지 마세요: {actions}
- follow_up: {{"required": 후속 조치가 필요한지 true/false, "codes": 후속 조치 목록}}. 목록은 다음 중에서: {follow_ups}
- summary: 상담 내용을 3~5문장의 한국어로 요약. 고객의 요청, 상담원이 확인하거나 처리한 것, 결과와 후속 조치를 담으세요."""


def system_prompt(domain_key: str) -> str:
    d = DOMAINS[domain_key]

    def join(xs: tuple[str, ...]) -> str:
        return ", ".join(xs) if xs else "(없음)"

    return _SYSTEM.format(
        company=d.company,
        label=d.label,
        categories=join(d.categories),
        resolutions=join(RESOLUTIONS),
        entity_types=join(tuple(et.label for et in d.entity_types)),
        actions=join(d.actions),
        follow_ups=join(d.follow_up_codes),
    )


def format_transcript(turns: list[dict]) -> str:
    """turns: [{"speaker": "상담원"|"고객", "text": ...}]"""
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def user_prompt(transcript: str) -> str:
    return f"상담 대화:\n{transcript}\n\n상담 기록 JSON:"


def build_messages(
    domain_key: str, transcript: str, shots: list[tuple[str, AfterCallRecord]] | None = None
) -> list[dict]:
    """Chat messages for one call. `shots` are (transcript, record) examples from the same domain."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt(domain_key)}]
    for shot_transcript, shot_record in shots or []:
        msgs.append({"role": "user", "content": user_prompt(shot_transcript)})
        msgs.append({"role": "assistant", "content": record_to_json(shot_record)})
    msgs.append({"role": "user", "content": user_prompt(transcript)})
    return msgs


def record_to_json(record: AfterCallRecord) -> str:
    """The training target text: compact but readable, keys in schema order, Hangul unescaped."""
    return json.dumps(record.model_dump(), ensure_ascii=False)


def prompt_hash() -> str:
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    h.update(_SYSTEM.encode())
    for key in sorted(DOMAINS):
        h.update(system_prompt(key).encode())
    return h.hexdigest()[:12]
