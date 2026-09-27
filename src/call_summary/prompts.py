"""The summarizer prompt: the same text for every model, trained or not. Changing it after measuring starts
must be recorded in docs/experiments.md (its hash goes into every run manifest)."""

from __future__ import annotations

import hashlib
import json

from .domains import DOMAINS
from .schema import RESOLUTIONS, AfterCallRecord

PROMPT_VERSION = "p1"  # every model and all training use p1; p2 is a prompt-only comparison (stage 3)
PROMPT_VERSIONS = ("p1", "p2")

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

# p2 = p1 + the labelling conventions written out, as an annotation guideline would state them.
_GUIDE_P2 = """

[기록 기준]
- resolution
  - 해결: 고객의 요청을 상담 중에 처리했거나 원하는 답을 줌. 통화 뒤에 수거·환불 입금·새 상품 발송이 남아 있어도 요청 자체를 처리했으면 해결
  - 부분 해결: 일부만 처리했거나, 확인이 더 필요해 다시 연락하기로 했거나, 기사 방문·서류 제출·재입고 대기·변경 예약·임대폰 발송처럼 다음 단계를 잡아 둔 경우
  - 미해결: 규정이나 상황 때문에 요청을 들어줄 수 없었음. 고객이 더 생각해 보기로 하고 끝난 경우도 미해결
  - 이관: 다른 부서로 넘김
- actions_taken: 이름·생년월일 등으로 본인 확인을 했으면 '본인 확인'을 넣는다. 조회만 한 것도 그 조회 처리를 넣는다. 다른 부서로 넘겼으면 '전문 부서 이관'을 넣는다
- follow_up: 통화가 끝난 뒤에 일어나기로 한 일이 하나라도 있으면 required는 true이고 그 일을 codes에 모두 적는다. 요청이 해결됐어도 뒤에 남은 일이 있으면 적는다. 남은 일이 없으면 required는 false, codes는 빈 목록
  - 콜백 예약: 상담원이 정해진 시각에 고객에게 다시 연락하기로 함
  - 환불 처리 대기: 환불·환급을 접수했거나 결제를 취소해 돈이 돌아올 예정
  - 수거 예정: 반품이나 방문 접수로 기사가 물건을 가지러 가기로 함. 교환·불량 교환처럼 새 상품을 보내는 경우에는 재배송 예정만 적는다
  - 재배송 예정: 새 상품이나 원래 받아야 할 물건을 다시 보내기로 함
  - 보상 검토: 파손·분실·부상·부정 사용 등으로 보상 부서가 검토하기로 함
  - 그 밖의 코드(재입고 알림, 기사 방문 예정, 임대폰 배송, 서류 제출 대기, 재발급 카드 배송)는 이름 그대로의 일이 남은 경우"""


def system_prompt(domain_key: str, version: str = PROMPT_VERSION) -> str:
    d = DOMAINS[domain_key]

    def join(xs: tuple[str, ...]) -> str:
        return ", ".join(xs) if xs else "(없음)"

    if version not in PROMPT_VERSIONS:
        raise ValueError(version)
    text = _SYSTEM + (_GUIDE_P2 if version == "p2" else "")
    return text.format(
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
    domain_key: str,
    transcript: str,
    shots: list[tuple[str, AfterCallRecord]] | None = None,
    version: str = PROMPT_VERSION,
) -> list[dict]:
    """Chat messages for one call. `shots` are (transcript, record) examples from the same domain."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt(domain_key, version)}]
    for shot_transcript, shot_record in shots or []:
        msgs.append({"role": "user", "content": user_prompt(shot_transcript)})
        msgs.append({"role": "assistant", "content": record_to_json(shot_record)})
    msgs.append({"role": "user", "content": user_prompt(transcript)})
    return msgs


def record_to_json(record: AfterCallRecord) -> str:
    """The training target text: compact but readable, keys in schema order, Hangul unescaped."""
    return json.dumps(record.model_dump(), ensure_ascii=False)


def prompt_hash(version: str = PROMPT_VERSION) -> str:
    """p1's hash is computed exactly as before p2 existed, so earlier manifests still match."""
    h = hashlib.sha256()
    h.update(version.encode())
    h.update(_SYSTEM.encode())
    if version == "p2":
        h.update(_GUIDE_P2.encode())
    for key in sorted(DOMAINS):
        h.update(system_prompt(key, version).encode())
    return h.hexdigest()[:12]
