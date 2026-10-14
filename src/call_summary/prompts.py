"""요약 프롬프트: 학습했든 안 했든 모든 모델에 같은 글을 준다. 측정을 시작한 뒤에 바꾸면
docs/experiments.md에 기록해야 한다(프롬프트 해시가 모든 실행 기록(manifest)에 들어간다)."""

# [파일 역할] 상담 후처리 기록을 요청하는 프롬프트(시스템 지시문, 사용자 메시지, few-shot 예시)를 만드는
# 모듈이다. 학습 전 모델, 비교용 큰 모델, 학습한 학생, 서비스가 모두 이 함수로 같은 메시지를 만든다.
# [왜 필요한가] 측정이 공정하려면 모든 모델이 같은 지시문을 받아야 하고, 학습한 모델은 학습 때 본 프롬프트를
# 서빙 때도 그대로 받아야 한다(train.py docstring "프롬프트는 평가 프롬프트와 똑같고"). 그래서
# 프롬프트를 한곳에 두고, 바뀌었는지는 prompt_hash로 실행 기록(manifest)에 남겨 확인한다.
# [주요 구성] _SYSTEM(p1 지시문) / _GUIDE_P2(p2에 덧붙이는 기록 기준) → system_prompt(업종 라벨을 채움) →
# format_transcript·user_prompt(대화 부분) → build_messages(채팅 메시지 목록) → record_to_json(학습 목표
# 문자열) → prompt_hash(지시문 지문).

from __future__ import annotations

# hashlib: 프롬프트 지문(sha256 앞 12자리)을 만들어 실행 기록에 남긴다. json: 정답 기록을 학습 목표 문자열로
# 직렬화한다.
import hashlib
import json

# DOMAINS: 업종별 회사 이름·문의 유형·핵심 값 유형·처리·후속 조치 목록. 지시문에 이 목록을 그대로 넣어, 모델이
# 고를 수 있는 라벨을 업종마다 닫힌 집합으로 알려 준다.
# RESOLUTIONS: 처리 결과 네 가지(해결/부분 해결/미해결/이관). AfterCallRecord: 출력 스키마.
from .domains import DOMAINS
from .schema import RESOLUTIONS, AfterCallRecord

# PROMPT_VERSION: 기본 프롬프트. 모든 모델과 모든 학습이 p1을 쓴다.
# PROMPT_VERSIONS: 고를 수 있는 판. p2는 3단계에서 "학습 이득이 기록 관례를 배운 것뿐인가"를 가르려고 14B에 준
# 비교용 프롬프트로, 학습 없이 프롬프트만 바꾼 비교에만 쓴다(docs/experiments.md 2단계 오류 분석).
# 2026-10-05 비교 대상(qwen3.5:4b 등)의 후보로도 쓰였다. evaluate.py의 --prompt 선택지가 이 튜플이다.
PROMPT_VERSION = "p1"
PROMPT_VERSIONS = ("p1", "p2")

# p1 시스템 지시문. {company}·{categories} 같은 자리는 system_prompt가 업종 정보로 채운다. JSON 예시의
# 중괄호는 str.format이 자리 표시로 읽지 않도록 두 겹으로 적었다.
# 지시문은 "JSON 하나로만" 답하라고 하고, 필드마다 고를 수 있는 목록을 나열하며, 대화에 없는 값을 지어내지
# 말고 고객이 고친 값은 고친 값만 적으라고 한다. 고친 값 규칙은 명세의 correction 사건(고객이 번호를 잘못
# 말했다 고침)과 짝을 이룬다.
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

# _GUIDE_P2: p2 = p1 + 이 글이다. 명세 생성기의 기록 관례(해결/부분 해결의 경계, 본인 확인을 처리에 넣는 것,
# 해결돼도 남은 일은 후속 조치로 적는 것 등)를 라벨링 지침서가 적듯이 풀어 쓴 글이다. 2단계에서 학습 전 모델이
# 틀린 것의 상당수가 프롬프트에 적지 않은 관례였기 때문에 추가했다(README 3단계 결과 요약). 모든 경우를 담지는
# 않는다(docs/experiments.md).
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


# 업종 하나의 시스템 지시문을 만든다. 입력: 업종 키(shop, telecom, parcel, card)와 프롬프트 판. 출력: 라벨
# 목록을 채운 지시문 문자열.
# 알 수 없는 판 이름은 ValueError로 막아, 오타로 다른 프롬프트를 쓴 측정이 조용히 기록되지 않게 한다.
def system_prompt(domain_key: str, version: str = PROMPT_VERSION) -> str:
    d = DOMAINS[domain_key]

    # 목록을 쉼표로 잇는 도우미. 비어 있으면 "(없음)"으로 적어 지시문에 빈 칸이 생기지 않게 한다.
    def join(xs: tuple[str, ...]) -> str:
        return ", ".join(xs) if xs else "(없음)"

    if version not in PROMPT_VERSIONS:
        raise ValueError(version)
    # p2는 p1 뒤에 기록 기준을 덧붙인 것이므로, p1 부분은 두 판에서 글자까지 같다.
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


# 발화 목록을 "상담원: ..." / "고객: ..." 줄로 잇는다. 데이터셋(Item.transcript), 서비스 요청, 음성 조건
# 전사가 모두 이 꼴로 바뀌어 모델에 들어간다.
def format_transcript(turns: list[dict]) -> str:
    """turns는 [{"speaker": "상담원"|"고객", "text": ...}] 꼴의 발화 목록이다."""
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


# 사용자 메시지: 전사를 감싸고 "상담 기록 JSON:"으로 끝내, 모델이 바로 JSON을 쓰기 시작하도록 유도한다.
def user_prompt(transcript: str) -> str:
    return f"상담 대화:\n{transcript}\n\n상담 기록 JSON:"


# 한 통화의 채팅 메시지 목록을 만든다. 순서: system → (예시마다 user 전사 → assistant 정답 JSON) → user 실제
# 전사.
# shots를 주면 few-shot(2단계 기준선의 2-shot, 2026-10-05 비교 대상)이 되고, 학습·서빙은 shots 없이 0-shot으로
# 부른다. 예시 정답도 record_to_json으로 직렬화해 학습 목표와 같은 꼴을 보여 준다.
# 호출 위치: evaluate.py, train.py(학습 입력), service.py(서빙), bench.py.
def build_messages(
    domain_key: str,
    transcript: str,
    shots: list[tuple[str, AfterCallRecord]] | None = None,
    version: str = PROMPT_VERSION,
) -> list[dict]:
    """통화 하나의 채팅 메시지 목록. `shots`는 같은 업종의 (전사, 기록) 예시다."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt(domain_key, version)}]
    for shot_transcript, shot_record in shots or []:
        msgs.append({"role": "user", "content": user_prompt(shot_transcript)})
        msgs.append({"role": "assistant", "content": record_to_json(shot_record)})
    msgs.append({"role": "user", "content": user_prompt(transcript)})
    return msgs


# 정답 기록을 한 줄 JSON 문자열로 만든다. 학습 목표이자 few-shot 예시 답이고, 오프라인 서비스의 고정 답도 이
# 함수로 만든다.
# model_dump는 스키마에 정의한 필드 순서를 지키므로 키 순서가 늘 같다. 5단계에서 JSON 스키마 강제 디코딩이 키
# 순서를 바꾸자 학습한 모델이 필드를 빠뜨린 일이 있어(README 5단계), 학습 때의 키 순서는 이 함수가 정한 것이
# 기준이다. ensure_ascii=False로 한글을 \u 이스케이프 없이 그대로 써 사람이 읽을 수 있게 한다(docstring).
def record_to_json(record: AfterCallRecord) -> str:
    """학습 목표 글: 짧지만 읽을 수 있게, 키는 스키마 순서대로, 한글은 이스케이프하지 않는다."""
    return json.dumps(record.model_dump(), ensure_ascii=False)


# 프롬프트 판의 지문(sha256 앞 12자리)을 만든다. 실행 기록(evaluate, train, bench의 manifest)과 서비스 /health
# 응답에 들어가, 측정이 어떤 프롬프트로 돌았는지와 서빙 프롬프트가 학습 때와 같은지 확인하는 데 쓴다.
# 판 이름, 지시문 원문, 네 업종의 완성된 지시문을 모두 해시에 넣어, 업종 라벨 목록(domains.py)만 바뀌어도
# 지문이 바뀐다.
# p2가 생기기 전과 같은 순서로 p1을 계산해 예전 실행 기록의 지문이 그대로 맞는다(테스트가 p1 지문 값을 고정해
# 확인한다).
def prompt_hash(version: str = PROMPT_VERSION) -> str:
    """p1의 해시는 p2가 생기기 전과 똑같이 계산하므로 예전 실행 기록(manifest)의 해시가 그대로 맞는다."""
    h = hashlib.sha256()
    h.update(version.encode())
    h.update(_SYSTEM.encode())
    # p2일 때만 기록 기준 원문을 더 넣는다. p1 경로에 이 줄이 끼지 않아야 예전 p1 지문이 그대로다.
    if version == "p2":
        h.update(_GUIDE_P2.encode())
    for key in sorted(DOMAINS):
        h.update(system_prompt(key, version).encode())
    return h.hexdigest()[:12]
