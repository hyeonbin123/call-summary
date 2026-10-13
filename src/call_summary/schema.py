"""The after-call record: the one output format every model is asked for and every score is computed on."""

# [파일 역할] 상담 후처리 기록(AfterCallRecord)의 형식을 pydantic 모델로 정의하고, 모델이 낸 글(답)에서 JSON을
# 꺼내 이 형식으로 검증하는 함수를 둔다.
# [왜 필요한가] 교사·비교 모델·학습한 학생·서비스가 모두 같은 출력 형식을 내야 같은 채점기로 잴 수 있다. 형식
# 정의를 한 곳에 두어 프롬프트(prompts.py), 정답 생성(specs.py), 채점(scoring.py), 서비스 응답
# 검증(service.py)이 같은 정의를 쓴다(docs/experiments.md "공통": 출력 스키마는 이 파일의 AfterCallRecord).
# [주요 구성] RESOLUTIONS(처리 결과 4가지) → Entity·FollowUp·AfterCallRecord(기록 형식) →
# record_json_schema(제약 디코딩용 JSON Schema) → ParseResult와 parse_reply(답 글 → 검증된 기록).

from __future__ import annotations

# json: 답 글에서 잘라 낸 {...} 부분을 파이썬 dict로 읽는다. Literal: 처리 결과를 정해진 네 값으로만 받게
# 한다.
import json
from typing import Literal

# pydantic: 필드 타입 검사, 알 수 없는 키 거부(extra="forbid"), JSON Schema 생성을 직접 짜지 않고 쓰려고
# 가져온다.
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 처리 결과(resolution)의 허용 값. 튜플은 프롬프트에 선택지를 나열할 때(prompts.py), Literal 타입은 검증할 때
# 쓴다. 두 곳에 같은 값을 적어 두었으므로 하나를 바꾸면 다른 하나도 바꿔야 한다.
RESOLUTIONS: tuple[str, ...] = ("해결", "부분 해결", "미해결", "이관")
Resolution = Literal["해결", "부분 해결", "미해결", "이관"]


# 핵심 값 하나(예: 주문번호 D1234567). type은 업종별 목록(domains.py)의 유형 이름, value는 통화에서 말한
# 그대로의 값이다. 비교할 때는 values.normalize로 정규화하므로 "3만 2천 원"과 "32,000원"을 같은 값으로 본다.
# extra="forbid": 모델이 정의에 없는 키(예: confidence)를 붙이면 스키마 통과로 치지 않는다. 형식 준수 자체를
# 재는 지표(schema)가 있어서다(tests/test_scoring.py가 확인).
class Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Entity type label from the domain's list, e.g. 주문번호")
    value: str = Field(description="The value as it was said in the call")


# 후속 조치: 통화 뒤에 남은 일이 있는지(required)와 그 코드 목록(codes). 코드는 업종별 목록에서 고른다.
class FollowUp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool
    codes: list[str] = Field(default_factory=list)


# 상담 후처리 기록 전체. 계획(docs/plan.md "과제 정의")의 필드 표와 같다: 문의 유형, 처리 결과, 핵심 값, 상담
# 중 한 처리 목록, 후속 조치, 3~5문장 요약.
# entities·actions_taken은 기본값(빈 목록)이 있어 빠져도 검증을 통과한다. 그래서 스키마 강제 디코딩에서
# actions_taken이 빠지는 일이 생겼다(docs/experiments.md 5단계 발견 1).
class AfterCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    resolution: Resolution
    entities: list[Entity] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    follow_up: FollowUp
    summary: str


# 기록 형식의 JSON Schema를 돌려준다. Ollama의 format 인자에 넘겨 디코딩 단계에서 JSON 형식을 강제할 때 쓴다.
# 호출 위치: evaluate.py(--schema), service.py(CALL_SUMMARY_SCHEMA=1일 때만), bench.py. 서비스는 이 강제가 키
# 순서를 바꿔 학습한 모델이 필드를 빠뜨려 기본으로 끈다(docs/plan.md "계획과 달라진 것").
def record_json_schema() -> dict:
    """JSON Schema handed to constrained decoders (Ollama `format`)."""
    return AfterCallRecord.model_json_schema()


# 답 하나를 읽은 결과. JSON이 있었는지와 스키마를 통과했는지를 나눠 둔다. 채점(scoring.score_item)은
# schema_ok가 아니면 모든 필드를 오답으로 치고, 평가 기록에는 실패 이유(error)를 남긴다.
class ParseResult(BaseModel):
    """What came out of one raw model reply."""

    json_ok: bool  # the reply contained a JSON object
    schema_ok: bool  # ...and it validated as an AfterCallRecord
    record: AfterCallRecord | None = None
    error: str | None = None


# 답 글에서 처음 나오는 짝이 맞는 {...} 구간을 찾는다. 모델이 ```json 울타리나 앞뒤 설명을 붙여도 JSON만
# 꺼내려는 것이다.
# 정규식 대신 한 글자씩 세는 이유: 요약 문자열 안의 중괄호("{괄호}")나 이스케이프된 따옴표를 괄호 깊이에 세지
# 않아야 하기 때문이다(tests/test_scoring.py의 brace_in_string).
def _first_json_object(text: str) -> str | None:
    """Return the first balanced {...} span, ignoring braces inside strings. Tolerates ``` fences."""
    start = text.find("{")
    # 여는 중괄호 위치마다 시도한다. 그 위치에서 짝이 닫히지 않으면 다음 여는 중괄호부터 다시 센다.
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            # 문자열 안에서는 백슬래시 다음 글자를 건너뛰고, 닫는 따옴표만 본다. 문자열 밖에서만 중괄호 깊이를
            # 센다.
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
                # 깊이가 0으로 돌아오면 바깥 객체가 닫힌 것이므로 그 구간을 돌려준다.
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


# 모델 답 글 하나를 ParseResult로 바꾼다. 단계: JSON 구간 찾기 → json.loads → 객체인지 확인 → AfterCallRecord
# 검증.
# 실패 단계에 따라 json_ok·schema_ok를 다르게 두어, 형식 실패를 "JSON이 아님"과 "JSON이지만 스키마 위반"으로
# 나눠 볼 수 있다. 예외를 밖으로 던지지 않으므로 평가 루프와 서비스가 답마다 try 없이 부를 수 있다.
# 호출 위치: evaluate.py, service.py(재시도 판단), service_eval.py, bench.py.
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
        # JSON으로는 읽혔지만 스키마를 통과하지 못한 경우다. 오류 내용 대신 오류 개수만 error에 남긴다.
        return ParseResult(json_ok=True, schema_ok=False, error=f"schema: {exc.error_count()} errors")
    return ParseResult(json_ok=True, schema_ok=True, record=record)
