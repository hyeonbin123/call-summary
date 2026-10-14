"""HTTP 서비스: POST /summarize가 상담 전사를 상담 후처리 기록으로 바꾼다.

모델 답은 AfterCallRecord로 검증을 통과하고 업종의 라벨 목록 안의 값만 써야 한다. 그렇지 않으면 서비스는
샘플링으로 한 번 다시 시도하고(탐욕 디코딩으로 다시 하면 거절한 답이 그대로 나온다), 그래도 안 되면 이유와
함께 502로 답한다. 반쯤 맞는 기록은 절대 내보내지 않는다. 모델 서버에 닿지 못하면 503을 준다.

기록 안의 번호(주문번호, 운송장번호, 접수번호, 카드 끝자리) 가운데 운영 형식에 맞지 않거나 전사에서 찾을
수 없는 것은 `needs_confirmation`에 나열한다(verify.py). 기록 자체는 바꾸지 않는다.

처리기에 닿은 /summarize 호출마다 JSON 한 줄을 로그로 남긴다(로거 `call_summary.service`, INFO):
요청 id, 업종, 발화 수와 글자 수, 결과, HTTP 상태, 시도 횟수, 지연, 시도들을 합친 프롬프트·생성 토큰
수(모델 서버가 알려 주지 않으면 null), 확인이 필요한 번호의 개수. 전사, 기록, 표시한 번호 값은 절대
남기지 않는다: 개인정보가 들어 있다.

모델과 함께 실행:  CALL_SUMMARY_MODEL=<ollama 모델> uvicorn call_summary.service:create_app --factory
모델 없이 실행:    uvicorn call_summary.service:create_offline_app --factory   (고정 답, 스캔용)
"""

# [파일 역할] 5단계 서빙의 HTTP 서비스(FastAPI). 상담 전사를 받아 Ollama에 올린 학습 모델(GGUF q8_0)로 상담
# 후처리 기록을 만들고, 검증·재시도·번호 확인 표시를 거쳐 돌려준다.
# [설계 이유] 모델 답을 그대로 내보내지 않고 (1) 스키마 검증, (2) 업종 라벨 목록 검사, (3) 실패 시 샘플링으로
# 한 번만 재시도, (4) 그래도 안 되면 반쯤 맞는 기록 대신 502를 준다(docs/plan.md "서빙"). 스키마 강제 디코딩은
# 학습한 모델의 필드를 빠뜨리게 해 기본으로 끈다(docs/experiments.md 5단계 발견 1).
# [엔드포인트] GET /health(모델 이름·프롬프트 지문), GET /stats(요청 수·지연 분포), GET /domains(업종별 라벨
# 목록), POST /summarize(본 기능).
# [보안·개인정보] 응답에 보안 헤더를 붙이고, 문서 페이지는 기본으로 숨긴다. 요청 로그에는 전사·기록·번호 값을
# 남기지 않는다(개인정보). 모델 없는 오프라인 모드(create_offline_app)는 HawkScan 보안
# 스캔용이다(docs/experiments.md "보안 스캔 기록").
# [연결] service_eval.py·loadtest.py가 이 서비스에 요청을 보내 품질과 부하를 재고, Dockerfile·compose.yaml이
# uvicorn으로 create_app을 띄운다.

from __future__ import annotations

# 표준 라이브러리: dataclasses.replace로 재시도용 provider를 복사해 만들고, threading.Lock으로 /stats 집계를
# 동시 요청에서 지키며, uuid로 요청 id를, time.perf_counter로 지연을 잰다. random은 오프라인 모드의 고정
# 번호를 만드는 데만 쓴다.
import dataclasses
import json
import logging
import os
import random
import sys
import threading
import time
import uuid
from typing import Literal

# FastAPI: 요청 본문을 pydantic 모델로 검증하고(형식이 틀리면 처리기 전에 422), OpenAPI 설명을 만든다.
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

# 프로젝트 모듈: 업종 라벨 목록(domains), 학습·평가와 같은 프롬프트 조립(prompts. 서빙 프롬프트가 학습 때와
# 같아야 한다), 모델 호출(providers), 기록 형식과 답 읽기(schema), 확인이 필요한 번호 검사(verify), /stats의
# 지연 백분위(scoring.pctl, 평가 도구와 같은 계산).
from .domains import DOMAINS
from .prompts import build_messages, format_transcript, prompt_hash, record_to_json
from .providers import OllamaProvider, Provider, Reply
from .schema import AfterCallRecord, Entity, FollowUp, parse_reply, record_json_schema
from .scoring import pctl
from .verify import IdentifierFlag, verify_entities

# 한 요청에 받는 발화 수의 상한(입력 크기 제한).
MAX_TURNS = 200
# Ollama에 요청하는 컨텍스트 길이. 모델을 내보낼 때 정한 num_ctx(README의 export --num-ctx 4096)와 같다.
NUM_CTX = 4096
# MAX_CHARS는 처음 20,000자였다가 컨텍스트 4,096토큰에 프롬프트와 답이 함께 들어가도록 4,000자로 줄였다(커밋
# fdc1b9c). 넘으면 413을 준다(README 서비스 문단).
# 프롬프트 토큰 수 ~= 571 + 0.605 * 전사 글자 수(dev 실행, Qwen3·qwen2.5 토크나이저로 잰 값). 4,000자는 약
# 3,000 프롬프트 토큰이라 답에 약 1,000토큰이 남는다. 더 긴 입력은 모델 서버가 잘라 버린다.
# 데이터셋 전사는 가장 긴 것이 1,485자다.
MAX_CHARS = 4_000
# 모든 응답에 붙이는 보안 헤더. 이 서비스는 JSON만 주므로 브라우저가 내용을 실행하거나 다른 페이지에 끼워 넣을
# 일이 없게 막는다: MIME 추측 금지(nosniff), 프레임 금지(X-Frame-Options, frame-ancestors), 리퍼러 안 보냄,
# 어떤 리소스도 불러오지 않는 CSP, 카메라·마이크·위치 권한 끔, 응답 캐시 금지(전사·기록에 개인정보가 있다).
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Cache-Control": "no-store",
}
# 요청마다 JSON 한 줄을 남기는 로거. README에 적힌 로거 이름이고, tests/test_service.py가 이 이름으로 로그
# 줄을 잡아 내용을 확인한다.
log = logging.getLogger("call_summary.service")


# uvicorn은 자기 로거만 설정하므로, 이 로거에 처리기를 붙이지 않으면 INFO 줄이 버려진다. 이미 처리기가
# 있으면(앱이나 pytest가 루트 로거를 설정한 경우) 건드리지 않아 줄이 두 번 찍히지 않게 한다. create_app이 앱을
# 만들 때 한 번 부른다.
def _ensure_log_handler() -> None:
    """uvicorn은 자기 로거만 설정하므로, 처리기가 없으면 이 모듈의 INFO 줄이 버려진다.
    앱(또는 pytest)이 이미 루트 로거에서 로깅을 처리하면 건드리지 않는다."""
    if log.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# 시도별 토큰 수를 더한다. 모델 서버가 토큰 수를 주지 않은 시도(None)는 빼고, 하나도 없으면 0이 아니라
# None(모름)을 돌려준다.
def _token_sum(values: list[int | None]) -> int | None:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


# OpenAPI 설명에 넣는 요청 예시. 보안 스캐너가 OpenAPI만 보고도 입력 검사를 통과하는 요청을 만들 수 있게
# 붙였다(커밋 a13fcb6, docs/experiments.md "보안 스캔 기록" 2026-10-03). hawk/summarize.har 씨앗 본문도 이
# 예시와 같다(tests/test_service.py가 확인).
EXAMPLE_REQUEST = {
    "domain": "shop",
    "turns": [
        {"speaker": "상담원", "text": "도토리마켓입니다. 무엇을 도와드릴까요?"},
        {"speaker": "고객", "text": "어제 한 주문을 취소하려고요."},
    ],
}


# 발화 하나. speaker는 정확히 "상담원"·"고객"만 받는다. text는 2,000자 상한.
class Turn(BaseModel):
    # 정규식이 아니라 Literal로 둔다: OpenAPI의 enum으로 클라이언트와 스캐너가 어떤 값이 통과하는지 안다.
    speaker: Literal["상담원", "고객"]
    # 빈 글도 받는다: 음성 인식기는 어떤 발화에서 아무것도 돌려주지 않는다(test-d에 그런 발화가 있다).
    text: str = Field(max_length=2000)


# POST /summarize 요청 본문. domain은 업종 키(shop, telecom, parcel, card), turns는 발화 1개 이상 MAX_TURNS개
# 이하.
class SummarizeRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    # enum은 여기서 문서(OpenAPI)에만 적고 검사는 처리기가 한다. 그래야 목록 밖 업종도 자기 422("unknown
    # domain")와 로그 한 줄을 그대로 받는다.
    domain: str = Field(max_length=32, json_schema_extra={"enum": list(DOMAINS)})
    turns: list[Turn] = Field(min_length=1, max_length=MAX_TURNS)


# POST /summarize 응답. record는 검증을 통과한 기록 그대로이고, needs_confirmation은 기록을 바꾸지 않고 따로
# 붙이는 번호 확인 목록이다(docs/experiments.md "확인이 필요한 번호 표시": 정확도 이득이 아니라 확인할 일을
# 상담원에게 알리는 장치). attempts는 1 또는 2(재시도해서 통과).
class SummarizeResponse(BaseModel):
    request_id: str
    record: AfterCallRecord
    # 처리하기 전에 고객에게 다시 읽어 확인할 번호: 형식이 틀렸거나 전사에 없는 것.
    needs_confirmation: list[IdentifierFlag] = Field(default_factory=list)
    model: str
    attempts: int
    latency_s: float


# 기록이 업종 허용 목록 밖의 라벨(유형, 처리, 후속 조치 코드, 핵심 값 유형)을 썼는지 찾아 문제 목록으로
# 돌려준다. 비어 있지 않으면 /summarize가 재시도하고, 두 번째도 그러면 502 응답의 problems에 이 목록을 넣는다.
# 채점의 off_list(scoring.py)와 같은 기준이다.
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


# 재시도용 provider를 만든다. 첫 시도는 탐욕 디코딩(temperature 0)이라 같은 요청을 다시 보내면 거절한 답이
# 그대로 나온다. 그래서 같은 모델에 temperature 0.3, seed+1로 샘플링한다(커밋 1344e49).
# dataclasses.replace로 새 객체를 만들어 공유 provider를 바꾸지 않는다. num_ctx·num_gpu는 그대로 두어 Ollama가
# 모델을 다시 올리지 않게 한다. Ollama가 아닌 provider(테스트용)는 그대로 돌려준다.
def _retry_provider(p: Provider) -> Provider:
    """두 번째 시도는 거절한 탐욕 디코딩을 되풀이하면 안 된다: 같은 모델·컨텍스트로 샘플링한다."""
    if isinstance(p, OllamaProvider):
        return dataclasses.replace(p, temperature=0.3, seed=p.seed + 1)
    return p


# 이 프로세스가 받은 /summarize의 결과별 수와 지연 목록(GET /stats). 메모리에만 있어 재시작하면 0부터 다시
# 센다. FastAPI는 일반 def 처리기를 스레드 풀에서 동시에 돌리므로 Lock으로 읽기·쓰기를 지킨다.
class Stats:
    """이 프로세스가 받은 /summarize 호출의 결과별 수와 지연(재시작하면 처음부터 다시 센다)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: list[float] = []
        self.counts = {"ok": 0, "retried_ok": 0, "rejected": 0, "model_unavailable": 0}

    # 결과 하나를 센다. 지연은 기록이 나간 요청(ok, retried_ok)만 넘겨받는다.
    def add(self, outcome: str, latency: float | None = None) -> None:
        with self._lock:
            self.counts[outcome] += 1
            if latency is not None:
                self.latencies.append(latency)

    # 현재 수와 지연 분포(p50, p95, 최대)를 돌려준다. Lock 안에서는 목록을 복사·정렬만 하고 계산은 밖에서
    # 한다. p50·p95는 scoring.pctl(evaluate·loadtest와 같은 계산, 보간 없음)이다.
    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies)
        out: dict = {"requests": dict(self.counts), "latency_s": None}
        if lat:
            out["latency_s"] = {
                "n": len(lat),
                "p50": round(pctl(lat, 0.5), 3),
                "p95": round(pctl(lat, 0.95), 3),
                "max": round(lat[-1], 3),
            }
        return out


# 앱 공장 함수(uvicorn --factory로 부른다). provider를 넘기면 그것을(테스트·오프라인 모드), 아니면 환경 변수로
# OllamaProvider를 만든다.
# 환경 변수: CALL_SUMMARY_MODEL(Ollama 모델 태그), OLLAMA_URL, CALL_SUMMARY_SCHEMA=1(스키마 강제 켜기),
# CALL_SUMMARY_NUM_GPU(GPU에 올릴 층 수. 14B처럼 VRAM이 넘치는 모델용, docs/experiments.md 1단계 진행 기록),
# CALL_SUMMARY_EXPOSE_OPENAPI=1(OpenAPI 공개).
def create_app(provider: Provider | None = None, expose_openapi: bool | None = None) -> FastAPI:
    if expose_openapi is None:
        expose_openapi = os.environ.get("CALL_SUMMARY_EXPOSE_OPENAPI") == "1"
    # Swagger·ReDoc 문서 페이지는 끄고, OpenAPI JSON은 요청할 때만 연다(오프라인 모드는 스캔을 위해 연다).
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
        # 기본은 끈다. Ollama의 스키마 문법은 필수 키를 알파벳 순서로 먼저 쓰게 하는데, 한 가지 키 순서로
        # 학습한 모델은 그러면 `actions_taken`을 빠뜨린다(docs/experiments.md 5단계).
        use_schema=os.environ.get("CALL_SUMMARY_SCHEMA") == "1",
        num_ctx=NUM_CTX,
        num_gpu=int(os.environ["CALL_SUMMARY_NUM_GPU"]) if os.environ.get("CALL_SUMMARY_NUM_GPU") else None,
    )
    # 재시도 provider도 앱을 만들 때 한 번만 만든다. 요청은 스레드 풀에서 동시에 돌므로, 요청마다 provider의
    # 설정을 바꾸면 동시에 도는 다른 요청이 바뀐 설정을 쓸 수 있다. 그래서 공유 provider는 절대 바꾸지 않는다.
    state: dict = {"provider": base, "retry_provider": _retry_provider(base)}

    # 모든 응답(오류 응답 포함)에 보안 헤더를 붙이는 미들웨어. setdefault라 처리기가 이미 정한 헤더는 덮어쓰지
    # 않는다.
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response

    # 상태 확인. 모델 이름과 프롬프트 지문(prompt_hash)을 함께 돌려 서빙 프롬프트가 학습·평가 때와 같은지
    # 확인할 수 있다. compose.yaml의 healthcheck가 이 경로를 부른다.
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": state["provider"].name, "prompt_hash": prompt_hash()}

    @app.get("/stats")
    def get_stats() -> dict:
        return stats.snapshot()

    # 업종별 라벨 목록. 호출하는 쪽(화면 등)이 받을 수 있는 유형·처리·코드를 알 수 있게 한다.
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

    # 본 기능. 단계: 로그 줄 준비 → 업종 확인(422) → 전사 조립·길이 확인(413) → 프롬프트 조립 → 최대 두 번
    # 시도(모델 오류 503, 형식·라벨 검사 실패면 재시도) → 번호 확인 표시 → 응답, 두 번 모두 실패하면 502.
    # 처리기에 닿은 요청은 어느 결과로 끝나든 finish()로 로그 한 줄을 남긴다. 본문 형식 검사(빠진 필드,
    # 허용되지 않은 발화자)에서 걸린 422는 처리기에 닿지 않아 줄이 없다(README 서비스 문단).
    @app.post("/summarize", response_model=SummarizeResponse)
    def summarize(req: SummarizeRequest) -> SummarizeResponse:
        p: Provider = state["provider"]
        request_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        # 요청 로그 한 줄의 틀. 개수·결과·지연·토큰 수만 담고 전사·기록·번호 값은 담지 않는다. 목록 밖 업종은
        # 받은 문자열을 그대로 남기지 않고 None으로 둔다.
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
            "needs_confirmation": None,
            "model": p.name,
        }
        prompt_tokens: list[int | None] = []
        completion_tokens: list[int | None] = []

        # 결과·상태·지연·시도를 합친 토큰 수를 채워 로그 한 줄을 쓴다. 지연을 넘기지 않으면 지금까지의 시간을
        # 잰다.
        def finish(outcome: str, status: int, latency: float | None = None) -> None:
            line.update(
                outcome=outcome,
                status=status,
                latency_s=latency if latency is not None else round(time.perf_counter() - t0, 3),
                prompt_tokens=_token_sum(prompt_tokens),
                completion_tokens=_token_sum(completion_tokens),
            )
            log.info(json.dumps(line, ensure_ascii=False))

        # 업종은 OpenAPI에만 enum으로 적고 검사는 여기서 한다. 그래야 목록 밖 업종도 처리기에 닿아 "unknown
        # domain" 422와 로그 한 줄이 남는다(SummarizeRequest의 기존 주석, 커밋 a13fcb6).
        if req.domain not in DOMAINS:
            finish("unknown_domain", 422)
            raise HTTPException(422, "unknown domain")
        # 발화를 학습·평가와 같은 "화자: 말" 줄 형식으로 잇는다. 길이 상한은 이 조립한 전사 기준이다.
        transcript = format_transcript([t.model_dump() for t in req.turns])
        line["chars"] = len(transcript)
        if len(transcript) > MAX_CHARS:
            finish("too_long", 413)
            raise HTTPException(413, f"transcript too long (max {MAX_CHARS} characters)")
        # 학습 때와 같은 p1 프롬프트, 예시 없이(0-shot). 학습한 모델이 이 조건으로 학습·측정됐다.
        messages = build_messages(req.domain, transcript)
        problems: list[str] = []
        for attempt in (1, 2):
            # 첫 시도는 탐욕 디코딩, 두 번째는 샘플링 provider. 스키마는 늘 넘기지만 OllamaProvider는
            # use_schema가 켜졌을 때만 Ollama에 보낸다.
            gen: Provider = p if attempt == 1 else state["retry_provider"]
            line["attempts"] = attempt
            try:
                reply = gen.generate(messages, json_schema=record_json_schema())
            except Exception as exc:  # noqa: BLE001 - 모델 서버 실패는 무엇이든 추적 정보 대신 503
                # 모델 서버 연결 실패·시간 초과 등 어떤 예외든 내부 추적 정보를 내보내지 않고 503으로 바꾼다.
                # 재시도하지 않는다.
                stats.add("model_unavailable")
                finish("model_unavailable", 503)
                raise HTTPException(503, "model server unavailable") from exc
            prompt_tokens.append(reply.prompt_tokens)
            completion_tokens.append(reply.completion_tokens)
            # 답에서 JSON을 꺼내 기록 형식으로 검증한다. 실패하면 이유를 남기고 다음 시도로.
            parsed = parse_reply(reply.text)
            if parsed.record is None:
                problems = [parsed.error or "invalid"]
                continue
            # 형식은 맞아도 업종 목록 밖 라벨을 쓰면 다시 시도한다.
            problems = off_list_labels(req.domain, parsed.record)
            if not problems:
                # 통과한 기록의 번호(주문번호, 운송장번호 등)를 형식과 전사 포함으로 검사한다. 서비스는 전사가
                # 음성 인식 결과인지 모르므로 verify는 늘 넓은 규칙(spoken=True)을 쓴다. 로그에는 표시한
                # 개수만 남긴다.
                flags = verify_entities(req.domain, parsed.record, transcript)
                line["needs_confirmation"] = len(flags)
                latency = round(time.perf_counter() - t0, 3)
                outcome = "ok" if attempt == 1 else "retried_ok"
                stats.add(outcome, latency)
                finish(outcome, 200, latency)
                return SummarizeResponse(
                    request_id=request_id,
                    record=parsed.record,
                    needs_confirmation=flags,
                    model=p.name,
                    attempts=attempt,
                    latency_s=latency,
                )
        # 두 번 모두 형식·라벨 검사에 실패: 반쯤 맞는 기록을 주지 않고 마지막 시도의 문제 목록과 함께 502를
        # 준다.
        stats.add("rejected")
        finish("rejected", 502)
        raise HTTPException(502, {"error": "model output rejected", "problems": problems})

    return app


# 모델 없이 늘 같은 기록을 주는 가짜 provider(오프라인 모드). 시스템 프롬프트에 들어 있는 회사 이름으로 업종을
# 알아내 그 업종의 첫 유형과 첫 번호 유형을 쓴다. 번호 값은 Random(0)으로 만들어 늘 같다.
# 이 번호는 스캔 요청의 전사에 없으므로 매번 needs_confirmation에 표시된다. 그래서 보안 스캔의 요청도 번호
# 검사 코드를 지난다(docs/experiments.md "확인이 필요한 번호 표시" 규칙).
class CannedProvider:
    """모델 없음: 시스템 프롬프트가 가리키는 업종에 맞는 유효한 기록을 돌려준다. 스캔과 시연용.

    기록에 그 업종의 고정 번호 하나가 들어 있어, 모든 요청이 자기 전사로 번호 검사도 거친다(호출한 쪽이
    그 번호를 말하지 않았다면 needs_confirmation에 나열된다)."""

    name = "offline:canned"

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        system = messages[0]["content"]
        domain = next((d for d in DOMAINS.values() if d.company in system), DOMAINS["shop"])
        id_type = next(et for et in domain.entity_types if et.kind == "id")
        record = AfterCallRecord(
            category=domain.categories[0],
            resolution="해결",
            entities=[Entity(type=id_type.label, value=id_type.make(random.Random(0)))],
            actions_taken=[],
            follow_up=FollowUp(required=False, codes=[]),
            summary="오프라인 모드의 고정 응답입니다.",
        )
        return Reply(text=record_to_json(record), latency_s=0.0)


# 보안 스캔(HawkScan)과 시연용 앱: CannedProvider + OpenAPI 공개. 스캐너가 OpenAPI로 경로와 요청 꼴을
# 알아낸다(docs/experiments.md "보안 스캔 기록").
def create_offline_app() -> FastAPI:
    return create_app(provider=CannedProvider(), expose_openapi=True)
