"""HTTP 서비스: POST /summarize가 상담 전사를 상담 후처리 기록으로 바꾼다.

모델 답은 AfterCallRecord로 검증을 통과하고 업종의 라벨 목록 안의 값만 써야 한다. 그렇지 않으면 서비스는
샘플링으로 한 번 다시 시도하고(탐욕 디코딩으로 다시 하면 거절한 답이 그대로 나온다), 그래도 안 되면 이유와
함께 502로 답한다. 반쯤 맞는 기록은 절대 내보내지 않는다. 모델 서버에 닿지 못하면 503을 준다.

입력 경계: 발화 글 안의 줄바꿈류(파이썬 str.splitlines가 줄 경계로 보는 문자 전부)와 탭은 전사를 조립하기
전에 공백 하나로 바꾼다. 고객 발화에 줄바꿈과 "상담원: ..."을 넣어 상담원 줄처럼 꾸미지 못하게 하려는
것이다. 그 밖의 제어문자가 든 발화는 422로 받지 않는다. 컨텍스트: 모델 서버가 컨텍스트 초과 오류를 주면
413(context exceeded), 답의 프롬프트 + 생성 토큰이 컨텍스트(4,096)를 다 채웠으면 입력이 잘렸을 수 있으므로
다시 묻지 않고 502(context exhausted)를 준다.

부하 경계: 모델 호출 구간에 동시에 들어가는 요청은 CALL_SUMMARY_MAX_INFLIGHT(기본 4)개까지다. 자리가 없으면
CALL_SUMMARY_MAX_QUEUE(기본 16)개까지만 줄을 서서 CALL_SUMMARY_QUEUE_WAIT_S(기본 30초) 동안 기다리고, 줄이 다
찼거나 그 시간이 지나면 모델을 부르지 않고 503(server busy)과 Retry-After를 준다. 모델 서버 호출 하나의 시간
한도는 CALL_SUMMARY_MODEL_TIMEOUT(기본 120초)이다. 요청 본문은 1 MiB까지만 받는다(넘으면 본문을 읽어 파싱하기
전에 413). /stats의 지연 분포는 최근 CALL_SUMMARY_STATS_WINDOW(기본 10,000)건으로 낸다.

기록 안의 번호(주문번호, 운송장번호, 접수번호, 카드 끝자리) 가운데 운영 형식에 맞지 않거나 전사에서 찾을
수 없는 것은 `needs_confirmation`에 나열한다(verify.py). 기록 자체는 바꾸지 않는다.

처리기에 닿은 /summarize 호출마다 JSON 한 줄을 로그로 남긴다(로거 `call_summary.service`, INFO):
요청 id, 업종, 발화 수와 글자 수, 정규화로 바뀐 발화 수와 화자 표시 꼴이 든 발화 수, 결과, HTTP 상태, 시도
횟수, 지연, 모델 호출 자리를 기다린 시간(queue_s), 시도들을 합친 프롬프트·생성 토큰 수(모델 서버가 알려 주지
않으면 null), num_predict에서 멈춘 답이 있었는지(length_stop)와 컨텍스트를 다 채운 답이
있었는지(context_full), 확인이 필요한 번호의 개수. 전사, 기록, 표시한 번호 값은 절대 남기지 않는다:
개인정보가 들어 있다.

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
# 남기지 않는다(개인정보). 발화 글의 줄바꿈류는 공백으로 바꿔 가짜 화자 줄을 막고 그 밖의 제어문자는 거절한다.
# 모델 없는 오프라인 모드(create_offline_app)는 HawkScan 보안 스캔용이다(docs/experiments.md "보안 스캔
# 기록").
# [부하 경계] 모델 호출은 동시 상한(CapacityLimiter)과 짧은 대기열 뒤에서만 돌고 넘치면 503 busy다. Ollama
# 호출의 시간 한도는 서비스용 값(기본 120초)이고, 요청 본문은 1 MiB 상한의 ASGI 미들웨어(BodySizeLimit)가
# 파싱 전에 자른다. /stats의 지연은 고정 크기 창(deque)으로 모아 메모리가 자라지 않는다.
# [연결] service_eval.py·loadtest.py가 이 서비스에 요청을 보내 품질과 부하를 재고, Dockerfile·compose.yaml이
# uvicorn으로 create_app을 띄운다. scripts/ctx_probe.py는 이 파일의 컨텍스트 판정 함수(context_flags,
# is_context_exceeded)로 실제 모델 서버 응답을 분류한다.

from __future__ import annotations

# 표준 라이브러리: dataclasses는 재시도용 provider 복사(replace)와 한도 설정(Limits)에, threading.Lock은
# /stats 집계를 동시 요청에서 지키는 데, uuid는 요청 id에, time.perf_counter는 지연과 대기 시간에 쓴다.
# random은 오프라인 모드의 고정 번호를 만드는 데만 쓴다. re는 발화 글의 줄바꿈류 정규화와 화자 표시 꼴 세기에,
# math는 한도 값 검사(유한한 수인지)와 Retry-After 올림에, collections.deque는 /stats의 고정 크기 지연 창에
# 쓴다.
import dataclasses
import json
import logging
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from collections import deque
from typing import Literal

# anyio: 입장 제어(CapacityLimiter, move_on_after)와 모델 호출 구간을 스레드로 보내는 to_thread.run_sync.
# Starlette의 필수 의존성이라 FastAPI와 함께 설치되고, FastAPI가 일반 def 처리기를 돌리는 스레드 풀도 같은
# anyio 것이다.
import anyio

# httpx: 모델 서버의 HTTP 오류(httpx.HTTPStatusError, OllamaProvider의 raise_for_status가 낸다)가 컨텍스트
# 초과 오류인지 가린다. 패키지의 기본 의존성이다(pyproject.toml).
import httpx

# FastAPI: 요청 본문을 pydantic 모델로 검증하고(형식이 틀리면 처리기 전에 422), OpenAPI 설명을 만든다.
# JSONResponse는 본문 상한 미들웨어가 FastAPI 오류 응답과 같은 꼴({"detail": ...})로 413을 직접 보내는 데
# 쓰고, starlette.types는 그 미들웨어의 ASGI 서명(scope·receive·send) 타입이다.
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.types import ASGIApp, Receive, Scope, Send

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
# fdc1b9c). 넘으면 413을 준다(README 서비스 문단). 데이터셋 전사는 가장 긴 것이 1,485자다. 이 상한은 한국어
# 대화 전사 기준이다. 프롬프트 토큰 수 ~= 571 + 0.605 * 전사 글자 수(dev 실행, Qwen3·qwen2.5 토크나이저로 잰
# 값)라 4,000자면 약 3,000토큰이고, 데이터셋에서 글자당 토큰이 가장 많은 전사(0.724)로 어림해도 약
# 3,460토큰이라 답(dev 정답 최대 271토큰)이 들어간다. 그러나 숫자·영문·이모지는 글자당 약 1토큰 이상이라
# 4,000자 안에서도 프롬프트만으로 4,096토큰을 넘을 수 있다(Qwen3 토크나이저로 센 인위적 입력: 띄어 쓴 숫자
# 4,175, 이모지 4,491토큰). 그래서 글자 상한만 믿지 않고 답의 토큰 합(context_full → 502)과 모델 서버의
# 컨텍스트 초과 오류(context_exceeded → 413)를 따로 본다(docs/experiments.md "서비스 컨텍스트 초과 동작
# 확인"). 글자 종류별 가중치로 토큰을 미리 어림하는 검사는 넣지 않았다: 간단한 가중치로 시험한 인위적 입력 8종
# 모두를 실제보다 작게 어림했고, 정확히 세려면 서비스에 토크나이저를 넣어야 한다.
MAX_CHARS = 4_000
# 요청 본문의 상한(바이트, 1 MiB). 발화 수·글자 수의 pydantic 상한과 전사 4,000자 상한은 본문을 다 읽고 파싱한
# 뒤에야 걸리므로, 그 전에 BodySizeLimit 미들웨어가 Content-Length(없으면 받은 바이트 수)로 자른다. 4,000자
# 상한을 통과할 수 있는 요청은 한글을 JSON 유니코드 이스케이프(글자당 여섯 바이트)로 적어도 약 30 KB라 1 MiB는
# 정상 요청에 닿지 않고, 그보다 큰 본문은 어차피 413(too_long)이나 422가 될 것을 메모리에 다 올리지 않게 하는
# 것이다.
MAX_BODY_BYTES = 1 << 20
# 발화 글의 줄 경계 정규화(가짜 화자 줄 막기). 전사는 발화마다 "화자: 말" 줄을 만들어 줄바꿈으로 잇기 때문에
# (prompts.format_transcript), 고객 발화 안의 줄바꿈 뒤에 "상담원: ..."을 적으면 모델이 받는 전사에 진짜
# 상담원 줄과 글자 꼴이 같은 줄이 생긴다(구조적 프롬프트 주입). 그래서 전사를 조립하기 전에 파이썬
# str.splitlines가 줄 경계로 보는 문자 전부(LF, CRLF(한 경계), CR, VT, FF, U+001C~U+001E, NEL(U+0085), U+2028,
# U+2029)와 탭을 하나마다 공백 하나로 바꾼다. str.splitlines를 기준으로 삼은 것은 이 목록이 LF만 줄로 보는
# 쪽(모델의 채팅 템플릿)보다 넓기 때문이다. 거절하지 않고 바꾸는 이유: 실제 음성 인식·채팅 연동은 한 발화의
# 조각을 줄바꿈으로 이어 보낼 수 있다. 이웃한 공백을 합치거나 앞뒤를 자르지는 않는다(바꾼 자리만 바뀌고 빈
# 발화는 빈 그대로). 커밋된 데이터셋 발화에는 이 문자들이 없어 기록된 프롬프트는 바뀌지
# 않는다(tests/test_service.py가 확인). 한 줄 안에 적은 "… 상담원: …" 같은 내용 수준 주입은 이 정규화로 막지
# 못한다(아래 _SPEAKER_MARKER로 센다).
_LINE_BREAKS = re.compile(r"\r\n|[\n\r\x0b\x0c\x1c-\x1e\x85\u2028\u2029\t]")
# 발화 글에 받지 않는 문자: 위 정규화 대상(줄바꿈류와 탭)을 뺀 나머지 C0·C1 제어문자(U+0000~U+0008,
# U+000E~U+001B, U+001F, U+007F~U+0084, U+0086~U+009F). 사람이 쓴 대화나 음성 인식 결과에 나올 일이 없고
# NUL·ESC 같은 문자는 로그·터미널·다른 시스템에서 달리 해석될 수 있어 422로 거절한다. Turn.text의 pattern이라
# 본문 검사에서 걸리고(처리기 전이라 로그 줄이 없다, README 서비스 문단) OpenAPI에도 적힌다. pydantic-core는
# 이 패턴을 Rust regex로 검사하므로 문자 범위를 16진 이스케이프로 적었다(^·$는 문자열 전체의 처음과 끝이다).
TEXT_PATTERN = r"^[^\x00-\x08\x0e-\x1b\x1f\x7f-\x84\x86-\x9f]*$"
# 화자 표시 꼴: "상담원:"·"고객:"(콜론 앞의 공백과 전각 콜론 "："도 포함). 정규화한 발화 글에서 찾아 그런 발화
# 수만 로그(speaker_marker_turns)에 남기고 거절하지는 않는다: 고객이 실제로 그렇게 말할 수도 있고, 목적은 운영
# 중 내용 수준 주입 시도가 얼마나 들어오는지 보는 것이다. 커밋된 데이터셋 발화에는 이 꼴이 하나도 없다.
_SPEAKER_MARKER = re.compile(r"(상담원|고객)\s*[:：]")
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


# 시도별 참·거짓 표시(length_stop, context_full)를 한 요청의 값으로 합친다. 아는 시도 가운데 하나라도 참이면
# 참, 아는 시도가 모두 거짓이면 거짓, 아는 시도가 없으면 None(모름)이다(_token_sum처럼 모름과 거짓을 가른다).
def _any_known(values: list[bool | None]) -> bool | None:
    known = [v for v in values if v is not None]
    return any(known) if known else None


# 발화 글 하나의 줄바꿈류와 탭을 공백 하나씩으로 바꾼다(_LINE_BREAKS의 주석). 입력: 발화 글. 출력: 바꾼
# 글(바꿀 것이 없으면 같은 글). /summarize 처리기가 전사를 조립하기 전에 모든 발화에 부르고, 테스트가 데이터셋
# 발화가 바뀌지 않는지 확인하는 데 쓴다.
def normalize_turn_text(text: str) -> str:
    return _LINE_BREAKS.sub(" ", text)


# 모델 답 하나가 컨텍스트 조건에 걸렸는지 본다. 입력: Reply와 그 요청의 num_ctx. 출력: (length_stop,
# context_full).
# - length_stop: done_reason이 "length"(num_predict에서 멈춤)인지. 모델 서버가 멈춘 이유를 알려 주지 않으면
#   None.
# - context_full: 프롬프트 + 생성 토큰이 num_ctx 이상인지(evaluate.run_checks와 같은 조건). 둘 중 하나라도
#   모르면 None. context_full이면 넘친 프롬프트가 잘려 들어갔거나(그러면 prompt_eval_count가 num_ctx 가까이
#   나온다) 생성 중에 컨텍스트를 밀어냈을 수 있어 답을 믿지 않는다. 다만 모델 서버가 넘침을 잘라서 처리하는지
#   오류로 돌려주는지에 따라 이 검사가 볼 수 있는 범위가 다르다: 오류로 오면 is_context_exceeded가 맡는다.
#   Ollama 0.35.1이 어느 쪽인지는 docs/experiments.md "서비스 컨텍스트 초과 동작 확인"에서 실제 서버로
#   확인한다(scripts/ctx_probe.py도 이 함수로 응답을 분류한다).
def context_flags(reply: Reply, num_ctx: int) -> tuple[bool | None, bool | None]:
    length_stop = None if reply.done_reason is None else reply.done_reason == "length"
    if reply.prompt_tokens is None or reply.completion_tokens is None:
        return length_stop, None
    return length_stop, reply.prompt_tokens + reply.completion_tokens >= num_ctx


# 모델 서버의 예외가 컨텍스트 초과 오류인지 본다: httpx.HTTPStatusError이고 응답 본문에 "context"와
# "exceed"(exceeds·exceeded 포함)가 함께 있으면(대소문자 무시) 그렇다고 본다. llama.cpp 서버의 오류 문구
# 꼴("... exceeds the available context size ...")과 "... exceeds the context length" 같은 꼴을 함께 덮으려고
# 낱말 둘로 짝을 맞췄다. 실제 Ollama 0.35.1의 응답을 보기 전에 넣은 방어적 짝 맞추기라, docs/experiments.md
# "서비스 컨텍스트 초과 동작 확인"의 결과에 맞춰 고칠 수 있다. 그 밖의 예외(연결 실패, 시간 초과)와 다른 HTTP
# 오류는 False다(처리기가 503으로 답한다).
def is_context_exceeded(exc: BaseException) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    try:
        body = exc.response.text.lower()
    except Exception:  # noqa: BLE001 - 본문을 읽지 못하면 모르는 오류로 친다(503)
        return False
    return "context" in body and "exceed" in body


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


# 발화 하나. speaker는 정확히 "상담원"·"고객"만 받는다. text는 2,000자 상한이고 줄바꿈류·탭을 뺀 제어문자는
# 받지 않는다(TEXT_PATTERN). 줄바꿈류·탭은 여기서 거절하지 않고 처리기가 공백으로 바꾼다(normalize_turn_text):
# pydantic 검사기에서 바꾸면 처리기가 바꾼 발화 수를 로그에 남길 수 없다.
class Turn(BaseModel):
    # 정규식이 아니라 Literal로 둔다: OpenAPI의 enum으로 클라이언트와 스캐너가 어떤 값이 통과하는지 안다.
    speaker: Literal["상담원", "고객"]
    # 빈 글도 받는다: 음성 인식기는 어떤 발화에서 아무것도 돌려주지 않는다(test-d에 그런 발화가 있다).
    text: str = Field(max_length=2000, pattern=TEXT_PATTERN)


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


# 환경 변수 하나를 정수 한도로 읽는다. 비어 있거나 없으면 default다(compose의 `${VAR:-}` 꼴이 빈 문자열을
# 넘기므로 빈 값은 "설정하지 않음"으로 본다. CALL_SUMMARY_NUM_GPU와 같은 처리). 정수가 아니거나(소수 포함)
# minimum보다 작으면 변수 이름이 든 ValueError로 앱 만들기를 멈춘다: 조용히 기본값으로 돌면 운영자가 정한
# 한도가 걸렸다고 잘못 믿게 된다.
def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {raw!r}")
    return value


# 환경 변수 하나를 초 단위 실수 한도로 읽는다(_env_int와 같은 규칙). inf·nan은 받지 않고, exclusive면
# minimum과 같은 값도 받지 않는다(시간 한도 0은 뜻이 없다).
def _env_float(name: str, default: float, minimum: float, exclusive: bool = False) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    bound = f"{'>' if exclusive else '>='} {minimum:g}"
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a finite number {bound}, got {raw!r}") from None
    if not math.isfinite(value) or value < minimum or (exclusive and value == minimum):
        raise ValueError(f"{name} must be a finite number {bound}, got {raw!r}")
    return value


# 서비스의 부하 한도. 모두 환경 변수로 바꿀 수 있고 기본값은 아래와 같다(README 서비스 문단).
# - max_inflight(CALL_SUMMARY_MAX_INFLIGHT, 4): 모델 호출 구간에 동시에 들어가는 요청 수. 기록된 부하 시험의
#   동시 1·2·4(docs/experiments.md 5단계 부하 시험)가 한도에 걸리지 않게 4다. Ollama는 요청을 하나씩
#   처리하므로 이 값을 올려도 처리량은 늘지 않고 한 요청의 지연만 는다(기록: 동시 1·2·4에서 처리량이 평평하고
#   p50이 동시 수에 비례).
# - max_queue(CALL_SUMMARY_MAX_QUEUE, 16): 자리를 기다릴 수 있는 요청 수. 한 건이 약 2.5초라 16건이면 약 40초
#   안에 차례가 온다.
# - queue_wait_s(CALL_SUMMARY_QUEUE_WAIT_S, 30초): 자리를 기다리는 시간의 상한. 넘으면 503 busy. 0이면 자리가
#   없을 때 바로 503이다.
# - model_timeout_s(CALL_SUMMARY_MODEL_TIMEOUT, 120초): 모델 서버 HTTP 호출 하나의 시간 한도. providers의
#   기본 600초는 14B 교사·판정 모델용이다. 서빙 모델의 기록은 부하 시험 /stats 최대 36.34초(첫 요청), 효율
#   측정의 첫 요청(모델 적재 포함) 52.9초라 120초면 콜드 스타트도 들어간다.
# - stats_window(CALL_SUMMARY_STATS_WINDOW, 10,000): /stats 지연 분포에 쓰는 최근 요청 수.
@dataclasses.dataclass(frozen=True)
class Limits:
    max_inflight: int = 4
    max_queue: int = 16
    queue_wait_s: float = 30.0
    model_timeout_s: float = 120.0
    stats_window: int = 10_000

    # 환경 변수에서 읽는다. 잘못된 값은 변수 이름이 든 ValueError(create_app이 그대로 올려 앱이 뜨지 않는다).
    @classmethod
    def from_env(cls) -> Limits:
        return cls(
            max_inflight=_env_int("CALL_SUMMARY_MAX_INFLIGHT", cls.max_inflight, minimum=1),
            max_queue=_env_int("CALL_SUMMARY_MAX_QUEUE", cls.max_queue, minimum=0),
            queue_wait_s=_env_float("CALL_SUMMARY_QUEUE_WAIT_S", cls.queue_wait_s, minimum=0.0),
            model_timeout_s=_env_float(
                "CALL_SUMMARY_MODEL_TIMEOUT", cls.model_timeout_s, minimum=0.0, exclusive=True
            ),
            stats_window=_env_int("CALL_SUMMARY_STATS_WINDOW", cls.stats_window, minimum=1),
        )

    # 503 busy 응답의 Retry-After(초, 정수 문자열). 대기 상한을 올림한 값이고 최소 1초다: 지금 자리가 없으니
    # 그만큼 뒤에 다시 보내라는 뜻이다.
    @property
    def retry_after(self) -> str:
        return str(max(1, math.ceil(self.queue_wait_s)))


# 이 프로세스가 받은 /summarize의 결과별 수와 최근 지연(GET /stats). 메모리에만 있어 재시작하면 0부터 다시
# 센다. add는 모델 호출 구간을 돌리는 작업 스레드와 이벤트 루프(503 busy) 양쪽에서 불리므로
# threading.Lock으로 읽기·쓰기를 지킨다. 결과: ok·retried_ok(기록을 줌), rejected(두 번 모두 형식·라벨 실패,
# 502), model_unavailable(모델 서버 오류, 503), context_full(답이 컨텍스트를 다 채움, 502),
# context_exceeded(모델 서버의 컨텍스트 초과 오류, 413), busy(입장 제어에서 거절, 503). 모델에 닿기 전에 끝난
# unknown_domain·too_long은 로그에만 남고 여기서는 세지 않는다(전과 같음).
# 지연은 길이가 window인 deque(기본 10,000건 = CALL_SUMMARY_STATS_WINDOW)에 모아, 오래 돌아도 메모리가 자라지
# 않고 p50·p95가 최근 창의 값이 된다. 기록된 부하 시험의 /stats(725건)는 창보다 작아 같은 값이다. 결과별 수는
# 창과 상관없이 전체를 센다.
class Stats:
    """이 프로세스가 받은 /summarize 호출의 결과별 수와 최근 window건의 지연(재시작하면 0부터 센다)."""

    def __init__(self, window: int = Limits.stats_window) -> None:
        self._lock = threading.Lock()
        self.window = window
        self.latencies: deque[float] = deque(maxlen=window)
        self.counts = {
            "ok": 0,
            "retried_ok": 0,
            "rejected": 0,
            "model_unavailable": 0,
            "context_full": 0,
            "context_exceeded": 0,
            "busy": 0,
        }

    # 결과 하나를 센다. 지연은 기록이 나간 요청(ok, retried_ok)만 넘겨받는다.
    def add(self, outcome: str, latency: float | None = None) -> None:
        with self._lock:
            self.counts[outcome] += 1
            if latency is not None:
                self.latencies.append(latency)

    # 현재 수와 최근 창의 지연 분포(p50, p95, 최대)를 돌려준다. Lock 안에서는 결과별 수와 지연 목록을 함께
    # 복사만 하고(두 값이 같은 순간의 것이 되도록), 정렬과 백분위 계산은 밖에서 한다. p50·p95는
    # scoring.pctl(evaluate·loadtest와 같은 계산, 보간 없음)이다. window도 적어 창 기반 값임을 밝힌다.
    def snapshot(self) -> dict:
        with self._lock:
            lat = list(self.latencies)
            counts = dict(self.counts)
        lat.sort()
        out: dict = {"requests": counts, "window": self.window, "latency_s": None}
        if lat:
            out["latency_s"] = {
                "n": len(lat),
                "p50": round(pctl(lat, 0.5), 3),
                "p95": round(pctl(lat, 0.95), 3),
                "max": round(lat[-1], 3),
            }
        return out


# 요청 본문 상한의 ASGI 미들웨어. Content-Length가 max_bytes를 넘으면 본문을 한 조각도 읽지 않고 413을
# 보내고, Content-Length가 없거나(chunked) 숫자가 아니면 받은 바이트를 세어 상한을 넘는 조각에서 멈추고
# 413을 보낸다(길이를 알렸더라도 실제 바이트로 다시 센다). 상한 안이면 받은 조각을 이어 한 http.request
# 메시지로 안쪽 앱에 넘기고, 그 뒤의 receive는 원래 것(연결 끊김 알림)으로 이어 준다. 그래서 FastAPI가 본문을
# 파싱하기 전에 걸리고, 이 413은 처리기에 닿지 않아 로그 줄이 없다(본문 검사 422와 같다). 응답 꼴은 FastAPI
# 오류와 같은 {"detail": "..."}이고 상한 값을 적는다. 보안 헤더 미들웨어(security_headers) 안쪽에 두어 이
# 413에도 보안 헤더가 붙는다(create_app의 등록 순서).
# Starlette 1.7에도 본문 상한(Starlette(max_body_size=...)의 RequestBodyLimitMiddleware)이 있지만 FastAPI
# 0.141은 그 인자를 넘기지 않고, 그 413은 평문 "Content Too Large"이며 사용자 미들웨어 바깥에 놓여 보안 헤더가
# 빠지므로 쓰지 않았다. BaseHTTPMiddleware 대신 순수 ASGI로 쓴 것은 본문을 읽기 전에 답하고 조각 단위로 세야
# 하기 때문이다.
class BodySizeLimit:
    """요청 본문이 max_bytes를 넘으면 안쪽 앱을 부르지 않고 413을 돌려준다."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._too_large(scope, receive, send)
            return
        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # 본문이 끝나기 전에 연결이 끊긴 것(http.disconnect). 안쪽 앱에 그 메시지를 그대로 넘긴다.
                first = message
                break
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_bytes:
                await self._too_large(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                first = {"type": "http.request", "body": b"".join(chunks), "more_body": False}
                break
        pending = [first]

        async def buffered_receive() -> dict:
            return pending.pop() if pending else await receive()

        await self.app(scope, buffered_receive, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        detail = f"request body too large (max {self.max_bytes} bytes)"
        await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)


# scope의 Content-Length 헤더 값(바이트 수). 없거나 숫자가 아니면 None(그러면 받은 바이트로 센다).
def _content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


# 앱 공장 함수(uvicorn --factory로 부른다). provider를 넘기면 그것을(테스트·오프라인 모드), 아니면 환경 변수로
# OllamaProvider를 만든다.
# 환경 변수: CALL_SUMMARY_MODEL(Ollama 모델 태그), OLLAMA_URL, CALL_SUMMARY_SCHEMA=1(스키마 강제 켜기),
# CALL_SUMMARY_NUM_GPU(GPU에 올릴 층 수. 14B처럼 VRAM이 넘치는 모델용, docs/experiments.md 1단계 진행 기록),
# CALL_SUMMARY_EXPOSE_OPENAPI=1(OpenAPI 공개), 부하 한도 다섯 개(Limits의 주석: CALL_SUMMARY_MAX_INFLIGHT,
# CALL_SUMMARY_MAX_QUEUE, CALL_SUMMARY_QUEUE_WAIT_S, CALL_SUMMARY_MODEL_TIMEOUT, CALL_SUMMARY_STATS_WINDOW.
# 잘못된 값이면 ValueError로 앱이 뜨지 않는다).
def create_app(provider: Provider | None = None, expose_openapi: bool | None = None) -> FastAPI:
    if expose_openapi is None:
        expose_openapi = os.environ.get("CALL_SUMMARY_EXPOSE_OPENAPI") == "1"
    # 부하 한도는 provider를 넘긴 경우(테스트·오프라인 모드)에도 모두 읽고 검사한다: 잘못된 설정은 어느 모드로
    # 띄우든 시작할 때 드러나야 한다.
    limits = Limits.from_env()
    # Swagger·ReDoc 문서 페이지는 끄고, OpenAPI JSON은 요청할 때만 연다(오프라인 모드는 스캔을 위해 연다).
    app = FastAPI(
        title="call-summary",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if expose_openapi else None,
    )
    stats = Stats(window=limits.stats_window)
    _ensure_log_handler()
    base: Provider = provider or OllamaProvider(
        model=os.environ.get("CALL_SUMMARY_MODEL", "qwen2.5:7b-instruct"),
        base_url=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
        # 기본은 끈다. Ollama의 스키마 문법은 필수 키를 알파벳 순서로 먼저 쓰게 하는데, 한 가지 키 순서로
        # 학습한 모델은 그러면 `actions_taken`을 빠뜨린다(docs/experiments.md 5단계).
        use_schema=os.environ.get("CALL_SUMMARY_SCHEMA") == "1",
        num_ctx=NUM_CTX,
        num_gpu=int(os.environ["CALL_SUMMARY_NUM_GPU"]) if os.environ.get("CALL_SUMMARY_NUM_GPU") else None,
        # 서비스용 시간 한도(Limits.model_timeout_s의 주석). providers의 기본값 600초는 14B 교사·판정용이다.
        timeout_s=limits.model_timeout_s,
    )
    # 재시도 provider도 앱을 만들 때 한 번만 만든다. 요청은 작업 스레드에서 동시에 돌므로, 요청마다 provider의
    # 설정을 바꾸면 동시에 도는 다른 요청이 바뀐 설정을 쓸 수 있다. 그래서 공유 provider는 절대 바꾸지 않는다.
    # dataclasses.replace로 만드는 재시도 provider는 timeout_s를 포함한 모든 필드를 물려받는다. waiting은 모델
    # 호출 자리를 기다리는 요청 수인데, 처리기(이벤트 루프)에서만 바꾸므로 잠금이 필요 없다.
    state: dict = {"provider": base, "retry_provider": _retry_provider(base), "waiting": 0}
    # 입장 제어의 동시 상한. 처리기가 모델 호출 구간에 들어갈 때 손으로 잡고(acquire) 끝나면 놓는다(release).
    # anyio.to_thread.run_sync의 limiter 인자로 같은 객체를 넘기지 않는다: 같은 태스크가 두 번 잡으려 해
    # RuntimeError가 난다. 작업 스레드는 anyio의 기본 limiter(40)로 돌리므로 모델 호출이 상한까지 차 있어도
    # /health 같은 일반 def 처리기의 스레드 자리는 남는다.
    limiter = anyio.CapacityLimiter(limits.max_inflight)

    # 요청 본문 상한(BodySizeLimit의 주석). 보안 헤더 미들웨어보다 먼저 등록한다: Starlette는 나중에 등록한
    # 미들웨어를 바깥에 두므로, 이 순서라야 본문 상한의 413에도 보안 헤더가 붙는다(테스트가 확인한다).
    app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

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

    # 본 기능. 단계: 로그 줄 준비 → 업종 확인(422) → 발화 글 정규화 → 전사 조립·길이 확인(413) → 프롬프트 조립
    # → 입장 제어(자리를 못 얻으면 503 busy) → 작업 스레드에서 최대 두 번 시도(모델 서버의 컨텍스트 초과 오류
    # 413, 그 밖의 모델 오류 503, 답이 컨텍스트를 다 채웠으면 502, 형식·라벨 검사 실패면 재시도) → 번호 확인
    # 표시 → 응답, 두 번 모두 실패하면 502. 처리기에 닿은 요청은 어느 결과로 끝나든 finish()로 로그 한 줄을
    # 남긴다. 본문 형식 검사(빠진 필드, 허용되지 않은 발화자, 발화 글의 제어문자)에서 걸린 422와 본문 상한의
    # 413은 처리기에 닿지 않아 줄이 없다(README 서비스 문단).
    # async def인 이유: 입장 제어의 대기는 이벤트 루프에서 하고(기다리는 요청이 스레드를 잡지 않는다), 모델
    # 호출처럼 막히는 구간만 anyio.to_thread.run_sync로 작업 스레드에 보낸다. 처리기 안에 동기 I/O를 두면
    # 이벤트 루프가 멈추므로 모델 호출은 반드시 스레드로 보낸다.
    @app.post("/summarize", response_model=SummarizeResponse)
    async def summarize(req: SummarizeRequest) -> SummarizeResponse:
        p: Provider = state["provider"]
        request_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        # 요청 로그 한 줄의 틀. 개수·결과·지연·토큰 수·표시만 담고 전사·기록·번호 값은 담지 않는다. 목록 밖
        # 업종은 받은 문자열을 그대로 남기지 않고 None으로 둔다. normalized_turns·speaker_marker_turns는
        # 전사를 조립할 때, queue_s(모델 호출 자리를 기다린 초)는 입장 제어를 지날 때,
        # length_stop·context_full은 모델 답을 받은 시도들로 채운다(그 전에 끝나면 None).
        line: dict = {
            "event": "summarize",
            "request_id": request_id,
            "domain": req.domain if req.domain in DOMAINS else None,
            "turns": len(req.turns),
            "chars": None,
            "normalized_turns": None,
            "speaker_marker_turns": None,
            "outcome": None,
            "status": None,
            "attempts": 0,
            "latency_s": None,
            "queue_s": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "length_stop": None,
            "context_full": None,
            "needs_confirmation": None,
            "model": p.name,
        }
        prompt_tokens: list[int | None] = []
        completion_tokens: list[int | None] = []
        length_stops: list[bool | None] = []
        context_fulls: list[bool | None] = []

        # 결과·상태·지연, 시도를 합친 토큰 수와 컨텍스트 표시를 채워 로그 한 줄을 쓴다. 지연을 넘기지 않으면
        # 지금까지의 시간을 잰다.
        def finish(outcome: str, status: int, latency: float | None = None) -> None:
            line.update(
                outcome=outcome,
                status=status,
                latency_s=latency if latency is not None else round(time.perf_counter() - t0, 3),
                prompt_tokens=_token_sum(prompt_tokens),
                completion_tokens=_token_sum(completion_tokens),
                length_stop=_any_known(length_stops),
                context_full=_any_known(context_fulls),
            )
            log.info(json.dumps(line, ensure_ascii=False))

        # 업종은 OpenAPI에만 enum으로 적고 검사는 여기서 한다. 그래야 목록 밖 업종도 처리기에 닿아 "unknown
        # domain" 422와 로그 한 줄이 남는다(SummarizeRequest의 기존 주석, 커밋 a13fcb6).
        if req.domain not in DOMAINS:
            finish("unknown_domain", 422)
            raise HTTPException(422, "unknown domain")
        # 발화 글의 줄바꿈류·탭을 공백으로 바꾼 뒤(가짜 화자 줄 막기, _LINE_BREAKS의 주석) 학습·평가와 같은
        # "화자: 말" 줄 형식으로 잇는다. 로그에는 바꾼 발화 수와 화자 표시 꼴이 든 발화 수만 남긴다. 길이
        # 상한은 이 조립한 전사 기준이다.
        turns = [t.model_dump() for t in req.turns]
        normalized = 0
        for t in turns:
            text = normalize_turn_text(t["text"])
            if text != t["text"]:
                normalized += 1
                t["text"] = text
        line["normalized_turns"] = normalized
        line["speaker_marker_turns"] = sum(1 for t in turns if _SPEAKER_MARKER.search(t["text"]))
        transcript = format_transcript(turns)
        line["chars"] = len(transcript)
        if len(transcript) > MAX_CHARS:
            finish("too_long", 413)
            raise HTTPException(413, f"transcript too long (max {MAX_CHARS} characters)")
        # 학습 때와 같은 p1 프롬프트, 예시 없이(0-shot). 학습한 모델이 이 조건으로 학습·측정됐다.
        messages = build_messages(req.domain, transcript)

        # 모델 호출 구간(두 번 시도, 번호 확인). 아래 입장 제어를 통과한 뒤 작업 스레드에서 돈다: 모델 서버
        # 호출은 동기 httpx라 이벤트 루프에서 직접 부르면 다른 요청과 /health까지 멈춘다. 여기서 낸
        # HTTPException은 run_sync를 지나 그대로 올라와 FastAPI가 응답으로 바꾼다.
        def run_model() -> SummarizeResponse:
            problems: list[str] = []
            for attempt in (1, 2):
                # 첫 시도는 탐욕 디코딩, 두 번째는 샘플링 provider. 스키마는 늘 넘기지만 OllamaProvider는
                # use_schema가 켜졌을 때만 Ollama에 보낸다.
                gen: Provider = p if attempt == 1 else state["retry_provider"]
                line["attempts"] = attempt
                try:
                    reply = gen.generate(messages, json_schema=record_json_schema())
                except Exception as exc:  # noqa: BLE001 - 모델 서버 실패는 무엇이든 추적 정보 대신 413·503
                    # 모델 서버가 컨텍스트 초과 오류를 주면 서버가 아니라 입력 크기 문제이므로 413으로
                    # 답한다(같은 입력이라 재시도하지 않는다). 그 밖에 연결 실패·시간 초과(timeout_s) 등 어떤
                    # 예외든 내부 추적 정보와 오류 본문을 내보내지 않고 503으로 바꾼다. 재시도하지 않는다.
                    if is_context_exceeded(exc):
                        stats.add("context_exceeded")
                        finish("context_exceeded", 413)
                        raise HTTPException(413, {"error": "context exceeded"}) from exc
                    stats.add("model_unavailable")
                    finish("model_unavailable", 503)
                    raise HTTPException(503, "model server unavailable") from exc
                prompt_tokens.append(reply.prompt_tokens)
                completion_tokens.append(reply.completion_tokens)
                # 컨텍스트 검사(context_flags의 주석). 기준 num_ctx는 provider의 값(OllamaProvider)이고, 없는
                # provider(테스트·오프라인 모드)는 서비스의 NUM_CTX다. 답이 컨텍스트를 다 채웠으면 형식·라벨
                # 검사를 통과하더라도 잘린 입력에서 나온 답일 수 있어 내보내지 않는다. 같은 입력을 다시 물어도
                # 같은 길이라 재시도하지 않고 502로 끝낸다. num_predict에서 멈춘 답(length_stop)은 표시만
                # 한다: 대개 JSON이 잘려 아래 형식 검사에서 걸린다.
                length_stop, context_full = context_flags(reply, getattr(gen, "num_ctx", None) or NUM_CTX)
                length_stops.append(length_stop)
                context_fulls.append(context_full)
                if context_full:
                    stats.add("context_full")
                    finish("context_full", 502)
                    raise HTTPException(502, {"error": "context exhausted", "problems": ["context_full"]})
                # 답에서 JSON을 꺼내 기록 형식으로 검증한다. 실패하면 이유를 남기고 다음 시도로.
                parsed = parse_reply(reply.text)
                if parsed.record is None:
                    problems = [parsed.error or "invalid"]
                    continue
                # 형식은 맞아도 업종 목록 밖 라벨을 쓰면 다시 시도한다.
                problems = off_list_labels(req.domain, parsed.record)
                if not problems:
                    # 통과한 기록의 번호(주문번호, 운송장번호 등)를 형식과 전사 포함으로 검사한다. 서비스는
                    # 전사가 음성 인식 결과인지 모르므로 verify는 늘 넓은 규칙(spoken=True)을 쓴다. 로그에는
                    # 표시한 개수만 남긴다.
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
            # 두 번 모두 형식·라벨 검사에 실패: 반쯤 맞는 기록을 주지 않고 마지막 시도의 문제 목록과 함께
            # 502를 준다.
            stats.add("rejected")
            finish("rejected", 502)
            raise HTTPException(502, {"error": "model output rejected", "problems": problems})

        # 입장 제어(Limits의 주석). 자리가 비어 있으면 바로 들어가고, 없으면 기다리는 수가 max_queue 미만일
        # 때만 queue_wait_s 동안 줄을 선다. 줄이 다 찼거나 시간이 지나면 모델을 부르지 않고 503 busy +
        # Retry-After다. 기다리는 동안 다른 요청의 release가 토큰을 대기 순서대로 바로 넘겨 주고, 시간이 지나
        # 취소된 요청은 토큰을 쥐지 않는다(anyio의 CapacityLimiter가 취소 때 반납과 다음 대기자 알림까지
        # 처리한다). 그래서 잡았으면 반드시 finally에서 놓고, 잡지 못했으면 놓지 않는다. 잡기와 놓기는 같은
        # 태스크(이 처리기)에서 한다.
        t_queue = time.perf_counter()
        admitted = False
        try:
            limiter.acquire_nowait()
            admitted = True
        except anyio.WouldBlock:
            if state["waiting"] < limits.max_queue:
                state["waiting"] += 1
                try:
                    with anyio.move_on_after(limits.queue_wait_s) as scope:
                        await limiter.acquire()
                    admitted = not scope.cancelled_caught
                finally:
                    state["waiting"] -= 1
        line["queue_s"] = round(time.perf_counter() - t_queue, 3)
        if not admitted:
            stats.add("busy")
            finish("busy", 503)
            raise HTTPException(503, {"error": "server busy"}, headers={"Retry-After": limits.retry_after})
        try:
            return await anyio.to_thread.run_sync(run_model)
        finally:
            limiter.release()

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
