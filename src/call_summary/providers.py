"""하나의 인터페이스 뒤에 둔 추론 백엔드들: 채팅 메시지를 넣으면 원 답 글이 나온다.

- ScriptedProvider: 미리 정한 답을 돌려준다. 테스트용.
- OllamaProvider: HTTP로 부르는 로컬 Ollama(기준선, 교사, 판정 모델, GGUF로 서빙하는 학생).
- HFProvider: transformers + 선택적 PEFT 어댑터(GGUF로 내보내기 전의 학생). 무거운 import는 쓸 때 한다.
"""

# [파일 역할] 모델을 부르는 실행기(provider)를 한 인터페이스 뒤에 모은 모듈이다. 모두 "채팅 메시지 목록을 받아
# 원 답 글(Reply)을 돌려준다"는 같은 꼴이라, 평가·서비스·생성·판정 코드는 어떤 백엔드인지 몰라도 된다.
# [왜 필요한가] 이 프로젝트는 같은 프롬프트·같은 채점으로 서로 다른 실행 경로를 비교한다: Ollama의 qwen2.5
# 기준선, transformers로 돌린 학습 전후 학생(4bit + LoRA 어댑터), GGUF로 바꿔 Ollama에 올린 서빙 학생.
# 실행기만 바꿔 끼우면 나머지 측정 코드가 그대로이므로 "서빙해도 품질이 그대로인가" 같은 비교가 공정해진다.
# [설계 이유] torch·transformers·httpx는 함수 안에서 불러온다(지연 import). 그래서 GPU·모델 없이 도는 테스트와
# CI가 ScriptedProvider만으로 이 모듈을 쓸 수 있다(README "테스트는 GPU·모델 없이 돈다").
# [주요 구성] Reply(답과 측정값) / Provider(인터페이스) / ScriptedProvider(테스트용) / OllamaProvider(HTTP) /
# Ollama 서버 메타데이터 도우미(parse_keep_alive, ollama_model_digest, ollama_version, ollama_fingerprint,
# ollama_load: 측정 기록의 모델 지문과 서비스의 /health·/ready·예열이 함께 쓴다) / HF 재현 조건 도우미
# (BASE_REVISIONS·revision_for: 바탕 모델의 고정 revision, library_versions: 학습·추론 라이브러리 판) /
# HFProvider(transformers).

from __future__ import annotations

import importlib.metadata
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

# Ollama /api/chat 응답에 들어 있는 시간 카운터(나노초) 이름. 5단계 효율 추가 측정(bench)에서 모델
# 적재·프롬프트 처리·생성 시간을 나눠 보려고 Reply에 함께 담는다(커밋 0ee468b).
OLLAMA_TIMINGS = ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")


# 모델 답 하나와 그 측정값. text는 원 답 글, latency_s는 클라이언트에서 잰 시간, 토큰 수는 백엔드가 알려 줄
# 때만 채운다.
# thinking과 done_reason은 측정 조건이 지켜졌는지(생각 모드가 꺼졌는지, num_predict에서 잘리지 않았는지)
# 확인하는 데 쓴다(evaluate.run_checks).
@dataclass
class Reply:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Ollama가 직접 잰 시간 카운터(나노초, total·load·prompt_eval·eval duration). 다른 백엔드에서는 None.
    timings: dict | None = None
    # thinking: Ollama 응답의 message.thinking(모델 파서가 답에서 떼어 낸 생각 글).
    # done_reason: Ollama가 생성을 멈춘 이유. 정상 종료는 "stop", num_predict에서 잘리면 "length"다.
    thinking: str | None = None
    done_reason: str | None = None


# 실행기 인터페이스(구조적 타입). name 속성과 generate(messages, json_schema)만 있으면 된다. json_schema는
# 스키마 강제 디코딩을 지원하는 백엔드만 쓰고 나머지는 무시한다.
class Provider(Protocol):
    name: str

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply: ...


# 테스트용 실행기. 미리 정한 답 목록을 차례로 돌려주거나, 메시지를 받아 답을 만드는 함수를 쓴다. 받은 메시지를
# calls에 남겨 테스트가 프롬프트를 확인할 수 있다(예: tests/test_scoring.py가 few-shot 메시지를 비교).
@dataclass
class ScriptedProvider:
    """메시지를 받는 함수가 만든 답(또는 고정 목록의 답을 차례로)을 돌려준다."""

    replies: list[str] | Callable[[list[dict]], str]
    name: str = "scripted"
    calls: list[list[dict]] = field(default_factory=list)

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        self.calls.append(messages)
        if callable(self.replies):
            text = self.replies(messages)
        else:
            text = self.replies[len(self.calls) - 1]
        return Reply(text=text, latency_s=0.0)


# 로컬 Ollama 서버(/api/chat)를 부르는 실행기. 기준선 모델, 대화 생성 교사(gen.py), 요약 판정 모델(judge.py),
# GGUF로 서빙하는 학생(service.py, evaluate --backend ollama, bench.py)이 모두 이것을 쓴다.
# 기본 디코딩은 탐욕적(temperature 0)이고 seed를 고정한다(docs/experiments.md 공통 "모든 모델 탐욕적").
@dataclass
class OllamaProvider:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    num_ctx: int = 8192
    num_predict: int = 1024
    # GPU에 올릴 층 수. None이면 Ollama가 정하는데, Windows에서는 VRAM을 꽉 채워 시스템 메모리로 넘치면 요청이
    # 10분 넘게 걸린 적이 있어 14B는 44층으로 제한했다(docs/experiments.md 1단계 진행 기록).
    num_gpu: int | None = None
    seed: int = 0
    # keep_alive: 마지막 요청 뒤 모델을 메모리에 남겨 둘 시간. 측정 중 요청마다 모델을 다시 올리지 않게 한다.
    # Ollama가 받는 꼴 그대로 요청 본문에 들어간다: Go 시간 문자열("30m", "1h")이나 초 단위 정수(음수는 계속
    # 상주, 0은 바로 내림. bench.unload가 0을 보낸다). 기본 "30m"은 기록된 모든 실행과 같은 값이고, 서비스는
    # CALL_SUMMARY_KEEP_ALIVE로 바꿀 수 있다(parse_keep_alive가 글을 이 두 꼴로 가른다).
    # timeout_s: 요청 하나의 HTTP 시간 한도(초).
    keep_alive: str | int = "30m"
    timeout_s: float = 600.0
    # use_schema: JSON 스키마를 Ollama format으로 넘겨 강제 디코딩할지. 기본은 끈다. 학습한 모델은 스키마
    # 문법이 키 순서를 바꾸면 필드를 빠뜨렸다(5단계 발견 1). 판정 모델은 켠다(judge.make_judge).
    # think: 생각 모드 스위치. False를 보내 Qwen3 계열의 생각 글을 끈다(공정하고 빠른 기준선을 위해). None이면
    # 요청에 넣지 않는다.
    use_schema: bool = False
    think: bool | None = False
    # 그 밖의 Ollama 옵션(예: presence_penalty). 위의 고정 디코딩 설정은 따로 필드가 있으므로 여기서
    # 넣지 않는다.
    extra_options: dict | None = None
    # truncate·shift: 컨텍스트가 넘칠 때의 동작을 정하는 Ollama 요청의 맨 위 필드(options 밖)다. None이면
    # 요청에 넣지 않아 서버 기본값을 쓴다. 기본이 None이라 이 필드를 더하기 전의 요청(기록된 모든 실행)과
    # 본문이 바이트까지 같다(tests/test_ctx_probe.py가 키 목록을 고정해 확인). False를 보내면 truncate는 넘친
    # 프롬프트를 잘라 넣지 않고 오류를 돌려주게, shift는 생성 중에 컨텍스트가 차면 앞부분을 밀어내며 계속 쓰지
    # 않고 멈추게 하는 설정이다. Ollama 0.35.1에서 실제로 그렇게 되는지와 서비스 요청에 넣을지는
    # docs/experiments.md "서비스 컨텍스트 초과 동작 확인"에서 정한다(scripts/ctx_probe.py가 두 값을 보내
    # 본다). 서비스는 아직 보내지 않는다.
    truncate: bool | None = None
    shift: bool | None = None

    # 필드로 따로 둔 디코딩 설정. extra_options로 이 값들을 덮어쓰면 실행 기록의 필드 값과 실제 요청이
    # 어긋나므로 __post_init__에서 막는다.
    _FIXED = ("temperature", "num_ctx", "num_predict", "num_gpu", "seed")

    def __post_init__(self) -> None:
        clash = sorted(set(self.extra_options or {}) & set(self._FIXED))
        if clash:
            raise ValueError(f"extra_options cannot set {clash}; use the provider fields")

    # 실행 기록에 남는 이름. 스키마 강제를 켠 실행은 ":schema"를 붙여 같은 모델의 두 조건을 구분한다(2단계의
    # 7B 스키마 강제 참고치 등).
    @property
    def name(self) -> str:
        return f"ollama:{self.model}" + (":schema" if self.use_schema else "")

    # 요청 본문을 만들어 /api/chat에 한 번 보내고 Reply로 바꾼다. stream=False라 답 전체가 한 번에 온다.
    # 실패한 HTTP 상태는 raise_for_status로 예외가 되어 부른 쪽(재시도·오류 집계)이 처리한다.
    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        import httpx

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        # 선택 설정은 값이 있을 때만 넣는다. 넣지 않으면 서버·모델 태그의 기본값을 쓴다. extra_options는 고정
        # 설정 뒤에 합치지만 __post_init__이 겹침을 막아 두었다.
        if self.num_gpu is not None:
            body["options"]["num_gpu"] = self.num_gpu
        body["options"].update(self.extra_options or {})
        if self.think is not None:
            body["think"] = self.think
        # 컨텍스트 넘침 동작(위 필드 주석). 값이 있을 때만 맨 위에 넣는다.
        if self.truncate is not None:
            body["truncate"] = self.truncate
        if self.shift is not None:
            body["shift"] = self.shift
        if self.use_schema and json_schema is not None:
            body["format"] = json_schema
        # 지연은 HTTP 왕복 전체를 클라이언트에서 잰다(서버 쪽 시간은 timings에 따로 있다).
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return Reply(
            text=data["message"]["content"],
            latency_s=time.perf_counter() - t0,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            timings={k: data.get(k) for k in OLLAMA_TIMINGS},
            # 생각 글이 빈 문자열이면 None으로 바꿔 "생각 글 없음"을 한 가지 값으로 센다.
            thinking=data["message"].get("thinking") or None,
            done_reason=data.get("done_reason"),
        )

    # Ollama /api/ps: 지금 올라간 모델 목록(size_vram 대 size로 GPU에 다 올라갔는지, context_length 등).
    # evaluate.loaded_models를 거쳐 judge_run·judge_calib가 판정 뒤 기록에 남긴다.
    def loaded_models(self) -> list[dict]:
        import httpx

        r = httpx.get(f"{self.base_url}/api/ps", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])


# keep_alive 글이 Ollama가 받는 두 꼴 가운데 어느 것인지 가르는 정규식. Ollama 서버는 keep_alive를 JSON
# 숫자(초. 음수는 계속 상주)나 Go의 time.ParseDuration이 읽는 문자열("30m", "1h30m", "1.5h", "-1m", "500ms".
# 단위는 ns, us/µs, ms, s, m, h이고 부호는 맨 앞에 한 번)로 받는다. 문자열 "-1"은 단위가 없어 ParseDuration이
# 거절하므로, 정수 글은 숫자로 보내야 한다.
_KEEP_ALIVE_INT = re.compile(r"[-+]?\d+")
_KEEP_ALIVE_DURATION = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:ns|us|µs|μs|ms|s|m|h))+")


# 환경 변수 같은 글을 Ollama 요청의 keep_alive 값으로 바꾼다. 입력: 글(앞뒤 공백은 지운다). 출력: 정수 글이면
# int(JSON 숫자로 나간다: "-1" → -1 계속 상주, "0" → 0 바로 내림, "600" → 600초), Go 시간 문자열이면 그
# 문자열 그대로. 그 밖의 글("forever", 단위 없는 소수 "1.5", 단위 앞 공백 "10 m", 빈 글)은 ValueError다:
# Ollama가 400으로 거절해 모든 요청이 실패하는 설정을 서비스가 시작할 때 걸러 내려는 것이다(service의
# ModelSettings.from_env). 기본값 "30m"은 이 함수를 거치지 않는다.
def parse_keep_alive(raw: str) -> str | int:
    """Ollama의 keep_alive 값: 정수 글은 초 단위 숫자로(음수는 계속 상주), Go 시간 문자열은 그대로.
    둘 다 아니면 ValueError."""
    text = raw.strip()
    if _KEEP_ALIVE_INT.fullmatch(text):
        return int(text)
    if _KEEP_ALIVE_DURATION.fullmatch(text):
        return text
    raise ValueError(
        f"keep_alive must be a Go duration such as 30m or 1h30m, or an integer number of seconds "
        f"(negative keeps the model loaded), got {raw!r}"
    )


# Ollama /api/tags에서 model(또는 model:latest)의 digest(sha256 64자리)를 찾는다. 입력: 서버 주소, 모델 태그,
# 요청 시간 한도(초). 출력: digest. 목록에 그 태그가 없으면 None이고, 서버에 닿지 못하거나 답이 JSON이 아니면
# httpx 예외 등이 그대로 올라온다(부르는 쪽이 "서버 없음"과 "모델 없음"을 가르는 데 쓴다: 서비스의 /ready).
# 측정 기록의 model_fingerprint(evaluate)가 쓰던 조회를 그대로 옮긴 것이라 기록된 manifest의 ollama_digest와
# 같은 값이 나온다. 상태 코드는 보지 않는다(오류 응답의 JSON에는 models가 없어 None이 된다).
def ollama_model_digest(base_url: str, model: str, timeout_s: float = 10.0) -> str | None:
    """/api/tags의 모델 digest. 태그가 없으면 None, 서버에 닿지 못하면 예외."""
    import httpx

    names = {model, f"{model}:latest"}
    tags = httpx.get(f"{base_url}/api/tags", timeout=timeout_s).json().get("models", [])
    return next((m.get("digest") for m in tags if m.get("name") in names or m.get("model") in names), None)


# Ollama /api/version의 서버 버전. 같은 가중치라도 실행기 버전이 다르면 측정 조건이 다르므로 지문의 일부다.
# 서버에 닿지 못하면 예외가 올라온다.
def ollama_version(base_url: str, timeout_s: float = 10.0) -> str | None:
    """/api/version의 version. 서버에 닿지 못하면 예외."""
    import httpx

    return httpx.get(f"{base_url}/api/version", timeout=timeout_s).json().get("version")


# Ollama 모델의 지문 {"ollama_digest", "ollama_version"}. evaluate.model_fingerprint(실행 기록 manifest)와
# 서비스의 시작 조회가 같은 함수를 써서, 잰 모델과 서빙 중인 모델을 digest로 맞춰 볼 수 있다. 구하지 못한 값은
# None이고 예외를 내지 않는다(모델 서버가 없으면 측정은 뒤에서 실제 오류로 멈추고, 서비스는 그래도 뜬다). 키
# 이름과 순서는 기록된 manifest와 같아야 한다(tests/test_evaluate.py가 고정).
def ollama_fingerprint(base_url: str, model: str, timeout_s: float = 10.0) -> dict:
    """Ollama 모델의 digest와 서버 버전. 구하지 못한 값은 None(실행은 멈추지 않는다)."""
    fp: dict = {}
    try:
        fp["ollama_digest"] = ollama_model_digest(base_url, model, timeout_s)
    except Exception:  # noqa: BLE001 - 모델 서버 없음: 기록에는 None, 실행은 뒤에서 실제 오류로 실패한다
        fp["ollama_digest"] = None
    try:
        fp["ollama_version"] = ollama_version(base_url, timeout_s)
    except Exception:  # noqa: BLE001
        fp["ollama_version"] = None
    return fp


# 프롬프트 없는 POST /api/generate {model, keep_alive}로 모델을 메모리에 올린다: Ollama는 빈 프롬프트 요청을
# 적재 요청으로 처리하고 토큰을 만들지 않는다(응답의 done_reason "load"). keep_alive 0이면 같은 요청이 모델을
# 내린다(bench.unload가 그렇게 쓴다). 서비스의 선택 예열(CALL_SUMMARY_WARMUP=1)이 시작할 때 한 번 부른다.
# 실패한 HTTP 상태와 연결 오류는 httpx 예외로 올라온다. timeout_s는 디스크에서 가중치를 읽는 시간까지 덮어야
# 한다(기록: q8_0 콜드 적재 최대 약 68초).
def ollama_load(base_url: str, model: str, keep_alive: str | int, timeout_s: float = 120.0) -> None:
    """모델을 메모리에 올린다(프롬프트 없는 /api/generate). 실패하면 httpx 예외."""
    import httpx

    body = {"model": model, "keep_alive": keep_alive}
    httpx.post(f"{base_url}/api/generate", json=body, timeout=timeout_s).raise_for_status()


# 바탕 모델(Hugging Face Hub)을 불러올 때 고정하는 커밋 revision. 값은 기록된 모든 학습·평가가 읽은 로컬 캐시
# 스냅샷의 커밋이다(실행 기록 manifest의 model_fingerprint.base_revision, docs/experiments.md 재측정 결과
# "바탕 모델 revision 1cfa9a72…"). 커밋 해시 revision은 캐시에 있으면 네트워크를 쓰지 않으므로, 고정해도 지금
# 캐시에서 같은 파일을 읽는다. revision 없이 불러오면 refs/main을 따라가 저장소 쪽이 가중치·토크나이저·채팅
# 템플릿을 바꿔 올리면 조용히 다른 모델을 재게 된다(공급망·재현성). 표에 없는 모델(로컬 경로, 다른 바탕)은
# 고정하지 않아 예전 동작 그대로다.
BASE_REVISIONS: dict[str, str] = {
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}


# 불러올 revision을 정한다. 입력: 모델 id(또는 로컬 경로)와 명령줄 --revision 값. 출력: --revision이 있으면 그
# 값(커밋 해시, 태그, "main" 모두 그대로), 없으면 BASE_REVISIONS의 값, 표에도 없으면 None(= revision 인자를
# 주지 않은 예전 동작, refs/main). train·evaluate(hf)·export가 같은 규칙으로 쓰고 결과를 train_config.json과
# manifest의 revision에 남긴다.
def revision_for(model_id: str, override: str | None = None) -> str | None:
    """바탕 모델을 불러올 revision: --revision 값 → BASE_REVISIONS → None(고정 없음)."""
    if override:
        return override
    return BASE_REVISIONS.get(model_id)


# 실행 기록에 남기는 학습·추론 라이브러리. 어댑터를 만든 쪽(train)과 잰 쪽(evaluate hf)의 판이 다르면 같은
# 어댑터라도 결과가 다를 수 있어(transformers의 채팅 템플릿·생성 코드, peft의 LoRA 적용, bitsandbytes의 4bit
# 커널) 조건의 일부로 적는다.
LIBS = ("torch", "transformers", "peft", "bitsandbytes", "accelerate")


# 설치된 라이브러리 판을 {이름: 판 또는 None}으로 모은다. importlib.metadata로 읽으므로 라이브러리를
# import하지 않고(torch import는 몇 초가 걸린다), 설치되지 않은 것은 None이다. cuda=True면 torch를 들여와
# torch.version.cuda (예: "12.8")와 cuDNN 판(정수)을 더한다. torch가 없거나 cuDNN을 쓸 수 없으면 둘 다
# None이고, 어떤 경우에도 예외를 내지 않는다(기록용 값이라 실행을 멈추지 않는다). 쓰는 곳: evaluate의 hf
# 백엔드 manifest(libs 키), train의 train_config.json(libs 키). Ollama·scripted 백엔드에서는 부르지 않는다.
def library_versions(cuda: bool = False) -> dict:
    """학습·추론 라이브러리의 설치 판(없으면 None). cuda=True면 torch의 CUDA·cuDNN 판도."""
    out: dict = {}
    for name in LIBS:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    if cuda:
        try:
            import torch

            out["cuda"] = torch.version.cuda
            out["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001 - torch 없음(학습 환경 아님)이거나 cuDNN 조회 실패: 기록만 비운다
            out["cuda"] = None
            out["cudnn"] = None
    return out


# transformers로 직접 돌리는 실행기. 학습 전 학생의 기준선(Qwen3 fp16)과, GGUF로 바꾸기 전의 학습한 학생(바탕
# 모델 + PEFT LoRA 어댑터, 4B는 4bit 바탕)을 잴 때 쓴다(evaluate --backend hf).
# RTX 2080 Ti(Turing)에는 bf16이 없어 fp16으로 돌린다(docs/plan.md).
@dataclass
class HFProvider:
    """transformers로 탐욕적 디코딩을 한다. 이 GPU(Turing)에는 bf16이 없어 fp16으로 돌린다."""

    model_id: str
    adapter: str | None = None
    load_4bit: bool = False
    max_new_tokens: int = 1024
    enable_thinking: bool = False
    # PyTorch가 쓸 GPU 메모리 상한(GB). Windows에서는 VRAM을 넘겨도 오류 없이 시스템 메모리로 새어 나가 몇
    # 배나 느려지므로, train.cap_vram으로 상한을 건다(train.cap_vram docstring).
    max_vram_gb: float | None = 9.5
    # 바탕 모델을 불러올 Hub revision(커밋 해시 등). None이면 transformers 기본(refs/main)이다. evaluate가
    # revision_for로 정해 넘기고 manifest의 revision에 같은 값을 남긴다. 실행 이름(name)에는 넣지 않는다:
    # 이름이 바뀌면 기록된 실행과의 대조(compare)가 깨진다.
    revision: str | None = None
    _model: object = field(default=None, init=False, repr=False)
    _tok: object = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return (
            f"hf:{self.model_id}"
            + (f"+{self.adapter}" if self.adapter else "")
            + (":4bit" if self.load_4bit else "")
        )

    # 처음 generate할 때 한 번만 모델과 토크나이저를 올린다. 무거운 import도 여기서 해 테스트 환경에서는
    # 실행되지 않는다.
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .train import cap_vram

        cap_vram(self.max_vram_gb)
        kwargs: dict = {"dtype": torch.float16, "device_map": "cuda"}
        # 4bit 바탕(QLoRA 학습 때와 같은 nf4, 계산은 fp16)으로 올린다. 학습(train.py)과 내보내기(export.py)는
        # double quant를 켜고 여기서는 끈다. 이 차이는 알려진 한계로 기록돼 있고, 모든 4bit 평가가 같은
        # 설정이라 비교에는 영향이 없다고 보았다(docs/experiments.md 4단계 한계).
        if self.load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        # 토크나이저와 바탕 가중치를 같은 revision으로 불러온다(None이면 refs/main). 캐시에 그 커밋의 스냅샷이
        # 있으면 네트워크를 쓰지 않는다.
        tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, revision=self.revision, **kwargs)
        # 어댑터가 있으면 바탕 위에 LoRA를 얹는다(합치지 않고 그대로 추론). eval()로 dropout 등을 끈다.
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        self._model, self._tok = model, tok

    # 한 건 생성은 크기 1인 배치로 처리한다. json_schema는 쓰지 않는다(transformers 경로에는 스키마 강제가
    # 없다).
    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        return self.generate_batch([messages])[0]

    # 여러 대화를 왼쪽 패딩으로 묶어 한 번에 탐욕적 생성한다. evaluate가 --batch > 1일 때 길이가 비슷한 건끼리
    # 묶어 부른다. 지연은 배치 전체 시간을 건수로 나눈 값이라 배치 크기가 1인 실행끼리만 비교할 수
    # 있다(docstring).
    def generate_batch(self, batch: list[list[dict]]) -> list[Reply]:
        """여러 대화를 한 번에 탐욕적 디코딩한다(왼쪽 패딩). 지연은 배치 전체의 벽시계 시간을 건수로
        똑같이 나눈 값이라, 건당 지연은 배치 크기가 1인 실행끼리만 비교할 수 있다."""
        import torch

        self._load()
        tok, model = self._tok, self._model
        # 모델의 채팅 템플릿으로 프롬프트 글을 만든다. enable_thinking=False면 Qwen3 템플릿이 빈 생각 블록을
        # 넣어 생각 모드를 끈다. 학습도 같은 템플릿으로 입력을 만든다(train.encode).
        prompts = [
            tok.apply_chat_template(  # type: ignore[attr-defined]
                m, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
            )
            for m in batch
        ]
        # 생성 모델은 오른쪽에 이어 쓰므로 패딩은 왼쪽에 둔다(오른쪽 패딩이면 패딩 뒤에서 생성이 시작된다).
        # 패딩 토큰이 없으면 EOS를 패딩으로 쓴다.
        tok.padding_side = "left"  # type: ignore[attr-defined]
        if tok.pad_token is None:  # type: ignore[attr-defined]
            tok.pad_token = tok.eos_token  # type: ignore[attr-defined]
        # 학습(train.encode)과 같은 방식으로 토큰화한다. 채팅 템플릿에 특수 토큰이 이미 들어 있으므로
        # add_special_tokens=False로 한 번 더 붙이지 않는다.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")  # type: ignore[operator]
        # no_grad로 기울기 계산을 끄고, do_sample=False로 탐욕적 생성을 한다. 답 상한은 max_new_tokens(기본
        # 1,024, Ollama의 num_predict와 같다).
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(  # type: ignore[attr-defined]
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,  # type: ignore[attr-defined]
            )
        latency = (time.perf_counter() - t0) / len(batch)
        # 출력은 [패딩된 입력 + 새 토큰]이므로, 입력 폭 뒤만 잘라 새로 만든 답으로 쓴다. 새 토큰 수는 패딩
        # 토큰이 아닌 것만 센다(일찍 끝난 건은 뒤가 패딩으로 채워진다). 프롬프트 토큰 수는 attention mask의
        # 1을 센다.
        width = enc["input_ids"].shape[1]
        replies = []
        for i in range(len(batch)):
            new = out[i, width:]
            n_new = int((new != tok.pad_token_id).sum())  # type: ignore[attr-defined]
            replies.append(
                Reply(
                    text=tok.decode(new, skip_special_tokens=True),  # type: ignore[attr-defined]
                    latency_s=latency,
                    prompt_tokens=int(enc["attention_mask"][i].sum()),
                    completion_tokens=n_new,
                )
            )
        return replies
