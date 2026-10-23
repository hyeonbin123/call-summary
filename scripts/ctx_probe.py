"""서비스 컨텍스트 초과 동작 확인: 컨텍스트를 넘는 요청을 Ollama가 어떻게 다루는지 적고, 서비스 요청에
`truncate: false`·`shift: false`를 넣을지 정하는 관문을 잰다(docs/experiments.md "서비스 컨텍스트 초과
동작 확인").

    uv run --no-sync python scripts/ctx_probe.py --model call-summary-4b-dq:q8_0 --dev datasets/dev.jsonl \
        --n-dev 40 --out reports/ctx-probe-q8_0.json

요청은 서비스의 첫 시도와 같다(p1 0-shot, num_ctx 4096, temperature 0, seed 0, num_predict 1024, 스키마 없음,
생각 끔). 두 묶음은 요청 맨 위의 truncate·shift 두 필드만 다르다. 보내는 순서: 옵션 없음 묶음(준비 요청 하나 +
dev) → 옵션 있음 묶음(같은 준비 요청 + 같은 dev) → 합성 입력 둘을 옵션 없음·옵션 있음으로 한 번씩. 끝나면
모델을 내린다. 모델 서버(GPU)가 필요하다.
"""

# [파일 역할] 서비스의 컨텍스트 초과 처리(서비스의 context_full 502, context_exceeded 413)가 실제 Ollama
# 0.35.1의 응답과 맞는지 보고, 서비스 요청에 넘침을 오류·멈춤으로 바꾸는 두 옵션을 넣을지 정하는 GPU 측정
# 도구다.
# [왜 필요한가] 서비스의 4,000자 상한은 한국어 대화 기준이라 숫자·이모지 위주 입력은 그 안에서도 컨텍스트
# 4,096토큰을 넘을 수 있다. 넘칠 때 Ollama가 앞을 잘라 200을 주는지 오류를 주는지는 서버를 돌려 봐야 안다. 두
# 옵션을 넣으면 서빙 요청이 바뀌므로, 넘치지 않는 dev 답이 옵션 때문에 바뀌지 않는다는 것을 같은 세션의 A/B로
# 확인한 뒤에만 넣는다(등록 규칙).
# [설계 이유] (1) 모델 서버 응답의 분류는 서비스의 함수(context_flags, is_context_exceeded, off_list_labels)를
# 그대로 써서, 서비스가 첫 시도에서 그 응답을 어떻게 다룰지를 적는다. (2) 두 묶음 앞에 같은 준비 요청을 하나씩
# 보내, 두 묶음이 같은 프롬프트 캐시 상태에서 같은 순서로 시작하게 한다(앞 요청과 겹치는 앞부분을 서버가
# 재사용하므로, 첫 dev 요청 앞의 요청이 다르면 계산 경로가 달라질 수 있다). (3) 합성 입력은 넘침이 실행기를
# 흔들 수 있어 dev 두 묶음 뒤에 보낸다. (4) 이 도구는 아무 기본값도 바꾸지 않는다: 관문 판정(decision)만 적고,
# 서비스에 옵션을 넣는 일은 결과를 적은 뒤 따로 한다.
# [연결] 결과 JSON은 reports/ctx-probe-<양자화>.json에 쓰고, docs/experiments.md의 결과 절이 그것을 옮긴다.
# tests/test_ctx_probe.py가 모델 없이 요청 본문·입력 선택·분류·관문을 검증한다.

from __future__ import annotations

# subprocess: git 확인이 실패했을 때의 예외 형식을 잡는다. datetime: 기록 시각을 시간대와 함께 적는다.
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx

# 프로젝트 모듈: Ollama /api/ps 조회와 모델 내리기(bench), 데이터셋 읽기, git·데이터 해시·모델 지문(evaluate의
# 도우미, 다른 측정 도구와 같은 기록 꼴), 서비스와 같은 프롬프트, 서비스의 컨텍스트·라벨 판정 함수.
from call_summary.bench import ollama_get, unload
from call_summary.dataset import Item, load_items
from call_summary.evaluate import _file_hash, _git, _git_strict, model_fingerprint
from call_summary.prompts import PROMPT_VERSION, build_messages, format_transcript, prompt_hash
from call_summary.providers import OllamaProvider
from call_summary.schema import parse_reply, record_json_schema
from call_summary.service import MAX_CHARS, NUM_CTX, context_flags, is_context_exceeded, off_list_labels

# 잴 모델(서빙 모델)과 참고로만 견줄 기록된 dev 실행(같은 모델, 2026-09-30, Ollama 0.34.2). 세션 사이에는 같은
# 입력에서도 답이 바뀐 기록이 있어 이 대조는 판정에 쓰지 않는다(docs/experiments.md 음성 조건 v3 단계 B 결과).
MODEL = "call-summary-4b-dq:q8_0"
REFERENCE_RUN = "reports/20260930-022406-call-summary-4b-dq_q8_0-s5b-dev-dq-q8_0"
# 두 묶음: 이름 → OllamaProvider에 더할 필드. 옵션 있음은 truncate·shift를 둘 다 끈다(넘친 프롬프트를 잘라
# 넣지 않고 오류로, 생성 중 컨텍스트가 차면 밀어내지 않고 멈추게). 이 순서로 보낸다.
VARIANTS: dict[str, dict] = {"default": {}, "no_truncate_no_shift": {"truncate": False, "shift": False}}
# 합성 입력: 택배 업종에서 고정된 고객 발화 하나를 서비스 상한(4,000자) 안에서 되풀이한다. 바탕 모델의 Qwen3
# 토크나이저(CPU)로 센 p1 프롬프트 전체 토큰 수는 prompt_over 4,175(3,995자, 프롬프트만으로 4,096 넘음),
# answer_over 3,998(3,983자, 프롬프트는 들어가지만 답 자리가 98토큰뿐이라 dev 정답 길이 p50 201토큰이면 생성
# 중에 넘침)이다. 띄어 쓴 숫자는 음성 인식이 번호를 숫자 하나씩 적는 꼴, 전화번호는 번호가 많은 통화의 꼴이다.
SYNTHETIC_DOMAIN = "parcel"
SYNTHETIC_UNITS = {
    "prompt_over": {"speaker": "고객", "text": "번호는 6 2 9 1 7 8 7 7 5 1 6 8 입니다"},
    "answer_over": {"speaker": "고객", "text": "제 번호는 010-4821-7765이고 집 번호는 02-351-8890이에요."},
}
# 두 묶음 앞에 보내는 같은 준비 요청(카드 업종의 짧은 전사, dev에 없는 업종). 답은 견주지 않는다. 첫 요청이라
# 모델 적재도 여기서 일어나 dev 첫 건의 시간에 섞이지 않는다.
WARMUP_DOMAIN = "card"
WARMUP_TURNS = [
    {"speaker": "상담원", "text": "안녕하세요, 무엇을 도와드릴까요?"},
    {"speaker": "고객", "text": "지난달 결제 내역을 확인하고 싶어요."},
]
# 오류 본문은 이 글자 수까지만 남긴다(기록 파일 크기를 묶고, 긴 내부 메시지를 통째로 옮기지 않으려고).
ERROR_BODY_CHARS = 500
# 관문 (1)에서 "옵션이 동작을 바꿨다"로 보는 칸: HTTP 상태, 멈춘 이유, 프롬프트·생성 토큰 수. 답 글은 넣지
# 않는다: 넘침 처리 방식이 같아도 글은 비결정성으로 다를 수 있고, 알고 싶은 것은 넘침을 다루는 방식이다.
BEHAVIOUR_KEYS = ("status", "done_reason", "prompt_eval_count", "eval_count")


# 고정 발화 하나를 전사(format_transcript)가 max_chars를 처음 넘을 때까지 되풀이한 뒤, 넘은 만큼 뒤에서 뺀다.
# 입력: 발화 dict, 글자 상한. 출력: 발화 목록(전사 글자 수 ≤ max_chars). 결정적이다(난수 없음).
def fill_turns(unit: dict, max_chars: int = MAX_CHARS) -> list[dict]:
    turns: list[dict] = []
    while len(format_transcript(turns)) < max_chars:
        turns.append(dict(unit))
    while len(format_transcript(turns)) > max_chars:
        turns.pop()
    return turns


# 합성 입력 둘을 만든다. 출력: 이름 → {domain, turns, transcript, chars}. 글자 수는 규칙에 적은 값(3,995자,
# 3,983자)이다(tests/test_ctx_probe.py가 고정한다).
def synthetic_inputs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, unit in SYNTHETIC_UNITS.items():
        turns = fill_turns(unit)
        transcript = format_transcript(turns)
        out[name] = {
            "domain": SYNTHETIC_DOMAIN,
            "turns": turns,
            "transcript": transcript,
            "chars": len(transcript),
        }
    return out


# dev에서 n건을 고른다: 파일에 업종이 처음 나온 순서대로, 업종마다 파일 앞의 건. n을 업종 수로 나눈 나머지는
# 뒤 업종부터 한 건씩 더 준다(dev는 택배 → 쇼핑몰 → 통신 순서라 40건이면 13·13·14건). 출력 순서는 업종 순서,
# 그 안에서 파일 순서다. n이 업종 수보다 작거나 한 업종의 건이 모자라면 ValueError.
def select_dev(items: Sequence[Item], n: int) -> list[Item]:
    domains = list(dict.fromkeys(it.domain for it in items))
    if not domains or n < len(domains):
        raise ValueError(f"need at least one item per domain ({len(domains)} domains), got n={n}")
    base, extra = divmod(n, len(domains))
    chosen: list[Item] = []
    for i, domain in enumerate(domains):
        k = base + (1 if i >= len(domains) - extra else 0)
        pool = [it for it in items if it.domain == domain]
        if len(pool) < k:
            raise ValueError(f"{domain} has {len(pool)} items, need {k}")
        chosen += pool[:k]
    return chosen


# 묶음 하나의 실행기. 서비스(create_app)가 만드는 provider와 같은 설정(num_ctx NUM_CTX, 나머지는
# OllamaProvider 기본값: temperature 0, seed 0, num_predict 1024, keep_alive 30m, 스키마 없음, 생각 끔)에
# 묶음의 필드만 더한다.
def make_provider(model: str, base_url: str, variant: str) -> OllamaProvider:
    return OllamaProvider(model=model, base_url=base_url, num_ctx=NUM_CTX, **VARIANTS[variant])


# 요청 하나를 보내고 결과 한 줄을 만든다. 입력: 실행기, 업종, 전사. 출력 dict:
# - status: HTTP 상태(200, 오류 상태, 연결 실패 등 HTTP 응답이 없으면 None), error: {type, body(앞 500자)}
#   또는 None
# - text, prompt_eval_count, eval_count, done_reason, load_duration_ns(모델을 다시 올렸는지 보는 값), wall_s
# - length_stop, context_full: 서비스의 context_flags 그대로
# - service: 서비스가 첫 시도에서 이 응답을 어떻게 다룰지. context_exceeded(413), model_unavailable(503),
#   context_full(502), ok(200), retry(형식·라벨 검사에 걸려 다시 물음). problems는 retry의 이유 모델 서버
#   실패는 무엇이든 잡아 기록하고 돌아온다(측정이 중간에 멈추지 않게).
def call(provider: OllamaProvider, domain: str, transcript: str) -> dict:
    row: dict = {
        "status": None,
        "error": None,
        "text": None,
        "prompt_eval_count": None,
        "eval_count": None,
        "done_reason": None,
        "load_duration_ns": None,
        "wall_s": None,
        "length_stop": None,
        "context_full": None,
        "service": None,
        "problems": [],
    }
    messages = build_messages(domain, transcript)
    t0 = time.perf_counter()
    try:
        reply = provider.generate(messages, json_schema=record_json_schema())
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 기록하고 다음 요청으로 넘어간다
        row["wall_s"] = round(time.perf_counter() - t0, 3)
        if isinstance(exc, httpx.HTTPStatusError):
            row["status"] = exc.response.status_code
            body = exc.response.text
        else:
            body = str(exc)
        row["error"] = {"type": type(exc).__name__, "body": body[:ERROR_BODY_CHARS]}
        row["service"] = "context_exceeded" if is_context_exceeded(exc) else "model_unavailable"
        return row
    row["wall_s"] = round(time.perf_counter() - t0, 3)
    length_stop, context_full = context_flags(reply, provider.num_ctx)
    row.update(
        status=200,
        text=reply.text,
        prompt_eval_count=reply.prompt_tokens,
        eval_count=reply.completion_tokens,
        done_reason=reply.done_reason,
        load_duration_ns=(reply.timings or {}).get("load_duration"),
        length_stop=length_stop,
        context_full=context_full,
    )
    # 서비스 처리기와 같은 순서: 컨텍스트 검사가 먼저(걸리면 재시도 없이 502), 그다음 형식, 그다음 라벨 목록.
    if context_full:
        row["service"] = "context_full"
        return row
    parsed = parse_reply(reply.text)
    if parsed.record is None:
        row.update(service="retry", problems=[parsed.error or "invalid"])
        return row
    problems = off_list_labels(domain, parsed.record)
    row.update(service="retry" if problems else "ok", problems=problems)
    return row


# 두 묶음의 답이 같은지: 둘 다 200이고 답 글이 글자까지 같아야 같다.
def _same(a: dict, b: dict) -> bool:
    return a["status"] == 200 and b["status"] == 200 and a["text"] == b["text"]


# 관문 판정(등록 규칙). 입력: 합성 입력 줄({name, default, no_truncate_no_shift}) 목록과 dev 줄 목록. 출력
# dict: options_change_behaviour와 changed_inputs(합성 입력 가운데 옵션이 상태·멈춘 이유·토큰 수 중 하나라도
# 바꾼 것), dev_n, dev_identical(두 묶음 답이 글자까지 같은 건 수), dev_errors(묶음별 200이 아닌 건 수),
# decision. decision은 (1) 옵션이 동작을 바꾼 합성 입력이 하나 이상 있고 (2) dev 답이 모두 같고 오류가 없을
# 때만 adopt(서비스 요청에 두 옵션을 넣는다), 아니면 keep_default(넣지 않는다). (1)이 없으면 이 서버 판에서
# 옵션이 아무것도 바꾸지 않으므로 넣을 이유가 없다.
def gate(synthetic: Sequence[dict], dev: Sequence[dict]) -> dict:
    changed = [
        s["name"]
        for s in synthetic
        if any(s["default"].get(k) != s["no_truncate_no_shift"].get(k) for k in BEHAVIOUR_KEYS)
    ]
    errors = {v: sum(1 for r in dev if r[v]["status"] != 200) for v in VARIANTS}
    identical = sum(1 for r in dev if _same(r["default"], r["no_truncate_no_shift"]))
    dev_ok = bool(dev) and identical == len(dev) and not any(errors.values())
    return {
        "options_change_behaviour": bool(changed),
        "changed_inputs": changed,
        "dev_n": len(dev),
        "dev_identical": identical,
        "dev_errors": errors,
        "decision": "adopt" if changed and dev_ok else "keep_default",
    }


# 참고 대조: 기록된 dev 실행(items.jsonl의 reply)과 답 글이 같은 건 수를 묶음마다 센다. 판정에는 쓰지 않는다.
# 기록 폴더가 없으면 None.
def reference_agreement(dev: Sequence[dict], ref_dir: Path) -> dict | None:
    path = ref_dir / "items.jsonl"
    if not path.exists():
        return None
    ref: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            ref[d["item_id"]] = d["reply"]
    out: dict = {"run": ref_dir.name, "n": len(dev)}
    for v in VARIANTS:
        out[f"identical_{v}"] = sum(
            1 for r in dev if r[v]["text"] is not None and r[v]["text"] == ref.get(r["item_id"])
        )
    return out


# 추적 파일 변경이 없는 깨끗한 트리인지. git 확인이 실패하면 깨끗하지 않은 것으로 친다(evaluate --official과
# 같은 fail closed). 테스트가 바꿔 끼울 수 있게 함수로 둔다.
def tree_is_clean() -> bool:
    try:
        return not _git_strict("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        return False


# 기록 시각: 이 PC의 시간대를 붙인 ISO 8601(초까지).
def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# 명령줄 진입점. 전제 확인(깨끗한 트리, Ollama에 올라간 모델 없음) → 두 묶음 dev → 합성 입력 → 모델 내리기 →
# 관문·참고 대조 → 기록 파일. 성공하면 0을 돌려준다(관문이 keep_default여도 0이다: 판정은 결과 자료다).
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            '서비스 컨텍스트 초과 동작 확인(docs/experiments.md "서비스 컨텍스트 초과 동작 확인"). '
            "서빙 모델에 서비스와 같은 요청을 옵션 없음과 truncate:false·shift:false로 보내, "
            "넘치는 합성 입력의 처리와 dev 답이 옵션 때문에 바뀌는지를 한 기록 파일로 남긴다. "
            "GPU(Ollama)가 필요하다."
        ),
        epilog=(
            "순서: 옵션 없음 묶음(카드 업종 준비 요청 + dev) → 옵션 있음 묶음(같은 준비 요청 + 같은 dev) "
            "→ 합성 입력 prompt_over·answer_over를 옵션 없음·옵션 있음으로 한 번씩 → 모델 내리기. "
            "관문(decision)은 옵션이 합성 입력의 동작을 바꾸고 dev 답이 모두 글자까지 같을 때만 adopt다. "
            "이 도구는 서비스 기본값을 바꾸지 않는다."
        ),
    )
    ap.add_argument("--model", default=MODEL, help=f"잴 Ollama 모델 태그 (기본 {MODEL}, 서빙 모델)")
    ap.add_argument(
        "--dev", default="datasets/dev.jsonl", help="dev 데이터셋 JSONL (기본 datasets/dev.jsonl)"
    )
    ap.add_argument(
        "--n-dev",
        type=int,
        default=40,
        help=(
            "dev에서 고를 건수. 업종마다 파일 앞의 건을 고르고 나머지는 뒤 업종부터 한 건씩 더 준다 "
            "(기본 40 = 13·13·14)"
        ),
    )
    ap.add_argument("--out", required=True, help="기록 파일(JSON) 경로. 예: reports/ctx-probe-q8_0.json")
    ap.add_argument(
        "--ollama", default="http://127.0.0.1:11434", help="Ollama 주소 (기본 http://127.0.0.1:11434)"
    )
    ap.add_argument(
        "--reference",
        default=REFERENCE_RUN,
        help="참고로만 답을 견줄 기록된 evaluate 실행 폴더(items.jsonl). 판정에는 쓰지 않는다",
    )
    ap.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "추적 파일이 바뀐 트리에서도 돈다(시험 실행용). 등록 측정은 규칙을 커밋한 깨끗한 트리에서만 한다"
        ),
    )
    args = ap.parse_args(argv)

    if not args.allow_dirty and not tree_is_clean():
        ap.error("needs a clean working tree (commit the rules first), or --allow-dirty for a trial run")
    try:
        dev_items = select_dev(load_items(args.dev), args.n_dev)
    except ValueError as exc:
        ap.error(str(exc))
    # GPU 작업은 한 번에 하나: 이미 올라간 모델이 있으면 재지 않는다(효율 측정 bench와 같은 규칙).
    before = ollama_get(args.ollama, "/api/ps")
    if before.get("models"):
        ap.error("another model is loaded in Ollama; stop it first (one GPU job at a time)")
    providers = {v: make_provider(args.model, args.ollama, v) for v in VARIANTS}
    fingerprint = model_fingerprint(providers["default"])
    started = _now()
    t_run = time.perf_counter()
    warmups: dict[str, dict] = {}
    per_variant: dict[str, list[dict]] = {}
    synthetic: list[dict] = []
    loaded: dict | None = None
    try:
        for variant, provider in providers.items():
            warmups[variant] = call(provider, WARMUP_DOMAIN, format_transcript(WARMUP_TURNS))
            # 첫 준비 요청으로 모델이 올라간 직후의 /api/ps(size, size_vram, context_length)를 한 번 남긴다.
            if loaded is None:
                loaded = ollama_get(args.ollama, "/api/ps")
            rows = []
            for n, it in enumerate(dev_items, 1):
                rows.append(call(provider, it.domain, it.transcript))
                if n % 10 == 0 or n == len(dev_items):
                    print(f"[{variant} {n}/{len(dev_items)}]", file=sys.stderr, flush=True)
            per_variant[variant] = rows
        for name, inp in synthetic_inputs().items():
            entry: dict = {"name": name, "domain": inp["domain"], "chars": inp["chars"]}
            for variant, provider in providers.items():
                entry[variant] = call(provider, inp["domain"], inp["transcript"])
                print(
                    f"[synthetic {name} {variant}] {entry[variant]['service']}", file=sys.stderr, flush=True
                )
            synthetic.append(entry)
    finally:
        # 측정이 끝나거나 멈추면 GPU를 돌려준다. 내리기가 실패해도 원래 예외를 가리지 않게 경고만 찍는다.
        try:
            unload(args.ollama, args.model)
        except Exception as exc:  # noqa: BLE001 - 기록용 정리 단계
            print(f"warning: could not unload {args.model}: {type(exc).__name__}", file=sys.stderr)
    run_s = round(time.perf_counter() - t_run, 1)
    after = ollama_get(args.ollama, "/api/ps")

    dev_rows = []
    for i, it in enumerate(dev_items):
        a, b = per_variant["default"][i], per_variant["no_truncate_no_shift"][i]
        dev_rows.append(
            {
                "item_id": it.item_id,
                "domain": it.domain,
                "default": a,
                "no_truncate_no_shift": b,
                "identical": _same(a, b),
            }
        )
    base = providers["default"]
    report = {
        "probe": "ctx_probe",
        "rule": "docs/experiments.md 서비스 컨텍스트 초과 동작 확인 (2026-10-24 규칙)",
        "model": args.model,
        "fingerprint": fingerprint,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": not tree_is_clean(),
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "request": {
            "num_ctx": base.num_ctx,
            "num_predict": base.num_predict,
            "temperature": base.temperature,
            "seed": base.seed,
            "think": base.think,
            "schema": base.use_schema,
            "keep_alive": base.keep_alive,
        },
        "variants": VARIANTS,
        "started": started,
        "finished": _now(),
        "run_s": run_s,
        "api_ps_before": before,
        "api_ps_loaded": loaded,
        "api_ps_after": after,
        "warmup": {"domain": WARMUP_DOMAIN, **warmups},
        "dev": {
            "data": args.dev,
            "data_sha": _file_hash(Path(args.dev)),
            "n": len(dev_rows),
            "per_domain": dict(Counter(r["domain"] for r in dev_rows)),
            "identical": sum(1 for r in dev_rows if r["identical"]),
            "rows": dev_rows,
        },
        "synthetic": synthetic,
        "gate": gate(synthetic, dev_rows),
        "reference": reference_agreement(dev_rows, Path(args.reference)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    # 화면에는 핵심만 ASCII JSON 한 줄로 찍는다(관문, 합성 입력마다 서비스 분류). 전체는 --out 파일에 있다.
    brief = {
        "gate": report["gate"],
        "synthetic": {s["name"]: {v: s[v]["service"] for v in VARIANTS} for s in synthetic},
        "reference": report["reference"],
        "out": str(out),
    }
    print(json.dumps(brief), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
