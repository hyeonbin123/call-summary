"""scripts/ctx_probe.py 테스트: 모델 없이 요청 본문과 입력 선택, 결과 분류, 관문 판정만 본다."""

# [테스트 묶음] 서비스 컨텍스트 초과 동작 확인 도구(scripts/ctx_probe.py)와 그 도구가 쓰는 OllamaProvider의
# truncate·shift 필드를 모델 없이 검증한다: 기본 요청 본문이 그대로인지(기록된 요청과 같음), 두 필드가 요청의
# 맨 위에 들어가는지, 두 묶음의 요청이 두 필드만 다른지, dev 40건 고르기(업종마다 파일 앞 13·13·14건), 합성
# 입력이 고정되어 있고 서비스 상한 안인지, 모델 서버 응답을 서비스가 첫 시도에서 어떻게 다룰지의 분류, 관문
# 판정, 가짜 Ollama로 등록한 요청 순서 전체.
# [왜 중요한가] 이 도구의 결과로 서비스 요청에 두 옵션을 넣을지 정한다(docs/experiments.md "서비스 컨텍스트
# 초과 동작 확인"). 두 묶음이 옵션 말고도 다르거나, 고르는 건이 바뀌거나, 판정 규칙이 등록과 다르면 관문이
# 다른 것을 재게 된다. 그리고 옵션을 넣지 않은 요청이 예전과 바이트까지 같아야 기록된 실행과 견줄 수 있다.

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

from call_summary import service
from call_summary.dataset import load_items
from call_summary.prompts import build_messages, format_transcript
from call_summary.providers import OllamaProvider

ROOT = Path(__file__).resolve().parents[1]


# scripts/ 아래 스크립트는 패키지가 아니라서 파일 경로로 불러온다(tests/test_asr_qwen.py와 같은 방식).
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


probe = _load("ctx_probe", ROOT / "scripts" / "ctx_probe.py")
MSGS = build_messages("shop", "상담원: 도토리마켓입니다.\n고객: 주문 취소하려고요.")


# httpx.post를 바꿔 /api/chat 요청 본문을 모으고, 정상 답(빈 JSON, 토큰 수 없음)을 돌려준다.
def _capture(monkeypatch) -> list[dict]:
    bodies: list[dict] = []

    def fake_post(url, **kw):
        bodies.append(kw["json"])
        return httpx.Response(200, json={"message": {"content": "{}"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    return bodies


# 기본 OllamaProvider의 요청 본문에는 truncate·shift가 없다. 키 목록까지 고정해, 새 필드를 더해도
# 서비스·평가가 지금까지 보낸 요청(기록된 모든 실행)이 바뀌지 않았음을 지킨다.
def test_default_request_body_is_unchanged(monkeypatch):
    bodies = _capture(monkeypatch)
    OllamaProvider(model="m", num_ctx=4096).generate(MSGS)
    (body,) = bodies
    assert set(body) == {"model", "messages", "stream", "keep_alive", "options", "think"}
    assert set(body["options"]) == {"temperature", "num_ctx", "num_predict", "seed"}


# 두 필드는 options 안이 아니라 요청의 맨 위에 들어가고(Ollama API의 자리), 나머지 본문은 기본과 같다.
# 서비스의 재시도 provider(dataclasses.replace로 만든다)도 두 값을 그대로 물려받는다.
def test_truncate_and_shift_are_top_level_request_fields(monkeypatch):
    bodies = _capture(monkeypatch)
    base = OllamaProvider(model="m", num_ctx=4096)
    strict = dataclasses.replace(base, truncate=False, shift=False)
    base.generate(MSGS)
    strict.generate(MSGS)
    service._retry_provider(strict).generate(MSGS)
    plain, opted, retry = bodies
    assert (opted["truncate"], opted["shift"]) == (False, False)
    assert "truncate" not in opted["options"] and "shift" not in opted["options"]
    assert {k: v for k, v in opted.items() if k not in ("truncate", "shift")} == plain
    assert (retry["truncate"], retry["shift"]) == (False, False)


# 도구의 두 묶음(옵션 없음, 옵션 있음)은 요청 본문이 두 필드만 다르고, 옵션 없음 묶음은 서비스와 같은
# 설정(num_ctx 4,096, 탐욕적, seed 0, 스키마 없음, 생각 끔)의 본문이다.
def test_probe_variants_differ_only_in_the_two_fields(monkeypatch):
    bodies = _capture(monkeypatch)
    for variant in probe.VARIANTS:
        probe.make_provider("call-summary-4b-dq:q8_0", "http://127.0.0.1:11434", variant).generate(MSGS)
    OllamaProvider(model="call-summary-4b-dq:q8_0", num_ctx=service.NUM_CTX).generate(MSGS)
    default, opted, like_service = bodies
    assert list(probe.VARIANTS) == ["default", "no_truncate_no_shift"]
    assert default == like_service
    assert {k for k in set(default) | set(opted) if default.get(k) != opted.get(k)} == {"truncate", "shift"}
    assert (opted["truncate"], opted["shift"]) == (False, False)


# dev 40건은 업종마다 파일에 나온 순서로 앞의 건을 고른다. dev는 택배 80 → 쇼핑몰 80 → 통신 80 순서라
# 13·13·14건이고 고른 순서도 파일 순서다.
def test_dev_selection_takes_the_first_items_of_each_domain():
    items = load_items(ROOT / "datasets" / "dev.jsonl")
    chosen = probe.select_dev(items, 40)
    want = []
    for domain, k in (("parcel", 13), ("shop", 13), ("telecom", 14)):
        want += [it for it in items if it.domain == domain][:k]
    assert [it.item_id for it in chosen] == [it.item_id for it in want]
    assert [it.item_id for it in probe.select_dev(items, 3)] == [
        want[0].item_id,
        want[13].item_id,
        want[26].item_id,
    ]
    with pytest.raises(ValueError):
        probe.select_dev(items, 2)


# 합성 입력 둘(택배 업종)은 고정된 발화 하나를 서비스 상한(4,000자) 안에서 되풀이한 것이다. 글자 수는 규칙에
# 적은 값(3,995자, 3,983자)과 같고, 서비스의 발화 검사와 정규화를 바꿈 없이 통과한다(서비스가 받는 입력이다).
def test_synthetic_inputs_are_fixed_and_fit_the_service():
    got = probe.synthetic_inputs()
    assert list(got) == ["prompt_over", "answer_over"]
    assert {k: v["chars"] for k, v in got.items()} == {"prompt_over": 3995, "answer_over": 3983}
    for v in got.values():
        assert v["domain"] == "parcel"
        assert v["transcript"] == format_transcript(v["turns"]) and len(v["transcript"]) == v["chars"]
        assert v["chars"] <= service.MAX_CHARS
        for t in v["turns"]:
            service.Turn.model_validate(t)
            assert service.normalize_turn_text(t["text"]) == t["text"]
    assert probe.synthetic_inputs() == got


# Ollama 응답 하나를 흉내 내는 httpx.post. status가 200이면 답 글과 토큰 수, 아니면 {"error": message}를 준다.
def _fake_reply(monkeypatch, status=200, text="{}", prompt=900, completion=200, done="stop", message=""):
    def fake_post(url, **kw):
        req = httpx.Request("POST", url)
        if status != 200:
            return httpx.Response(status, json={"error": message}, request=req)
        data = {
            "message": {"content": text},
            "prompt_eval_count": prompt,
            "eval_count": completion,
            "done_reason": done,
            "load_duration": 5,
        }
        return httpx.Response(200, json=data, request=req)

    monkeypatch.setattr(httpx, "post", fake_post)


# 정상 답 하나(쇼핑몰 기록, 목록 안 라벨).
def _good_json() -> str:
    from call_summary.prompts import record_to_json
    from call_summary.specs import make_spec

    return record_to_json(make_spec("dev", "shop", 3, category="주문 취소").gold("요약"))


# 모델 서버 응답 하나를 서비스가 첫 시도에서 어떻게 다룰지: 컨텍스트 초과 오류는 413(context_exceeded), 다른
# HTTP 오류는 503(model_unavailable), 토큰 합이 컨텍스트를 채우면 502(context_full), 통과하면 ok, 형식·라벨
# 검사에 걸리면 다시 묻는다(retry). 분류는 서비스의 함수(service.context_flags, service.is_context_exceeded)를
# 그대로 쓴다.
@pytest.mark.parametrize(
    "kw,view",
    [
        ({"status": 500, "message": "the request exceeds the available context size"}, "context_exceeded"),
        ({"status": 404, "message": "model not found"}, "model_unavailable"),
        ({"prompt": 3998, "completion": 98, "done": "length"}, "context_full"),
        ({}, "ok"),
        ({"text": "not json"}, "retry"),
    ],
)
def test_call_classifies_like_the_service(kw, view, monkeypatch):
    if "text" not in kw and kw.get("status", 200) == 200:
        kw = {**kw, "text": _good_json()}
    _fake_reply(monkeypatch, **kw)
    row = probe.call(probe.make_provider("m", "http://127.0.0.1:11434", "default"), "shop", "고객: 취소요")
    assert row["service"] == view
    if kw.get("status", 200) != 200:
        assert row["status"] == kw["status"] and row["error"]["body"] and row["text"] is None
    else:
        assert row["status"] == 200 and row["error"] is None
    if view == "context_full":
        assert (row["length_stop"], row["context_full"]) == (True, True)


# 관문 판정에 쓰는 줄 하나를 만든다(필요한 칸만).
def _row(status=200, text="{}", done="stop", prompt=900, completion=200):
    return {
        "status": status,
        "text": text if status == 200 else None,
        "done_reason": done,
        "prompt_eval_count": prompt,
        "eval_count": completion,
        "error": None if status == 200 else {"type": "HTTPStatusError", "body": "x"},
    }


# 관문: (1) 합성 입력 하나 이상에서 옵션이 동작(상태, 멈춘 이유, 토큰 수)을 바꾸고 (2) dev 답이 모두 글자까지
# 같고 오류가 없을 때만 adopt. 하나라도 어긋나면 keep_default.
def test_gate_needs_a_behaviour_change_and_identical_dev_answers():
    changed = [
        {
            "name": "prompt_over",
            "default": _row(prompt=4096, completion=0),
            "no_truncate_no_shift": _row(400),
        },
        {"name": "answer_over", "default": _row(), "no_truncate_no_shift": _row()},
    ]
    same = [{"name": s["name"], "default": _row(), "no_truncate_no_shift": _row()} for s in changed]
    dev = [
        {"item_id": f"d{i}", "default": _row(text=f"a{i}"), "no_truncate_no_shift": _row(text=f"a{i}")}
        for i in range(40)
    ]
    assert probe.gate(changed, dev)["decision"] == "adopt"
    assert probe.gate(changed, dev)["dev_identical"] == 40
    assert probe.gate(same, dev)["decision"] == "keep_default"
    one_diff = [*dev[:-1], {**dev[-1], "no_truncate_no_shift": _row(text="other")}]
    assert probe.gate(changed, one_diff)["decision"] == "keep_default"
    one_error = [*dev[:-1], {**dev[-1], "no_truncate_no_shift": _row(500)}]
    assert probe.gate(changed, one_error)["decision"] == "keep_default"


# 가짜 Ollama(/api/ps, /api/tags, /api/version, /api/chat, /api/generate)로 main을 끝까지 돌린다. 등록한 순서:
# 옵션 없음 묶음(준비 요청 + dev) → 옵션 있음 묶음(준비 요청 + dev) → 합성 입력 둘 × 두 묶음. 끝나면 모델을
# 내린다 (keep_alive 0). 기록 파일에 관문 판정, 참고 대조, /api/ps가 들어간다.
def test_main_sends_the_registered_order_to_a_fake_ollama(tmp_path, monkeypatch):
    model = "call-summary-4b-dq:q8_0"
    state = {"loaded": False}
    chats: list[dict] = []
    unloads: list[dict] = []

    def fake_get(url, **kw):
        req = httpx.Request("GET", url)
        if url.endswith("/api/ps"):
            models = (
                [{"name": model, "size": 1, "size_vram": 1, "context_length": 4096}]
                if state["loaded"]
                else []
            )
            return httpx.Response(200, json={"models": models}, request=req)
        if url.endswith("/api/tags"):
            return httpx.Response(
                200, json={"models": [{"name": model, "digest": "52a2d2f65261"}]}, request=req
            )
        assert url.endswith("/api/version"), url
        return httpx.Response(200, json={"version": "0.35.1"}, request=req)

    def fake_post(url, **kw):
        req = httpx.Request("POST", url)
        body = kw["json"]
        if url.endswith("/api/generate"):
            unloads.append(body)
            state["loaded"] = False
            return httpx.Response(200, json={}, request=req)
        chats.append(body)
        state["loaded"] = True
        chars = len(body["messages"][-1]["content"])
        if body.get("truncate") is False and chars > 3000:
            return httpx.Response(500, json={"error": "input exceeds the context length"}, request=req)
        data = {
            "message": {"content": "{}"},
            "prompt_eval_count": 500 + chars,
            "eval_count": 200,
            "done_reason": "stop",
        }
        return httpx.Response(200, json=data, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    out = tmp_path / "probe.json"
    argv = [
        "--model",
        model,
        "--dev",
        str(ROOT / "datasets" / "dev.jsonl"),
        "--n-dev",
        "3",
        "--out",
        str(out),
    ]
    assert probe.main([*argv, "--allow-dirty"]) == 0
    opted = [b.get("truncate") is False and b.get("shift") is False for b in chats]
    assert opted == [False] * 4 + [True] * 4 + [False, True, False, True]
    assert chats[0]["messages"] == chats[4]["messages"]  # 두 묶음의 준비 요청은 같다
    assert unloads == [{"model": model, "keep_alive": 0}]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["item_id"] for r in report["dev"]["rows"]] == [
        "dev-parcel-00000",
        "dev-shop-00000",
        "dev-telecom-00000",
    ]
    assert report["dev"]["identical"] == 3
    assert [s["name"] for s in report["synthetic"]] == ["prompt_over", "answer_over"]
    over = report["synthetic"][0]
    assert over["default"]["service"] == "context_full"
    assert over["no_truncate_no_shift"]["service"] == "context_exceeded"
    assert report["gate"]["decision"] == "adopt"
    assert report["fingerprint"] == {"ollama_digest": "52a2d2f65261", "ollama_version": "0.35.1"}
    assert report["api_ps_before"] == {"models": []} and report["api_ps_after"] == {"models": []}
    assert report["reference"]["n"] == 3


# 다른 모델이 올라가 있으면(다른 GPU 작업) 재지 않고 멈춘다. 깨끗하지 않은 트리도 --allow-dirty 없이는 멈춘다.
def test_main_refuses_a_busy_server_and_a_dirty_tree(tmp_path, monkeypatch):
    def busy_get(url, **kw):
        return httpx.Response(200, json={"models": [{"name": "other"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", busy_get)
    argv = ["--out", str(tmp_path / "x.json"), "--dev", str(ROOT / "datasets" / "dev.jsonl")]
    with pytest.raises(SystemExit):
        probe.main([*argv, "--allow-dirty"])
    monkeypatch.setattr(probe, "tree_is_clean", lambda: False)
    with pytest.raises(SystemExit):
        probe.main(argv)
    assert not (tmp_path / "x.json").exists()
