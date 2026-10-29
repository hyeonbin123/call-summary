# [테스트 묶음] 서비스 경로 품질 측정 도구 call_summary.service_eval을 검증한다. 서비스에 보낸 결과를 일반
# 평가 실행과 같은 채점으로 세는지, 거절(502)을 형식 실패로 세는지, 상태·시도 집계가 맞는지 본다. 뒤의 두
# 시험은 명령(main)이 서비스의 /ready를 먼저 불러 그 응답과 갱신된 모델 지문을 manifest에 남기는지, 준비되지
# 않은 서비스에서는 공식 실행(--official)을 거절하는지 본다.
# [왜 중요한가] "서비스 경로가 품질을 잃지 않음" 판정은 이 도구의 결과와 모델을 직접 잰 결과의 짝지은 차이로
# 내린다(docs/experiments.md 5단계 "다시 재기"). 채점 방식이 다르면 비교가 성립하지 않는다. 잰 모델과 서빙
# 모델이 같은 가중치인지는 manifest의 digest로만 맞춰 볼 수 있다.

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from call_summary import service, service_eval
from call_summary.dataset import write_jsonl
from call_summary.prompts import record_to_json
from call_summary.providers import OllamaProvider, ScriptedProvider
from call_summary.service import create_app
from call_summary.service_eval import run_service_items, service_summary
from tests.test_scoring import _item

# 서빙 모델 call-summary-4b-dq:q8_0의 digest(tests/test_service.py의 DIGEST와 같은 값).
DIGEST = "52a2d2f652612d64736f136a48de27df1dd9ce1ae3c5993a5f34cab49f81b3e3"


# 가짜 provider로 서비스를 띄우고 4건을 보낸다. 앞 세 건은 정답을 받아 exact이고, 네 번째는 두 번 모두 깨진
# 답이라 서비스가 502로 거절해 예측 기록 없이 형식 실패로 채점된다(5단계 확정 규칙: 거절은 형식 실패로 센다).
# 상태 집계 200×3·502×1, 시도 집계(200 응답만) 1×3, exact 0.75를 확인한다.
def test_service_items_are_scored_like_an_evaluation_run():
    items = [_item(i) for i in range(4)]
    replies = [record_to_json(it.gold()) for it in items[:3]] + ["bad", "still bad"]
    with TestClient(create_app(ScriptedProvider(replies))) as client:
        rows, scores = run_service_items(client, items)
    assert [s.exact for s in scores] == [True, True, True, False]
    assert rows[3]["status"] == 502 and rows[3]["pred"] is None and not scores[3].schema_ok
    table = service_summary(rows, scores)
    assert table["status"] == {"200": 3, "502": 1}
    assert table["attempts"] == {"1": 3}
    assert table["point"]["exact"] == 0.75


# 모델 서버(Ollama) 흉내: 서비스가 부르는 httpx.get(/api/tags, /api/version)과 httpx.post(/api/chat)를 바꾼다.
# down_tags번째까지의 /api/tags 호출은 연결 거부라(서비스가 모델 서버보다 먼저 뜬 상태) 그 뒤에만 서빙 모델
# 태그와 digest가 보인다. /api/chat은 연결이 되면 reply를 모델 답으로 돌려주고, 부른 횟수를 센다.
class FakeOllama:
    def __init__(self, monkeypatch, reply: str, down_tags: int):
        self.reply = reply
        self.down_tags = down_tags
        self.tags_calls = 0
        self.chat_calls = 0
        monkeypatch.setattr(httpx, "get", self.get)
        monkeypatch.setattr(httpx, "post", self.post)

    def up(self) -> bool:
        return self.tags_calls > self.down_tags

    def get(self, url, **kw):
        req = httpx.Request("GET", url)
        if url.endswith("/api/tags"):
            self.tags_calls += 1
            if not self.up():
                raise httpx.ConnectError("refused", request=req)
            models = [{"name": "m:q8_0", "model": "m:q8_0", "digest": DIGEST}]
            return httpx.Response(200, json={"models": models}, request=req)
        assert url.endswith("/api/version"), url
        if not self.up():
            raise httpx.ConnectError("refused", request=req)
        return httpx.Response(200, json={"version": "0.35.1"}, request=req)

    def post(self, url, **kw):
        assert url.endswith("/api/chat"), url
        self.chat_calls += 1
        req = httpx.Request("POST", url)
        if not self.up():
            raise httpx.ConnectError("refused", request=req)
        return httpx.Response(200, json={"message": {"content": self.reply}}, request=req)


# service_eval.main이 쓸 서비스 하나를 준비한다: OllamaProvider로 만든 앱(가짜 모델 서버 앞), main이 여는
# httpx.Client 대신 그 앱의 TestClient(with로 열려 시작 조회가 돈다), dev 1건짜리 데이터 파일. 실행 기록은
# tmp_path 아래(outputs/runs/, reports/)에 쓰이도록 작업 폴더를 옮긴다. /ready는 캐시 없이 부를 때마다
# 모델 서버를 다시 확인하게 한다(READY_TTL_S 0: 시작 조회 뒤 모델 서버가 뜬 것을 바로 보게).
def _service_for_main(monkeypatch, tmp_path, down_tags: int):
    item = _item(0)
    fake = FakeOllama(monkeypatch, record_to_json(item.gold()), down_tags)
    monkeypatch.setattr(service, "READY_TTL_S", 0.0)
    for name in ("CALL_SUMMARY_MODEL_DIGEST", "CALL_SUMMARY_WARMUP", "CALL_SUMMARY_MAX_INFLIGHT"):
        monkeypatch.delenv(name, raising=False)
    app = create_app(OllamaProvider(model="m:q8_0"))
    monkeypatch.setattr(httpx, "Client", lambda **kw: TestClient(app))
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [item.to_dict()])
    monkeypatch.chdir(tmp_path)
    return fake, data


# 실행 폴더 하나의 manifest.json을 읽는다(그 폴더 아래에 실행이 하나뿐이어야 한다).
def _only_manifest(folder) -> dict:
    (path,) = list(folder.glob("*/manifest.json"))
    return json.loads(path.read_text(encoding="utf-8"))


# 서비스가 모델 서버보다 먼저 떠서 시작 조회가 닿지 못했으면 /health의 model_digest는 null이다. service_eval은
# /ready를 먼저 불러(모델 서버를 다시 확인하고 /health의 지문도 갱신한다) 그 응답을 manifest의 service_ready에
# 남기고, 그 뒤에 받은 service_health에도 digest와 Ollama 버전이 들어간다. 고치기 전에는 /health만 불러 이런
# 실행이 digest 없이 기록됐다.
def test_run_records_ready_and_the_refreshed_model_fingerprint(monkeypatch, tmp_path):
    fake, data = _service_for_main(monkeypatch, tmp_path, down_tags=1)
    assert service_eval.main(["--data", str(data), "--label", "t"]) == 0
    manifest = _only_manifest(tmp_path / "outputs" / "runs")
    ready = manifest["service_ready"]
    assert (ready["http_status"], ready["status"], ready["reason"]) == (200, "ready", None)
    assert (ready["model_digest"], ready["ollama_version"]) == (DIGEST, "0.35.1")
    health = manifest["service_health"]
    assert (health["model_digest"], health["ollama_version"]) == (DIGEST, "0.35.1")
    assert manifest["provider"] == "service:ollama:m:q8_0" and fake.chat_calls == 1


# 모델 서버가 닿지 않아 /ready가 503(no_model_server)이면 공식 실행은 요청을 하나도 보내지 않고 이유를 찍으며
# 멈춘다(argparse 오류, 종료 코드 2, reports/에 아무것도 쓰지 않음): 어느 가중치를 쟀는지 남길 수 없는 실행을
# 공식 기록으로 두지 않는다. 공식이 아닌 실행은 경고만 찍고 돌며 manifest에 그 503 응답을 남긴다. 고치기
# 전에는 공식 실행도 그대로 돌아 모든 요청이 503으로 기록됐다.
def test_official_run_needs_a_ready_service(monkeypatch, tmp_path, capsys):
    fake, data = _service_for_main(monkeypatch, tmp_path, down_tags=10**6)
    monkeypatch.setattr(service_eval, "_git_strict", lambda *args: "")
    with pytest.raises(SystemExit) as stop:
        service_eval.main(["--data", str(data), "--official", "--label", "t"])
    assert stop.value.code == 2
    assert "/ready" in capsys.readouterr().err
    assert not (tmp_path / "reports").exists() and fake.chat_calls == 0

    assert service_eval.main(["--data", str(data), "--label", "t"]) == 0
    assert "warning" in capsys.readouterr().err
    ready = _only_manifest(tmp_path / "outputs" / "runs")["service_ready"]
    assert (ready["http_status"], ready["status"], ready["reason"]) == (503, "not_ready", "no_model_server")
