# [테스트 묶음] 5단계 부하 시험 도구 call_summary.loadtest를 모델 없이 검증한다: 오프라인 서비스에 동시 요청을
# 보내 처리량·지연·상태를 세는지, 전송 실패(시간 초과)를 실행 중단이 아닌 오류 상태로 세는지, 동시성 단계마다
# 결과를 파일에 남겨 뒤 단계가 실패해도 앞 결과가 남는지.
# [왜 중요한가] 부하 표(동시 1·2·4의 처리량, p50/p95, 오류 수)가 이 도구에서 나온다(docs/experiments.md
# 5단계). 타임아웃·연결 끊김을 error:<예외> 상태로 세고 단계마다 --out을 다시 쓰도록 고쳤다(커밋 6238213).

import json
import threading
from types import SimpleNamespace

# TestClient + create_offline_app: 모델 없는 오프라인 서비스(고정 답)에 실제 HTTP 경로로 요청한다. httpx는
# 예외 종류(ReadTimeout, ConnectError)를 만들고 httpx.Client를 가짜로 바꾸는 데 쓴다.
import httpx
import pytest
from fastapi.testclient import TestClient

from call_summary import loadtest
from call_summary.dataset import write_jsonl
from call_summary.loadtest import run_load
from call_summary.service import create_offline_app
from tests.test_scoring import _item


# 동시 3으로 9건을 보내 모두 200이고 지연·처리량이 계산되며, 서비스의 /stats에도 ok 9건이 잡히는지
# 확인한다(클라이언트 수와 서비스 수가 맞는다).
def test_run_load_against_offline_service():
    items = [_item(i, category=None) for i in range(9)]
    with TestClient(create_offline_app()) as client:
        res = run_load(client, items, concurrency=3)
        assert res["n"] == 9 and res["status"] == {"200": 9}
        assert res["latency_s"]["p50"] >= 0 and res["throughput_rps"] > 0
        assert client.get("/stats").json()["requests"]["ok"] == 9


# 세 번째 요청만 시간 초과를 내는 가짜 클라이언트. 동시 요청 스레드가 순번을 같이 세므로 Lock으로 지킨다.
class FlakyClient:
    """세 번째 요청은 시간 초과가 나고, 나머지 요청은 모두 200으로 답한다."""

    def __init__(self) -> None:
        self.n = 0
        self.lock = threading.Lock()

    def post(self, path, json=None):
        with self.lock:
            self.n += 1
            n = self.n
        if n == 3:
            raise httpx.ReadTimeout("timed out")
        return SimpleNamespace(status_code=200)


# 시간 초과 한 건은 "error:ReadTimeout" 상태로 세고 나머지 다섯 건은 정상 집계된다. 지연 분포는 200 응답만으로
# 계산된다.
def test_a_failed_request_is_counted_not_fatal():
    items = [_item(i, category=None) for i in range(6)]
    res = run_load(FlakyClient(), items, concurrency=2)
    assert res["n"] == 6 and res["status"] == {"200": 5, "error:ReadTimeout": 1}
    assert res["latency_s"] is not None


# 부하 시험의 p50·p95는 evaluate와 같은 계산(scoring.pctl)이다. 요청마다 시계를 정해진 만큼 넘기는 가짜
# 클라이언트로 지연 1~20초(짝수 개)를 만든다: p50은 가운데 두 값의 평균(10.5)이 아니라 v[10] = 11초, p95는
# v[int(0.95·20)] = 20초가 아니라 v[int(0.95·19)] = 19초다. 요청마다 건 id·상태·지연이 보낸 순서대로 남아
# 계산식을 바꿔도 기록에서 다시 셀 수 있다.
def test_load_percentiles_use_the_shared_rule(monkeypatch):
    delays = [7, 3, 12, 1, 20, 15, 9, 4, 18, 2, 11, 6, 16, 8, 19, 5, 14, 10, 17, 13]
    clock = {"now": 0.0, "i": 0}
    monkeypatch.setattr(loadtest, "time", SimpleNamespace(perf_counter=lambda: clock["now"]))

    class Client:
        def post(self, path, json=None):
            clock["now"] += delays[clock["i"]]
            clock["i"] += 1
            return SimpleNamespace(status_code=200)

    items = [_item(i, category=None) for i in range(len(delays))]
    res = run_load(Client(), items, concurrency=1)
    assert res["latency_s"]["p50"] == 11.0 and res["latency_s"]["p95"] == 19.0
    assert [r["latency_s"] for r in res["requests"]] == delays
    assert [r["item_id"] for r in res["requests"]] == [it.item_id for it in items]
    assert {r["status"] for r in res["requests"]} == {"200"}


# httpx.Client를 가짜로 바꿔, 동시 1·2 두 단계를 다 돈 뒤 /stats 요청에서 서비스가 사라진 상황을 만든다.
# main은 예외로 끝나지만 --out 파일에는 끝난 두 단계의 결과가 이미 남아 있어야 한다(단계마다 저장).
def test_finished_stages_are_saved_before_a_later_failure(tmp_path, monkeypatch):
    class Client:
        def __init__(self, **kw) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, path, json=None):
            return SimpleNamespace(status_code=200)

        def get(self, path):
            if path == "/stats":
                raise httpx.ConnectError("service went away")
            return SimpleNamespace(json=lambda: {"status": "ok"})

    data = tmp_path / "items.jsonl"
    write_jsonl(data, [_item(i, category=None).to_dict() for i in range(2)])
    out = tmp_path / "load.json"
    monkeypatch.setattr(httpx, "Client", Client)
    with pytest.raises(httpx.ConnectError):
        loadtest.main(["--data", str(data), "--concurrency", "1", "2", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["concurrency"] for r in report["runs"]] == [1, 2]
