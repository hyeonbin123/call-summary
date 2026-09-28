from fastapi.testclient import TestClient

from call_summary.loadtest import run_load
from call_summary.service import create_offline_app
from tests.test_scoring import _item


def test_run_load_against_offline_service():
    items = [_item(i, category=None) for i in range(9)]
    with TestClient(create_offline_app()) as client:
        res = run_load(client, items, concurrency=3)
        assert res["n"] == 9 and res["status"] == {"200": 9}
        assert res["latency_s"]["p50"] >= 0 and res["throughput_rps"] > 0
        assert client.get("/stats").json()["requests"]["ok"] == 9
