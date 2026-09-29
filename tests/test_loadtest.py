import json
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from call_summary import loadtest
from call_summary.dataset import write_jsonl
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


class FlakyClient:
    """The third request times out; every other one answers 200."""

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


def test_a_failed_request_is_counted_not_fatal():
    items = [_item(i, category=None) for i in range(6)]
    res = run_load(FlakyClient(), items, concurrency=2)
    assert res["n"] == 6 and res["status"] == {"200": 5, "error:ReadTimeout": 1}
    assert res["latency_s"] is not None


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
