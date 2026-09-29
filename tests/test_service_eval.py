from fastapi.testclient import TestClient

from call_summary.prompts import record_to_json
from call_summary.providers import ScriptedProvider
from call_summary.service import create_app
from call_summary.service_eval import run_service_items, service_summary
from tests.test_scoring import _item


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
