# [테스트 묶음] 서비스 경로 품질 측정 도구 call_summary.service_eval을 검증한다. 서비스에 보낸 결과를 일반
# 평가 실행과 같은 채점으로 세는지, 거절(502)을 형식 실패로 세는지, 상태·시도 집계가 맞는지 본다.
# [왜 중요한가] "서비스 경로가 품질을 잃지 않음" 판정은 이 도구의 결과와 모델을 직접 잰 결과의 짝지은 차이로
# 내린다(docs/experiments.md 5단계 "다시 재기"). 채점 방식이 다르면 비교가 성립하지 않는다.

from fastapi.testclient import TestClient

from call_summary.prompts import record_to_json
from call_summary.providers import ScriptedProvider
from call_summary.service import create_app
from call_summary.service_eval import run_service_items, service_summary
from tests.test_scoring import _item


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
