# [테스트 묶음] train과 정답 기록이 글자까지 같은 건을 세는 보고 도구(call_summary.overlap)를 검증한다:
# 정답 기록 전체가 같은 건만 세는지(요약은 보지 않음), 핵심 값·처리·후속 조치 코드의 순서는 보지 않는지,
# 기록된 실행을 주면 겹친 건과 나머지의 exact를 나누는지, 명령줄 표, 그리고 공개 문서에 적은 숫자를 저장소의
# 데이터와 실행 기록으로 다시 내는지.
# [왜 중요한가] 분할 확정(finalize)은 번호 값이 train과 겹치는 건만 뺀다. 번호가 없는 유형에서는 정답 기록
# 전체가 train의 한 건과 같은 건이 dev·test에 남을 수 있어, 그 수와 그 건들의 exact를
# docs/experiments.md 1단계 "생성 결과와 확정"의 2026-10-29 덧붙임에 적었다. 그 숫자가 커밋된 도구에서
# 나와야 누구나 다시 낼 수 있다.

import dataclasses
import json
from pathlib import Path

import pytest

from call_summary import overlap
from call_summary.dataset import Item, load_items, write_jsonl
from call_summary.specs import make_spec

ROOT = Path(__file__).resolve().parents[1]


# 핵심 값 3개, 처리 2개, 후속 조치 코드 2개가 있는 train 명세(택배 파손 신고). 순서를 바꿔 볼 거리가 있는 첫
# 번호를 고른다(make_spec이 결정적이라 늘 같은 명세가 나온다).
def _base_spec():
    return next(
        s
        for s in (make_spec("train", "parcel", i, category="파손 신고") for i in range(50))
        if len(s.slots) >= 2 and len(s.actions) >= 2 and len(s.follow_up) >= 2
    )


# 명세 내용을 다른 분할·번호로 옮긴다(spec_id와 split만 바뀌고 정답 기록은 그대로). changes로 필드를 바꾼다.
def _copy(spec, split: str, index: int, **changes):
    return dataclasses.replace(spec, spec_id=f"{split}-{spec.domain}-{index:05d}", split=split, **changes)


# 명세 하나로 데이터셋 한 건을 만든다. 발화는 비워 둔다(정답 기록은 명세에서만 나오므로 대화는 필요 없다).
def _item(spec, summary: str = "요약입니다.") -> Item:
    return Item(spec.spec_id, spec.split, spec.domain, spec, turns=[], summary=summary)


# 정답 기록 전체가 train의 한 건과 같은 dev 건만 센다. 참고 요약이 달라도 같은 기록이다(요약은 구조 필드가
# 아니다). 처리 결과 하나, 핵심 값의 말투 하나만 달라도 다른 기록이다(정규화하면 같은 값이어도 글자가 다르면
# 학습 정답의 글이 다르다). train에 같은 기록이 두 건 있어도 dev 한 건은 한 번만 센다.
def test_counts_only_items_whose_whole_record_matches_train():
    base = _base_spec()
    train = [_item(base), _item(_copy(base, "train", 99))]
    same = _item(_copy(base, "dev", 0), summary="다른 참고 요약")
    other = "해결" if base.resolution != "해결" else "이관"
    other_resolution = _item(_copy(base, "dev", 1, resolution=other))
    changed = tuple(
        dataclasses.replace(sv, value="85만 8천 원") if sv.type == "금액" else sv for sv in base.slots
    )
    assert changed != base.slots  # 금액 슬롯이 있어 말투가 바뀐다
    other_value = _item(_copy(base, "dev", 2, slots=changed))

    keys = overlap.train_keys(train)
    assert len(keys) == 1
    rep = overlap.split_report([same, other_resolution, other_value], keys)
    assert rep["n"] == 3 and rep["overlap"] == 1
    assert rep["ids"] == [same.item_id]
    assert rep["by_category"] == {"parcel/파손 신고": 1}


# 핵심 값·처리·후속 조치 코드는 순서를 보지 않는다. 채점(scoring.score_item)이 이 셋을 순서 없이 견주므로,
# 순서만 다른 기록은 같은 정답이다.
def test_order_of_values_actions_and_codes_does_not_matter():
    base = _base_spec()
    reordered = _copy(
        base,
        "dev",
        0,
        slots=tuple(reversed(base.slots)),
        actions=tuple(reversed(base.actions)),
        follow_up=tuple(reversed(base.follow_up)),
    )
    assert reordered.slots != base.slots and reordered.actions != base.actions
    assert overlap.record_key(_item(reordered)) == overlap.record_key(_item(base))
    rep = overlap.split_report([_item(reordered)], overlap.train_keys([_item(base)]))
    assert rep["overlap"] == 1


# 기록된 실행을 주면 그 실행의 exact를 겹친 건과 나머지로 나눈다: 겹친 건 1건(맞음), 나머지 2건(1건 맞음).
# 데이터 경로는 manifest의 data를 쓴다. 기록된 실행은 Windows에서 써 역슬래시 경로라 그 꼴로 적어 둔다
# (dataset.manifest_data가 슬래시 꼴로 바꾼다). 데이터셋에 없는 건이 실행에 있으면 데이터가 맞지 않는 것이라
# 멈춘다.
def test_run_report_splits_exact_between_overlapping_and_other_items(tmp_path):
    base = _base_spec()
    keys = overlap.train_keys([_item(base)])
    other = "해결" if base.resolution != "해결" else "이관"
    dev = [
        _item(_copy(base, "dev", 0)),
        _item(_copy(base, "dev", 1, resolution=other)),
        _item(_copy(base, "dev", 2, resolution=other)),
    ]
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict() for it in dev])
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(data).replace("/", "\\")}), encoding="utf-8")
    exact = [True, False, True]
    rows = [{"item_id": it.item_id, "score": {"exact": e}} for it, e in zip(dev, exact, strict=True)]
    write_jsonl(run / "items.jsonl", rows)

    got = overlap.run_report(run, keys)
    assert (got["n"], got["overlap"], got["rest"]) == (3, [1, 1], [1, 2])
    assert got["run"] == "run"

    write_jsonl(run / "items.jsonl", [{"item_id": "dev-parcel-99999", "score": {"exact": True}}])
    with pytest.raises(ValueError, match="dev-parcel-99999"):
        overlap.run_report(run, keys)


# 명령줄: 분할 표(건수, 겹친 건 수와 비율, 업종/유형별 수)와 실행 표(겹친 건·나머지 exact)를 찍는다.
# --show는 겹친 건 id도 찍는다. 분할도 실행도 주지 않으면 사용법 오류다.
def test_main_prints_split_and_run_tables(tmp_path, capsys):
    base = _base_spec()
    other = "해결" if base.resolution != "해결" else "이관"
    train_file = tmp_path / "train.jsonl"
    write_jsonl(train_file, [_item(base).to_dict()])
    dev = [_item(_copy(base, "dev", 0)), _item(_copy(base, "dev", 1, resolution=other))]
    dev_file = tmp_path / "dev.jsonl"
    write_jsonl(dev_file, [it.to_dict() for it in dev])
    run = tmp_path / "run-a"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"data": str(dev_file)}), encoding="utf-8")
    write_jsonl(run / "items.jsonl", [{"item_id": it.item_id, "score": {"exact": True}} for it in dev])

    assert overlap.main([str(dev_file), "--train", str(train_file), "--run", str(run), "--show"]) == 0
    out = capsys.readouterr().out
    assert "1건, 서로 다른 정답 기록 1개" in out
    assert f"| {dev_file} | 2 | 1 (50.0%) | parcel/파손 신고 1 |" in out
    assert f"| run-a | {dev_file.as_posix()} | 1/1 (100.0%) | 1/1 (100.0%) |" in out
    assert dev[0].item_id in out and dev[1].item_id not in out

    with pytest.raises(SystemExit):
        overlap.main(["--train", str(train_file)])


# 공개 문서(docs/experiments.md 1단계 "생성 결과와 확정"의 2026-10-29 덧붙임)의 숫자를 저장소의 데이터와
# 기록된 실행으로 다시 낸다: 정답 기록이 train과 같은 건 dev 18, test-a 23, test-b 2, test-c 0(모두 번호가
# 없는 통신 유형), 3단계 학습한 학생의 exact(겹친 건 / 나머지) dev 17/18 · 207/222, test-a 22/23 · 265/277,
# test-b 2/2 · 52/58, 서빙 모델(q8_0) dev 18/18 · 209/222.
def test_documented_numbers_match_the_committed_datasets_and_runs():
    keys = overlap.train_keys(load_items(ROOT / "datasets" / "train.jsonl"))
    reports = {
        s: overlap.split_report(load_items(ROOT / "datasets" / f"{s}.jsonl"), keys)
        for s in ("dev", "test-a", "test-b", "test-c")
    }
    overlaps = {s: r["overlap"] for s, r in reports.items()}
    assert overlaps == {"dev": 18, "test-a": 23, "test-b": 2, "test-c": 0}
    assert reports["dev"]["by_category"] == {
        "telecom/분실·정지": 8,
        "telecom/요금제 변경": 8,
        "telecom/데이터 소진": 2,
    }
    assert reports["test-a"]["by_category"] == {
        "telecom/요금제 변경": 10,
        "telecom/분실·정지": 8,
        "telecom/데이터 소진": 5,
    }
    assert reports["test-b"]["by_category"] == {"telecom/요금제 변경": 2}

    runs = {
        "20260928-210805-Qwen_Qwen3-4B-s3-qwen3-4b-qlora-e2": ("dev", [17, 18], [207, 222]),
        "20260929-002710-Qwen_Qwen3-4B-s3-test-a-qwen3-4b-qlora-e2": ("test-a", [22, 23], [265, 277]),
        "20260929-011326-Qwen_Qwen3-4B-s3-test-b-qwen3-4b-qlora-e2": ("test-b", [2, 2], [52, 58]),
        "20260930-022406-call-summary-4b-dq_q8_0-s5b-dev-dq-q8_0": ("dev", [18, 18], [209, 222]),
    }
    for name, (split, hit, rest) in runs.items():
        got = overlap.run_report(ROOT / "reports" / name, keys, data=ROOT / "datasets" / f"{split}.jsonl")
        assert (got["overlap"], got["rest"]) == (hit, rest), name
