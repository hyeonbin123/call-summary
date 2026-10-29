# [테스트 묶음] dev 남은 오류의 정답 잡음 감사 도구(call_summary.label_audit)를 검증한다. (1) 서로 다른 학습
# 묶음 사이의 일치만 합의로 세는지(같은 묶음 안의 실행끼리 일치는 세지 않음) (2) 요약, 핵심 값의 순서와 말투,
# 처리·후속 조치 코드의 순서를 무시하고 채점과 같은 기준으로 같은 기록을 가리는지 (3) 서빙 모델 오류와 합의
# 후보의 합집합을 데이터 순서로 검토 대상으로 삼는지 (4) 검토 양식에 모델 답이 들어가지 않는지 (5) 입력이
# 어긋나면 조용히 세지 않고 멈추는지 (6) 채운 양식을 가장 나쁜 필드 기준으로 세어 결론 문장을 내는지.
# [왜 중요한가] 검토 대상과 분류 규칙은 검토 전에 docs/experiments.md "dev 남은 오류의 정답 잡음 감사"에
# 고정했다. 같은 설정으로 다시 학습했거나 실행기만 바꾼 실행끼리의 일치까지 합의로 세면 후보가 부풀고, 양식에
# 모델 답이 보이면 판정자가 대화를 읽기 전에 답에 끌린다.

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from call_summary import label_audit
from call_summary.dataset import Item, read_jsonl, write_jsonl
from call_summary.schema import parse_reply
from call_summary.scoring import score_item
from call_summary.specs import make_spec
from call_summary.values import normalize

ROOT = Path(__file__).resolve().parents[1]
# 등록한 대상(docs/experiments.md "dev 남은 오류의 정답 잡음 감사")의 학습 묶음과 실행, 커밋한 대상 목록·검토
# 양식.
REGISTERED_GROUPS = {
    "clean4b": [
        "20260928-210805-Qwen_Qwen3-4B-s3-qwen3-4b-qlora-e2",
        "20261014-172808-Qwen_Qwen3-4B-fix-s3-qwen3-4b-qlora-final",
    ],
    "mixed4b": [
        "20260930-013127-Qwen_Qwen3-4B-s5b-dev-hf4bit-mixed",
        "20261015-222131-Qwen_Qwen3-4B-fix-s5-dev-hf4bit-mixed",
        "20260930-022406-call-summary-4b-dq_q8_0-s5b-dev-dq-q8_0",
    ],
    "lora17b": ["20260928-171039-Qwen_Qwen3-1.7B-s3-qwen3-1.7b-lora-e2"],
}
SERVING_RUN = "20260930-022406-call-summary-4b-dq_q8_0-s5b-dev-dq-q8_0"
AUDIT_FILE = ROOT / "reports" / "label_audit" / "dev-20261029.json"
REVIEW_FILE = ROOT / "reports" / "label_audit" / "dev-20261029-review.jsonl"


# 금액 슬롯(말투를 바꿔 볼 값)과 처리 2개·후속 조치 코드 2개(순서를 바꿔 볼 목록)가 있는 택배 파손 신고 명세의
# 첫 번호를 고른다. make_spec이 결정적이라 늘 같은 명세가 나온다.
def _base_spec():
    return next(
        s
        for s in (make_spec("dev", "parcel", i, category="파손 신고") for i in range(50))
        if any(sv.type == "금액" for sv in s.slots) and len(s.actions) >= 2 and len(s.follow_up) >= 2
    )


# dev 데이터셋 n건을 만들어 파일로 쓴다. 명세 내용은 같고 번호만 다르다. 상담원 인사 뒤에 고객이 핵심 값을
# 하나씩 말하게 해, 검토 양식의 관련 발화 번호(값이 나온 발화)를 확인할 수 있게 한다.
def _dataset(tmp_path: Path, n: int) -> tuple[list[Item], Path]:
    base = _base_spec()
    items = []
    for i in range(n):
        spec = dataclasses.replace(base, spec_id=f"dev-parcel-{i:05d}")
        turns = [{"speaker": "상담원", "text": "무엇을 도와드릴까요?"}]
        turns += [{"speaker": "고객", "text": f"{sv.type}는 {sv.value}예요."} for sv in spec.slots]
        items.append(Item(spec.spec_id, "dev", "parcel", spec, turns=turns, summary="참고 요약"))
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [it.to_dict() for it in items])
    return items, data


# 모델 답(기록 dict, 요약 포함). 정답 기록에서 시작해 changes로 필드를 바꿔 틀린 답을 만든다.
def _answer(item: Item, **changes) -> dict:
    record = item.gold().model_dump()
    record["summary"] = "모델 요약"
    record.update(changes)
    return record


# 정답과 다른 처리 결과. 틀린 답을 만드는 가장 간단한 방법이다.
def _other_resolution(item: Item) -> str:
    return "해결" if item.spec.resolution != "해결" else "이관"


# 기록된 평가 실행 하나를 흉내 낸다: manifest(데이터 경로는 Windows에서 쓴 기록처럼 역슬래시 꼴, data_sha는
# evaluate와 같은 sha256 앞 12자리)와 items.jsonl(pred와 채점기 score_item으로 매긴 score). answers에 없는
# 건은 정답(요약만 다름)을, None이면 형식 실패를 낸 것으로 친다.
def _run(root: Path, name: str, data: Path, items: list[Item], answers: dict[str, dict | None]) -> Path:
    run = root / name
    run.mkdir()
    sha = hashlib.sha256(data.read_bytes()).hexdigest()[:12]
    manifest = {"data": str(data).replace("/", "\\"), "data_sha": sha}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = []
    for it in items:
        pred = answers[it.item_id] if it.item_id in answers else _answer(it)
        reply = json.dumps(pred, ensure_ascii=False) if pred is not None else "답 없음"
        score = score_item(it.item_id, it.domain, it.gold(), parse_reply(reply), it.transcript)
        rows.append({"item_id": it.item_id, "pred": pred, "score": score.to_dict()})
    write_jsonl(run / "items.jsonl", rows)
    return run


def _target(audit: dict, item_id: str) -> dict:
    return next(t for t in audit["targets"] if t["item_id"] == item_id)


# 같은 묶음의 두 실행이 같은 틀린 기록을 내도 합의가 아니다(같은 설정으로 다시 학습했거나 실행기만 바꾼 실행).
# 서로 다른 두 묶음이 같은 틀린 기록을 낸 건만 후보다. 두 묶음이 서로 다른 틀린 기록을 낸 건도 후보가 아니다.
def test_only_agreement_between_groups_counts(tmp_path):
    items, data = _dataset(tmp_path, 3)
    a, b, c = items
    wrong = {it.item_id: _answer(it, resolution=_other_resolution(it)) for it in items}
    other_c = _answer(c, actions_taken=list(c.spec.actions[:1]))
    a1 = _run(tmp_path, "a1", data, items, dict(wrong))
    a2 = _run(tmp_path, "a2", data, items, {a.item_id: wrong[a.item_id]})
    b1 = _run(tmp_path, "b1", data, items, {b.item_id: wrong[b.item_id], c.item_id: other_c})

    audit = label_audit.build_audit({"A": [a1, a2], "B": [b1]}, errors_of=a2)
    assert [(t["item_id"], t["reasons"]) for t in audit["targets"]] == [
        (a.item_id, ["serving_error"]),
        (b.item_id, ["cross_group"]),
    ]
    agreed = [ans for ans in _target(audit, b.item_id)["answers"] if ans["agreed"]]
    assert [(ans["groups"], ans["runs"]) for ans in agreed] == [(["A", "B"], ["a1", "b1"])]
    assert agreed[0]["diff"] == {"resolution": {"gold": b.spec.resolution, "pred": _other_resolution(b)}}
    # a: 묶음 A의 두 실행만 같은 답을 냈다. 그 답은 대상 목록에 남지만 합의된 답이 아니다.
    assert [(ans["groups"], ans["runs"], ans["agreed"]) for ans in _target(audit, a.item_id)["answers"]] == [
        (["A"], ["a1", "a2"], False)
    ]
    assert audit["counts"]["cross_group"] == 1


# 같은 기록인지는 채점(scoring.score_item)과 같은 기준으로 가린다: 요약은 보지 않고, 핵심 값은 (유형, 정규화
# 값) 이라 "10만 1천 원"과 "101,000원"이 같고, 핵심 값·처리·후속 조치 코드의 순서는 보지 않는다. 요약만 다른
# 답은 틀린 답이 아니다.
def test_same_record_ignores_summary_order_and_value_wording(tmp_path):
    items, data = _dataset(tmp_path, 1)
    (a,) = items
    gold = a.gold()
    amount = next(e.value for e in gold.entities if e.type == "금액")
    n = int(normalize("amount", amount))
    reworded = f"{n:,}원" if "," not in amount else f"{n}원"
    assert reworded != amount and normalize("amount", reworded) == normalize("amount", amount)
    other = _other_resolution(a)
    first = _answer(a, resolution=other, summary="첫 요약")
    second = _answer(
        a,
        resolution=other,
        summary="전혀 다른 요약",
        entities=[
            {"type": e.type, "value": reworded if e.type == "금액" else e.value}
            for e in reversed(gold.entities)
        ],
        actions_taken=list(reversed(gold.actions_taken)),
        follow_up={"required": True, "codes": list(reversed(gold.follow_up.codes))},
    )
    assert label_audit.structure("parcel", first) == label_audit.structure("parcel", second)
    a1 = _run(tmp_path, "a1", data, items, {a.item_id: first})
    b1 = _run(tmp_path, "b1", data, items, {a.item_id: second})
    c1 = _run(tmp_path, "c1", data, items, {a.item_id: _answer(a, summary="요약만 다름")})

    audit = label_audit.build_audit({"A": [a1], "B": [b1], "C": [c1]}, errors_of=c1)
    (t,) = audit["targets"]
    assert t["reasons"] == ["cross_group"] and t["fields"] == ["resolution"]
    assert [(ans["groups"], ans["agreed"]) for ans in t["answers"]] == [(["A", "B"], True)]
    assert audit["counts"]["serving_errors"] == 0 and audit["errors_of_exact"] == [1, 1]


# 검토 대상은 서빙 모델이 틀린 건과 합의 후보의 합집합이고 데이터 순서를 따른다. 서빙 모델 실행이 한 묶음에
# 들어 있어도 된다(등록한 대상에서도 섞어 학습한 4B 묶음에 든다). 서빙 모델이 맞힌 후보는 서빙 칸이 비어 있다.
def test_targets_are_the_union_of_serving_errors_and_candidates(tmp_path):
    items, data = _dataset(tmp_path, 4)
    ids = [it.item_id for it in items]
    wrong = {it.item_id: _answer(it, resolution=_other_resolution(it)) for it in items}
    serving = _run(tmp_path, "serving", data, items, {ids[1]: wrong[ids[1]], ids[3]: wrong[ids[3]]})
    a2 = _run(tmp_path, "a2", data, items, {ids[2]: wrong[ids[2]]})
    b1 = _run(tmp_path, "b1", data, items, {ids[1]: wrong[ids[1]], ids[2]: wrong[ids[2]]})

    audit = label_audit.build_audit({"A": [serving, a2], "B": [b1]}, errors_of=serving)
    assert [(t["item_id"], t["reasons"]) for t in audit["targets"]] == [
        (ids[1], ["serving_error", "cross_group"]),
        (ids[2], ["cross_group"]),
        (ids[3], ["serving_error"]),
    ]
    assert audit["counts"] == {
        "serving_errors": 2,
        "cross_group": 2,
        "all_groups": 2,
        "both": 1,
        "targets": 3,
    }
    assert audit["groups"] == {"A": ["serving", "a2"], "B": ["b1"]}
    assert (audit["errors_of"], audit["errors_of_exact"], audit["n_items"]) == ("serving", [2, 4], 4)
    assert _target(audit, ids[2])["serving"] is None
    item3 = items[3]
    assert _target(audit, ids[3])["serving"]["diff"] == {
        "resolution": {"gold": item3.spec.resolution, "pred": _other_resolution(item3)}
    }


# 검토 양식에는 명세, 요약을 뺀 정답 기록, 번호 붙은 발화, 정답 핵심 값이 나온 발화 번호, 차이 필드 이름과 빈
# 판정 칸만 있고 모델 답은 없다. 모델 답(여기서는 목록 밖 처리 "가짜 처리")은 대상 목록에만 있다.
def test_review_template_has_no_model_answers(tmp_path):
    items, data = _dataset(tmp_path, 1)
    (a,) = items
    fake = _answer(a, actions_taken=[*a.spec.actions, "가짜 처리"])
    a1 = _run(tmp_path, "a1", data, items, {a.item_id: fake})
    b1 = _run(tmp_path, "b1", data, items, {a.item_id: fake})

    audit = label_audit.build_audit({"A": [a1], "B": [b1]}, errors_of=a1)
    (row,) = label_audit.review_template(audit)
    assert "가짜 처리" in json.dumps(audit, ensure_ascii=False)
    assert "가짜 처리" not in json.dumps(row, ensure_ascii=False)
    assert row["fields"] == ["actions_taken"]
    assert row["review"] == {"actions_taken": {"reading": None, "class": None, "turns": [], "note": ""}}
    gold = a.gold().model_dump()
    assert row["gold"] == {k: v for k, v in gold.items() if k != "summary"}
    assert row["spec"] == json.loads(json.dumps(a.spec.to_dict(), ensure_ascii=False))
    first = a.spec.slots[0]
    assert row["turns"][:2] == [
        "1 상담원: 무엇을 도와드릴까요?",
        f"2 고객: {first.type}는 {first.value}예요.",
    ]
    assert row["value_turns"][f"{first.type} {first.value}"] == [2]


# 입력이 어긋나면 조용히 세지 않고 멈춘다: 묶음이 하나뿐(묶음 안의 일치는 합의가 아니다), 같은 실행이 두
# 묶음에, --min-groups가 묶음 수 밖, 기록된 exact가 이 도구의 같음 기준과 어긋남, 데이터셋에 없는 건, 실행의
# data_sha가 지금 데이터 파일과 다름(데이터가 바뀌었으면 정답도 다를 수 있다).
def test_rejects_inconsistent_inputs(tmp_path):
    items, data = _dataset(tmp_path, 2)
    a1 = _run(tmp_path, "a1", data, items, {})
    b1 = _run(tmp_path, "b1", data, items, {})
    with pytest.raises(ValueError, match="two groups"):
        label_audit.build_audit({"A": [a1]}, errors_of=a1)
    with pytest.raises(ValueError, match="a1"):
        label_audit.build_audit({"A": [a1], "B": [a1, b1]}, errors_of=a1)
    with pytest.raises(ValueError, match="min-groups"):
        label_audit.build_audit({"A": [a1], "B": [b1]}, errors_of=a1, min_groups=3)

    tampered = _run(tmp_path, "tampered", data, items, {})
    rows = list(read_jsonl(tampered / "items.jsonl"))
    rows[0]["score"]["exact"] = False
    write_jsonl(tampered / "items.jsonl", rows)
    with pytest.raises(ValueError, match=items[0].item_id):
        label_audit.build_audit({"A": [a1], "B": [tampered]}, errors_of=a1)

    stranger = _run(tmp_path, "stranger", data, items, {})
    rows = list(read_jsonl(stranger / "items.jsonl"))
    rows[1]["item_id"] = "dev-parcel-99999"
    write_jsonl(stranger / "items.jsonl", rows)
    with pytest.raises(ValueError, match="dev-parcel-99999"):
        label_audit.build_audit({"A": [a1], "B": [stranger]}, errors_of=a1)

    old = _run(tmp_path, "old", data, items, {})
    manifest = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    manifest["data_sha"] = "000000000000"
    (old / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="data_sha"):
        label_audit.build_audit({"A": [a1], "B": [b1]}, errors_of=old)


# 채운 양식을 센다. 필드 분류는 정답 필드에 대한 판단이고, 답 하나의 분류는 그 답이 정답과 다른 필드 가운데
# 가장 나쁜 것(모델 오류 > 애매 > 정답 잡음)이다. 검토 대상 한 건의 분류는 서빙 모델이 틀린 건이면 서빙 모델
# 답, 아니면 합의된 답으로 낸다. 가설 지표(후보 가운데 정답 잡음 비율)는 합의된 답의 분류로 센다. 1단계 판단의
# 기본 대응(어긋남 → 정답 잡음, 뒷받침 → 모델 오류, 애매 → 애매)과 다르게 분류한 필드는 바꾼 필드로 센다.
def test_tally_takes_the_worst_field_of_each_answer(tmp_path):
    items, data = _dataset(tmp_path, 4)
    i0, i1, i2, _ = items
    without_amount = [e for e in i0.gold().model_dump()["entities"] if e["type"] != "금액"]
    serving_i0 = _answer(i0, entities=without_amount)
    agreed_i0 = _answer(i0, entities=without_amount, actions_taken=list(i0.spec.actions[:1]))
    agreed_i1 = _answer(i1, follow_up={"required": False, "codes": []})
    serving_i2 = _answer(i2, category="배송 조회")
    serving = _run(tmp_path, "serving", data, items, {i0.item_id: serving_i0, i2.item_id: serving_i2})
    a2 = _run(tmp_path, "a2", data, items, {i0.item_id: agreed_i0, i1.item_id: agreed_i1})
    b1 = _run(tmp_path, "b1", data, items, {i0.item_id: agreed_i0, i1.item_id: agreed_i1})
    audit = label_audit.build_audit({"A": [serving, a2], "B": [b1]}, errors_of=serving)
    assert _target(audit, i0.item_id)["fields"] == ["entities", "actions_taken"]

    rows = label_audit.review_template(audit)
    filled = {
        (i0.item_id, "entities"): ("어긋남", "정답 잡음"),
        (i0.item_id, "actions_taken"): ("뒷받침", "모델 오류"),
        (i1.item_id, "follow_up"): ("뒷받침", "애매"),
        (i2.item_id, "category"): ("어긋남", "정답 잡음"),
    }
    for row in rows:
        for field, cell in row["review"].items():
            cell["reading"], cell["class"] = filled[(row["item_id"], field)]

    got = label_audit.tally(audit, rows)
    assert got["labels"] == {"정답 잡음": 2, "모델 오류": 0, "애매": 1}
    assert got["serving_labels"] == {"정답 잡음": 2, "모델 오류": 0, "애매": 0}
    assert got["candidate_labels"] == {"정답 잡음": 0, "모델 오류": 1, "애매": 1}
    assert (got["candidate_noise_share"], got["hypothesis_met"]) == (0.0, False)
    assert (got["fields"], got["changed"]) == (4, 1)
    assert got["exact"] == [2, 4] and got["ceiling"] == pytest.approx(1.0)
    assert [(it["item_id"], it["label"], it["serving"], it["agreed"]) for it in got["items"]] == [
        (i0.item_id, "정답 잡음", "정답 잡음", "모델 오류"),
        (i1.item_id, "애매", None, "애매"),
        (i2.item_id, "정답 잡음", "정답 잡음", None),
    ]
    text = label_audit.conclusion(got)
    assert text.splitlines()[0] == (
        "검토 대상 3건 중 정답 잡음 2, 모델 오류 0, 애매 1. 서빙 모델 dev 오류 2건 중 2건이 정답 잡음이다."
    )
    assert "교차 묶음 후보 2건 중 정답 잡음 0" in text and "가설(절반 이상) 맞지 않음" in text
    assert "점 추정 100.0% (기록 50.0%, 2/4)" in text


# 채우지 않았거나 정한 말이 아닌 판정, 대상과 맞지 않는 양식은 세지 않고 멈춘다(빠진 판정을 조용히 건너뛰면
# 비율이 틀린다).
def test_tally_rejects_an_unfinished_review(tmp_path):
    items, data = _dataset(tmp_path, 1)
    (a,) = items
    wrong = _answer(a, resolution=_other_resolution(a))
    a1 = _run(tmp_path, "a1", data, items, {a.item_id: wrong})
    b1 = _run(tmp_path, "b1", data, items, {a.item_id: wrong})
    audit = label_audit.build_audit({"A": [a1], "B": [b1]}, errors_of=a1)
    rows = label_audit.review_template(audit)
    with pytest.raises(ValueError, match=f"{a.item_id} resolution"):
        label_audit.tally(audit, rows)
    rows[0]["review"]["resolution"].update({"reading": "뒷받침", "class": "틀림"})
    with pytest.raises(ValueError, match=f"{a.item_id} resolution"):
        label_audit.tally(audit, rows)
    rows[0]["review"]["resolution"]["class"] = "모델 오류"
    assert label_audit.tally(audit, rows)["labels"]["모델 오류"] == 1
    with pytest.raises(ValueError, match="missing"):
        label_audit.tally(audit, [])


# 명령줄: build가 대상 목록(JSON)과 검토 양식(JSONL)을 쓰고 대상 id와 차이 필드 이름만 찍는다(모델 답은 찍지
# 않는다). --group의 쉼표로 실행 여럿을 한 묶음에 넣는다. tally가 채운 양식으로 결론 문장을 찍는다. --group
# 꼴이 틀리면 사용법 오류다.
def test_main_builds_the_files_and_tallies(tmp_path, capsys):
    items, data = _dataset(tmp_path, 2)
    a = items[0]
    wrong = _answer(a, resolution=_other_resolution(a))
    a1 = _run(tmp_path, "a1", data, items, {a.item_id: wrong})
    a2 = _run(tmp_path, "a2", data, items, {})
    b1 = _run(tmp_path, "b1", data, items, {a.item_id: wrong})
    out = tmp_path / "audit" / "dev.json"
    template = tmp_path / "audit" / "dev-review.jsonl"
    argv = ["build", "--group", f"A={a1},{a2}", "--group", f"B={b1}", "--errors-of", str(a2)]
    assert label_audit.main([*argv, "--out", str(out), "--template", str(template)]) == 0
    printed = capsys.readouterr().out
    assert "검토 대상 1" in printed and f"{a.item_id}  cross_group  resolution" in printed
    assert _other_resolution(a) not in printed.split("검토 대상 1")[1]
    audit = json.loads(out.read_text(encoding="utf-8"))
    assert audit["groups"] == {"A": ["a1", "a2"], "B": ["b1"]}

    rows = list(read_jsonl(template))
    rows[0]["review"]["resolution"].update({"reading": "뒷받침", "class": "모델 오류"})
    write_jsonl(template, rows)
    assert label_audit.main(["tally", "--audit", str(out), "--review", str(template)]) == 0
    assert "검토 대상 1건 중 정답 잡음 0, 모델 오류 1, 애매 0." in capsys.readouterr().out

    with pytest.raises(SystemExit):
        label_audit.main(
            ["build", "--group", "A", "--errors-of", str(a1), "--out", str(out), "--template", "x"]
        )


# 등록한 대상을 저장소의 데이터와 실행 기록으로 다시 낸다: 서빙 모델 오류 13 (exact 227/240), 교차 묶음 후보 9
# (세 묶음이 모두 같은 기록을 낸 것 1건, dev-telecom-00088), 둘 다인 것 8, 검토 대상 14. 실행 6개의 기록된
# exact가 이 도구의 같음 기준과 모두 맞는다(어긋나면 build_audit가 멈춘다). 커밋한 대상 목록은 다시 만든 것과
# 같고, 커밋한 검토 양식은 판정 칸 말고는 다시 만든 양식과 같다(판정 칸은 채워져 있을 수 있다).
def test_registered_targets_match_the_committed_runs(monkeypatch):
    monkeypatch.chdir(ROOT)  # manifest의 데이터 경로(datasets/dev.jsonl)는 저장소 기준 상대 경로다
    groups = {g: [ROOT / "reports" / r for r in runs] for g, runs in REGISTERED_GROUPS.items()}
    audit = label_audit.build_audit(groups, errors_of=ROOT / "reports" / SERVING_RUN)
    assert audit["counts"] == {
        "serving_errors": 13,
        "cross_group": 9,
        "all_groups": 1,
        "both": 8,
        "targets": 14,
    }
    assert audit["errors_of_exact"] == [227, 240] and audit["data_sha"] == "cb64eb0a5431"
    all_three = [t["item_id"] for t in audit["targets"] if any(len(a["groups"]) == 3 for a in t["answers"])]
    assert all_three == ["dev-telecom-00088"]
    assert [t["item_id"] for t in audit["targets"] if t["reasons"] == ["cross_group"]] == ["dev-shop-00048"]
    assert json.loads(AUDIT_FILE.read_text(encoding="utf-8")) == json.loads(
        json.dumps(audit, ensure_ascii=False)
    )

    fresh = label_audit.review_template(audit)
    committed = list(read_jsonl(REVIEW_FILE))
    assert len(committed) == len(fresh) == 14
    for got, want in zip(committed, fresh, strict=True):
        assert {k: v for k, v in got.items() if k != "review"} == {
            k: v for k, v in want.items() if k != "review"
        }
        assert list(got["review"]) == want["fields"]
