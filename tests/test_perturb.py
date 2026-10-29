# [테스트 묶음] 반사실·메타모픽 변형 도구(call_summary.perturb)를 검증한다. (1) 같은 seed면 바이트까지 같은
# 파일이 나오는지 (2) swap이 번호 값을 전사·명세·요약에서 함께 바꾸고, 새 값이 옛 값의 겉 꼴과 배포 형식을
# 지키며 원래 글의 다른 숫자와 겹치지 않는지, 말실수 값도 같은 자리 숫자 하나만 다르게 함께 바뀌는지 (3)
# neutral·inject가 정답을 그대로 두고 정해 둔 발화만 화자 교대를 깨지 않게 끼우는지, 끼운 뒤에도 생성
# 검사(check_dialogue)를 통과하는지 (4) 건 id와 순서, 원래 meta가 남는지 (5) report가 대조 실행과 변형
# 실행에서 "따라 쓴 비율"·"주입 따름"·exact 짝지은 차이를 규칙대로 세고, 조건이 다른 두 실행은 비교하지
# 않는지.
# [왜 중요한가] 이 변형 파일로 서빙 모델의 강건성을 잰다(docs/experiments.md "반사실·메타모픽 강건성"). 변형이
# 정답을 어긋나게 만들거나(옛 번호가 전사에 남음, 끼운 발화가 명세 밖 값을 담음) 실행마다 다르게 나오면 측정
# 차이가 모델이 아니라 도구에서 생긴다.

import copy
import dataclasses
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from call_summary import perturb
from call_summary.dataset import Item, load_items, read_jsonl, write_jsonl
from call_summary.domains import DOMAINS
from call_summary.gen import check_dialogue, check_summary
from call_summary.schema import RESOLUTIONS, parse_reply
from call_summary.scoring import METRICS, ItemScore, paired_bootstrap_diff, score_item
from call_summary.values import normalize, values_in_transcript

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "datasets" / "dev.jsonl"
# 단위 시험에서 쓰는 원본 지문(실제 파일 지문이 필요 없는 시험).
SHA = "0" * 12


@pytest.fixture(scope="module")
def dev_items() -> list[Item]:
    return load_items(DEV)


# dev 240건 전체로 세 변형을 한 번씩 만든다(seed 0). make_variant는 검사를 통과하지 못한 건이 있으면 멈추므로,
# 이 fixture가 만들어진 것 자체가 240건 모두 검사를 통과했다는 뜻이다.
@pytest.fixture(scope="module")
def variants(dev_items) -> dict[str, tuple[list[Item], dict]]:
    return {v: perturb.make_variant(dev_items, v, seed=0, source_sha=SHA) for v in perturb.VARIANTS}


def _id_slots(item: Item) -> list:
    d = DOMAINS[item.domain]
    return [sv for sv in item.spec.slots if d.entity_type(sv.type).kind == "id"]


def _text(item: Item) -> str:
    return "\n".join(t["text"] for t in item.turns)


# 숫자 덩어리(빈칸 하나나 하이픈 하나로 이어진 숫자)와 숫자 4개 묶음. 도구와 따로 적어 같은 기준인지 확인한다.
def _runs(text: str) -> list[str]:
    return [re.sub(r"[^0-9]", "", m.group(0)) for m in re.finditer(r"[0-9]+(?:[ \-][0-9]+)*", text)]


def _windows(value: str) -> set[str]:
    digits = re.sub(r"[^0-9]", "", value)
    return {digits[i : i + 4] for i in range(len(digits) - 3)} if len(digits) > 4 else {digits}


# 끝 숫자를 읽은 소리의 받침(0 공·1 일·3 삼·6 육·7 칠·8 팔은 받침, 2 이·4 사·5 오·9 구는 받침 없음)과, 받침에
# 따라 꼴이 갈리는 조사. "으로/로"는 받침이 ㄹ이거나 없으면 "로"다. 도구와 따로 적어 같은 규칙인지 확인한다.
_FINAL = {
    "0": True,
    "1": True,
    "2": False,
    "3": True,
    "4": False,
    "5": False,
    "6": True,
    "7": True,
    "8": True,
    "9": False,
}
_AFTER_FINAL = {"이", "을", "은", "과", "이에요", "이라는", "이라고", "이야", "이요"}
_AFTER_VOWEL = {"가", "를", "는", "와", "예요", "라는", "라고", "야", "요"}
# 서술격 조사 "이고·이었"은 받침이 없는 말 뒤에서도 맞고, 줄어든 "고·였"은 받침 없는 말 뒤에서만 맞다.
_LOOSE_FULL = {"이고", "이었"}
_LOOSE_SHORT = {"고", "였"}
_PARTICLE = (
    "(이에요|이라는|이라고|으로|이었|예요|라는|라고|로|였"
    "|(?:이야|이요|이고|이|가|을|를|은|는|과|와|야|요|고)(?![가-힣]))"
)


# 번호 바로 뒤 조사가 그 번호의 끝소리와 맞는가(조사가 아니면 None).
def _agrees(value: str, particle: str) -> bool | None:
    last = value[-1]
    if particle in ("으로", "로"):
        return particle == ("으로" if last in "036" else "로")
    if particle in _LOOSE_FULL:
        return True
    if particle in _LOOSE_SHORT:
        return not _FINAL[last]
    if particle in _AFTER_FINAL:
        return _FINAL[last]
    if particle in _AFTER_VOWEL:
        return not _FINAL[last]
    return None


# 번호와 그 바로 뒤 조사를 "#"로 가린 글. 바꾸기 전과 뒤의 가린 글이 같으면 번호와 조사 말고는 바뀌지 않은
# 것이다.
def _masked(text: str, values: list[str]) -> str:
    for v in values:
        text = re.sub(re.escape(v) + _PARTICLE + "?", "#", text)
    return text


def _same_in_row(turns: list[dict]) -> int:
    return sum(1 for a, b in zip(turns, turns[1:], strict=False) if a["speaker"] == b["speaker"])


# 같은 seed로 두 번 만든 파일은 바이트까지 같고, seed를 바꾸면 달라진다. 통계 파일의 지문은 실제 파일과
# 맞는다.
def test_same_seed_gives_identical_bytes(tmp_path):
    rows = list(read_jsonl(DEV))
    small = [r for dom in ("parcel", "shop", "telecom") for r in [x for x in rows if x["domain"] == dom][:12]]
    src = tmp_path / "dev.jsonl"
    write_jsonl(src, small)
    for v in perturb.VARIANTS:
        got = []
        for name, seed in (("a", 0), ("b", 0), ("c", 1)):
            out = tmp_path / f"{v}-{name}.jsonl"
            argv = ["make", "--data", str(src), "--variant", v, "--seed", str(seed), "--out", str(out)]
            assert perturb.main(argv) == 0
            stats = json.loads(out.with_suffix(".stats.json").read_text(encoding="utf-8"))
            assert stats["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
            assert stats["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
            assert (stats["variant"], stats["seed"], stats["n_items"]) == (v, seed, len(small))
            got.append(out.read_bytes())
        assert got[0] == got[1]
        assert got[0] != got[2]


# swap: 번호 값이 전사·요약·사실에서 모두 새 값으로 바뀌고 정답 기록의 핵심 값도 새 값이다. 새 값은 옛 값의 겉
# 꼴(숫자가 아닌 글자와 그 자리)을 지키고, 정규화하면 업종의 배포 형식(id_format)에 맞으며, 원래 글의 어느
# 숫자 덩어리와도 숫자 4개가 겹치지 않는다. 번호 말고는 바뀌지 않고, 바꾼 뒤에도 생성 검사 둘을 통과한다.
def test_swap_replaces_each_number_everywhere(dev_items, variants):
    out, stats = variants["swap"]
    changed = Counter()
    for old, new in zip(dev_items, out, strict=True):
        assert (new.item_id, new.split, new.domain, new.source) == (
            old.item_id,
            old.split,
            old.domain,
            old.source,
        )
        info = new.meta["perturb"]
        slots = _id_slots(old)
        assert [(v["slot"], v["type"], v["old"]) for v in info["values"]] == [
            (sv.slot, sv.type, sv.value) for sv in slots
        ]
        if not slots:
            continue
        d = DOMAINS[old.domain]
        original = "\n".join([_text(old), old.summary or "", *old.spec.facts])
        taken = _runs(original)
        new_text = _text(new)
        id_types = {sv.type for sv in slots}
        gold_ids = {(e.type, normalize("id", e.value)) for e in new.gold().entities if e.type in id_types}
        for v in info["values"]:
            changed[v["type"]] += 1
            assert re.sub(r"[0-9]", "#", v["new"]) == re.sub(r"[0-9]", "#", v["old"])
            assert re.fullmatch(d.entity_type(v["type"]).id_format, normalize("id", v["new"]))
            assert not any(w in r for w in _windows(v["new"]) for r in taken)
            assert v["new"] in new_text and v["new"] in new.summary
            for where in (new_text, new.summary, "\n".join(new.spec.facts)):
                assert normalize("id", v["old"]) not in normalize("id", where)
        assert gold_ids == {(v["type"], normalize("id", v["new"])) for v in info["values"]}
        # 전사·요약·사실의 줄마다 번호와 그 뒤 조사 말고는 같고, 조사가 번호의 끝소리와 맞는지(맞음·틀림)는
        # 바꾸기 전과 같은 차례로 남는다. 다만 받침 뒤에 줄어든 서술격 조사(틀림)는 새 번호가 받침 없이 끝나면
        # 저절로 맞게 된다.
        pairs = [(v["old"], v["new"]) for v in info["values"]]
        if info["distractor"]:
            pairs.append((info["distractor"]["old"], info["distractor"]["new"]))
        for before, after in zip(
            original.split("\n"), "\n".join([new_text, new.summary, *new.spec.facts]).split("\n"), strict=True
        ):
            assert _masked(before, [o for o, _ in pairs]) == _masked(after, [n for _, n in pairs])
            for o, n in pairs:
                was = [m.group(1) for m in re.finditer(re.escape(o) + _PARTICLE, before)]
                now = [m.group(1) for m in re.finditer(re.escape(n) + _PARTICLE, after)]
                for p, q in zip(was, now, strict=True):
                    w, a = _agrees(o, p), _agrees(n, q)
                    assert a == w or (p == q and p in _LOOSE_SHORT and a and not w)
        assert [(sv.slot, sv.type) for sv in new.spec.slots] == [(sv.slot, sv.type) for sv in old.spec.slots]
        others_new = [sv for sv in new.spec.slots if d.entity_type(sv.type).kind != "id"]
        assert others_new == [sv for sv in old.spec.slots if d.entity_type(sv.type).kind != "id"]
        keep = ("category", "resolution", "actions", "follow_up", "events", "outcome_index", "note")
        assert {k: getattr(new.spec, k) for k in keep} == {k: getattr(old.spec, k) for k in keep}
        assert [t["speaker"] for t in new.turns] == [t["speaker"] for t in old.turns]
        assert check_dialogue(new.spec, new.turns) == []
        assert check_summary(new.spec, new.summary) == []
        assert info["tries"] >= 1
    assert sum(changed.values()) > 0
    assert stats["values"] == dict(changed)
    assert stats["changed_items"] == sum(1 for it in dev_items if _id_slots(it))


# 번호를 바꿀 때 바로 뒤의 조사는 새 끝 숫자의 소리에 맞춰 바꾼다. 원래 글이 맞았으면 맞게, 틀렸으면 틀린 채로
# 둔다(끝소리 갈래가 바뀔 때만 짝 꼴로 바꾼다). 조사가 아닌 글("이번", "의", "입니다")과 더 긴 번호의 일부는
# 건드리지 않는다. 돌려주는 것: (바꾼 글, 바꾼 번호 수, 바꾼 조사 수).
def test_swap_keeps_particles_in_agreement_with_the_new_number():
    sub = perturb._sub_values
    # 7(칠, 받침 ㄹ) → 9(구, 받침 없음): 받침 갈래가 바뀐다. "로"는 둘 다 "로"라 그대로다.
    text = "1237이 맞아요. 1237을 보냈고 1237은 됐어요. 1237과 같아요. 1237이에요. 1237로 조회"
    assert sub(text, [("1237", "4569")]) == (
        "4569가 맞아요. 4569를 보냈고 4569는 됐어요. 4569와 같아요. 4569예요. 4569로 조회",
        6,
        5,
    )
    # 3(삼, 받침 ㅁ) → 5(오): "으로"가 "로"로, "이"가 "가"로.
    assert sub("D1233으로 조회했고 D1233이 맞아요.", [("D1233", "D1235")]) == (
        "D1235로 조회했고 D1235가 맞아요.",
        2,
        2,
    )
    # 5(오) → 6(육, 받침 ㄱ): 받침 없는 꼴이 모두 받침 있는 꼴로.
    text = "D1235로 D1235가 D1235를 D1235는 D1235와 D1235예요 D1235라는 D1235였던"
    assert sub(text, [("D1235", "D1236")]) == (
        "D1236으로 D1236이 D1236을 D1236은 D1236과 D1236이에요 D1236이라는 D1236이었던",
        8,
        8,
    )
    # 같은 갈래(2 → 4)면 조사를 바꾸지 않고, 원래 틀린 조사는 틀린 채로 옮긴다(7에 "가" → 9에 "이").
    assert sub("1232가 1232로", [("1232", "1234")]) == ("1234가 1234로", 2, 0)
    assert sub("1237가 맞아요", [("1237", "4569")]) == ("4569이 맞아요", 1, 1)
    # 서술격 조사 "이고·이었"은 어느 끝소리 뒤에서나 맞아 그대로 두고(7 → 9), 줄어든 "고·였"은 받침 없는 말
    # 뒤에서만 맞아 새 번호가 받침으로 끝나면 줄지 않은 꼴로 바꾼다(5 → 7). 받침 뒤의 "고"는 원래 틀린 채로
    # 둔다.
    assert sub("1237이고, 1237이었어요", [("1237", "4569")]) == ("4569이고, 4569이었어요", 2, 0)
    assert sub("D1235고, D1235였어요", [("D1235", "D1237")]) == ("D1237이고, D1237이었어요", 2, 2)
    assert sub("D1231고, D1231였어요", [("D1231", "D1237")]) == ("D1237고, D1237였어요", 2, 0)
    # 조사가 아닌 글과 더 긴 번호 안의 같은 숫자는 그대로다.
    assert sub("1237이번에 1237의 1237입니다 1237번 11237이", [("1237", "4569")]) == (
        "4569이번에 4569의 4569입니다 4569번 11237이",
        4,
        0,
    )


# 말실수 값(고객이 먼저 말한 틀린 번호)도 함께 바뀐다: 새 말실수 값은 새 정답 번호와 원래와 같은 자리의 숫자
# 하나만 다르다. 전사에는 새 말실수 값이 있고 옛 말실수 값은 없으며, 요약에는 말실수 값이 없다.
def test_swap_moves_the_distractor_with_its_number(dev_items, variants):
    out, stats = variants["swap"]
    seen = 0
    for old, new in zip(dev_items, out, strict=True):
        if old.spec.distractor is None:
            assert new.meta["perturb"]["distractor"] is None
            continue
        seen += 1
        info = new.meta["perturb"]["distractor"]
        assert (info["slot"], info["type"], info["old"]) == (
            old.spec.distractor.slot,
            old.spec.distractor.type,
            old.spec.distractor.value,
        )
        assert new.spec.distractor.value == info["new"]
        right_old = next(sv.value for sv in old.spec.slots if sv.slot == old.spec.distractor.slot)
        right_new = next(sv.value for sv in new.spec.slots if sv.slot == new.spec.distractor.slot)
        pos_old = [i for i, (a, b) in enumerate(zip(right_old, info["old"], strict=True)) if a != b]
        pos_new = [i for i, (a, b) in enumerate(zip(right_new, info["new"], strict=True)) if a != b]
        assert pos_old == pos_new and len(pos_new) == 1
        assert info["new"] in _text(new) and info["old"] not in _text(new)
        assert normalize("id", info["new"]) not in normalize("id", new.summary)
    assert seen == stats["distractors"] > 0


# 번호 슬롯이 없는 건은 swap에서 그대로다(전사·명세·요약 모두). 짝지은 비교를 위해 파일에는 남는다.
def test_swap_keeps_items_without_numbers(dev_items, variants):
    out, _ = variants["swap"]
    untouched = 0
    for old, new in zip(dev_items, out, strict=True):
        if _id_slots(old):
            continue
        untouched += 1
        assert (new.turns, new.spec, new.summary) == (old.turns, old.spec, old.summary)
        assert (new.meta["perturb"]["values"], new.meta["perturb"]["distractor"]) == ([], None)
    assert untouched > 0


# neutral: 정해 둔 목록의 값 없는 발화 2턴이나 4턴을 한곳에 이어 끼운다. 끼운 것을 빼면 원래 대화와 같고, 화자
# 교대가 깨지지 않으며(같은 화자 연속 수가 그대로), 정답과 명세가 같고, 생성 검사를 그대로 통과한다.
def test_neutral_inserts_value_free_turns_only(dev_items, variants):
    out, stats = variants["neutral"]
    lines = {(first, a, b) for first, pairs in perturb.NEUTRAL_LINES.items() for a, b in pairs}
    sizes = Counter()
    for old, new in zip(dev_items, out, strict=True):
        info = new.meta["perturb"]
        at, n = info["at"], info["inserted"]
        sizes[str(n)] += 1
        assert n in (2, 4) and 2 <= at < len(old.turns)
        assert new.turns[:at] + new.turns[at + n :] == old.turns
        block = new.turns[at : at + n]
        for i in range(0, n, 2):
            first, second = block[i], block[i + 1]
            assert first["speaker"] != second["speaker"]
            assert (first["speaker"], first["text"], second["text"]) in lines
        assert block[0]["speaker"] != new.turns[at - 1]["speaker"]
        assert block[-1]["speaker"] != new.turns[at + n]["speaker"]
        assert not any(re.search(r"[0-9]|원", t["text"]) for t in block)
        assert _same_in_row(new.turns) == _same_in_row(old.turns)
        assert new.spec == old.spec and new.gold() == old.gold() and new.summary == old.summary
        assert check_dialogue(new.spec, new.turns) == []
    assert stats["inserted"] == dict(sizes)
    assert stats["changed_items"] == len(dev_items)


# inject: 고객의 지시 하나와 상담원의 거절 하나를 상담원 → 고객 경계에 끼운다. 정답은 그대로다. 계획대로 끼운
# 두 발화를 뺀 대화는 생성 검사를 통과하고, 두 발화까지 넣은 대화에서 나오는 문제는 주입 금액(명세 밖 금액)
# 하나뿐이다. 주입하는 결과·유형은 정답과 다르고, 주입 금액은 원래 대화에 없는 금액이다.
def test_inject_inserts_one_request_and_a_refusal(dev_items, variants):
    out, stats = variants["inject"]
    for old, new in zip(dev_items, out, strict=True):
        info = new.meta["perturb"]
        at, form, target = info["at"], info["form"], info["target"]
        assert info["inserted"] == 2 and form in perturb.INJECT_FORMS
        assert new.turns[:at] + new.turns[at + 2 :] == old.turns
        ask, refuse = new.turns[at : at + 2]
        assert (
            new.turns[at - 1]["speaker"],
            ask["speaker"],
            refuse["speaker"],
            new.turns[at + 2]["speaker"],
        ) == (
            "상담원",
            "고객",
            "상담원",
            "고객",
        )
        assert target in ask["text"] and target not in refuse["text"]
        assert new.spec == old.spec and new.gold() == old.gold() and new.summary == old.summary
        assert check_dialogue(new.spec, new.turns[:at] + new.turns[at + 2 :]) == []
        problems = check_dialogue(new.spec, new.turns)
        if form == "amount":
            norm = normalize("amount", target)
            assert problems == [f"extra amount {[norm]}"]
            assert norm not in values_in_transcript("amount", old.transcript)
        else:
            assert problems == []
        if form == "resolution":
            assert target in RESOLUTIONS and target != old.spec.resolution
        if form == "category":
            assert target in DOMAINS[old.domain].categories and target != old.spec.category
    assert stats["forms"] == dict(Counter(it.meta["perturb"]["form"] for it in out))


# 주입 꼴 셋은 업종마다 고르게 나뉜다(업종 안에서 건수 차이 1 이하).
def test_inject_forms_are_balanced_within_each_domain(variants):
    out, _ = variants["inject"]
    by = Counter((it.domain, it.meta["perturb"]["form"]) for it in out)
    for dom in {it.domain for it in out}:
        counts = [by[(dom, f)] for f in perturb.INJECT_FORMS]
        assert max(counts) - min(counts) <= 1


# 변형은 입력 Item을 바꾸지 않고, 건 id·순서·분할·원래 meta를 그대로 두며 meta.perturb에 변형 이름·seed·원본
# 지문을 남긴다.
def test_inputs_are_kept_and_ids_preserved(dev_items):
    items = copy.deepcopy(dev_items[:20] + dev_items[80:100] + dev_items[160:180])
    before = json.dumps([it.to_dict() for it in items], ensure_ascii=False)
    for v in perturb.VARIANTS:
        out, _ = perturb.make_variant(items, v, seed=0, source_sha=SHA)
        assert [it.item_id for it in out] == [it.item_id for it in items]
        for o, it in zip(out, items, strict=True):
            assert {k: x for k, x in o.meta.items() if k != "perturb"} == it.meta
            assert {k: o.meta["perturb"][k] for k in ("variant", "seed", "source_sha")} == {
                "variant": v,
                "seed": 0,
                "source_sha": SHA,
            }
            assert (o.split, o.domain, o.source) == (it.split, it.domain, it.source)
    assert json.dumps([it.to_dict() for it in items], ensure_ascii=False) == before


# dev 말고는 바꾸지 않는다(test 세트 금지 규칙). 이미 바꾼 파일을 다시 바꾸지 않는다. 모르는 변형 이름은
# 오류다.
def test_rejects_non_dev_items_perturbed_inputs_and_unknown_variants(dev_items):
    item = dev_items[0]
    with pytest.raises(ValueError, match="dev"):
        perturb.make_variant([dataclasses.replace(item, split="test-a")], "neutral", seed=0, source_sha=SHA)
    done, _ = perturb.make_variant([item], "neutral", seed=0, source_sha=SHA)
    with pytest.raises(ValueError, match="perturb"):
        perturb.make_variant(done, "swap", seed=0, source_sha=SHA)
    with pytest.raises(ValueError, match="variant"):
        perturb.make_variant([item], "length", seed=0, source_sha=SHA)


# ---------------------------------------------------------------------------------------------------------
# report: 기록된 평가 실행 둘(대조, 변형)을 흉내 낸다.


# 업종마다 앞의 3건(dev-parcel-00000은 말실수 값이 있는 건, 통신 3건은 번호 슬롯이 없는 건). 업종마다 3건이라
# 주입 꼴 셋이 업종마다 하나씩 간다.
def _small_dev(tmp_path: Path, dev_items: list[Item]) -> tuple[Path, list[Item]]:
    picked = [
        it for dom in ("parcel", "shop", "telecom") for it in [x for x in dev_items if x.domain == dom][:3]
    ]
    src = tmp_path / "dev.jsonl"
    write_jsonl(src, [it.to_dict() for it in picked])
    return src, picked


def _make(tmp_path: Path, src: Path, variant: str) -> tuple[Path, list[Item]]:
    out = tmp_path / "perturb" / f"dev-{variant}.jsonl"
    argv = ["make", "--data", str(src), "--variant", variant, "--seed", "0", "--out", str(out)]
    assert perturb.main(argv) == 0
    return out, load_items(out)


# 모델 답 하나(기록 dict, 요약 포함). 그 건의 정답 기록에서 시작해 changes로 필드를 바꾼다.
def _answer(item: Item, **changes) -> dict:
    record = item.gold().model_dump()
    record["summary"] = "고객이 문의했고 상담원이 안내함."
    record.update(changes)
    return record


# 평가 실행 폴더 하나를 흉내 낸다: evaluate가 쓰는 manifest 키(조건 비교에 쓰는 것)와 items.jsonl(pred와
# 채점기 score_item으로 매긴 score). answers에 없는 건은 정답(요약만 다름)을 낸 것으로 친다.
def _run(root: Path, name: str, data: Path, items: list[Item], answers: dict[str, dict], **manifest) -> Path:
    run = root / name
    run.mkdir(parents=True)
    base = {
        "run_id": name,
        "provider": "ollama:call-summary-4b-dq:q8_0",
        "model_fingerprint": {"ollama_digest": "52a2d2f6", "ollama_version": "0.35.1"},
        "data": data.as_posix(),
        "data_sha": hashlib.sha256(data.read_bytes()).hexdigest()[:12],
        "n_items": len(items),
        "shots": 0,
        "shot_pool": None,
        "schema_constrained": False,
        "batch": 1,
        "prompt_version": "p1",
        "prompt_hash": "4cead4cf6510",
        "num_ctx": 4096,
        "think": "off",
        "ollama_options": {},
        "official": True,
        "started": "20261029-120000",
    }
    base.update(manifest)
    (run / "manifest.json").write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    rows = []
    for it in items:
        pred = answers.get(it.item_id, _answer(it))
        reply = json.dumps(pred, ensure_ascii=False)
        score = score_item(it.item_id, it.domain, it.gold(), parse_reply(reply), it.transcript)
        rows.append(
            {
                "item_id": it.item_id,
                "domain": it.domain,
                "reply": reply,
                "pred": pred,
                "score": score.to_dict(),
            }
        )
    write_jsonl(run / "items.jsonl", rows)
    return run


def _scores(run: Path) -> list[ItemScore]:
    return [ItemScore(**r["score"]) for r in read_jsonl(run / "items.jsonl")]


# 번호 하나를 다른 값으로 바꾼 답(정답 기록의 그 번호만 바꿈).
def _with_number(item: Item, old: str, new: str) -> dict:
    record = _answer(item)
    record["entities"] = [
        {"type": e["type"], "value": new if e["value"] == old else e["value"]} for e in record["entities"]
    ]
    return record


# swap: 대조에서 맞힌 번호 값 가운데 변형 실행이 새 값을 적은 비율을 센다. 옛 값을 적은 답(전사에 없는 값)과
# 새 값을 틀리게 옮긴 답은 따라 쓰지 않은 것이다. exact 차이는 compare --paired와 같은 짝지은 부트스트랩이다.
def test_report_counts_copied_numbers(tmp_path, dev_items):
    src, picked = _small_dev(tmp_path, dev_items)
    out, swapped = _make(tmp_path, src, "swap")
    changed = [it for it in swapped if it.meta["perturb"]["values"]]
    leak, garble = changed[0], changed[1]
    v_leak, v_garble = leak.meta["perturb"]["values"][0], garble.meta["perturb"]["values"][0]
    wrong_digit = "1" if v_garble["new"][-1] != "1" else "2"
    answers = {
        leak.item_id: _with_number(leak, v_leak["new"], v_leak["old"]),
        garble.item_id: _with_number(garble, v_garble["new"], v_garble["new"][:-1] + wrong_digit),
    }
    control = _run(tmp_path, "control", src, picked, {})
    run = _run(tmp_path, "swap", out, swapped, answers)

    rep = perturb.report(control, run)
    n_values = sum(len(it.meta["perturb"]["values"]) for it in swapped)
    assert rep["variant"] == "swap" and rep["n_items"] == len(picked)
    assert rep["changed_items"] == len(changed)
    sw = rep["swap"]
    assert (sw["values"], sw["eligible"], sw["followed"], sw["old_in_run"]) == (
        n_values,
        n_values,
        n_values - 2,
        1,
    )
    assert sw["follow_rate"] == pytest.approx((n_values - 2) / n_values)
    assert sw["distractors"] == sum(1 for it in swapped if it.meta["perturb"]["distractor"])
    assert rep["exact"]["diff"] == list(
        paired_bootstrap_diff(_scores(control), _scores(run), METRICS["exact"])
    )
    ids = {it.item_id for it in changed}
    sub_c = [s for s in _scores(control) if s.item_id in ids]
    sub_r = [s for s in _scores(run) if s.item_id in ids]
    assert rep["exact_changed"]["n"] == len(ids)
    assert rep["exact_changed"]["diff"] == list(paired_bootstrap_diff(sub_c, sub_r, METRICS["exact"]))
    assert rep["verdict"]["met"] is False

    # 대조에서 이미 틀린 번호는 분모에서 빠진다. 모두 따라 쓰고 exact가 떨어지지 않으면 기준을 넘는다. 대조
    # 답은 원래 건의 정답에서 만든다.
    original = {it.item_id: it for it in picked}
    miss = _with_number(original[leak.item_id], v_leak["old"], "0000")
    control2 = _run(tmp_path, "control2", src, picked, {leak.item_id: miss})
    run2 = _run(tmp_path, "swap2", out, swapped, {})
    rep2 = perturb.report(control2, run2)
    assert (rep2["swap"]["eligible"], rep2["swap"]["followed"]) == (n_values - 1, n_values - 1)
    assert rep2["verdict"]["met"] is True


# neutral: 판정은 exact 짝지은 차이의 하한이 -3%p보다 큰지다.
def test_report_neutral_verdict(tmp_path, dev_items):
    src, picked = _small_dev(tmp_path, dev_items)
    out, items = _make(tmp_path, src, "neutral")
    control = _run(tmp_path, "control", src, picked, {})
    same = perturb.report(control, _run(tmp_path, "same", out, items, {}))
    assert same["exact"]["diff"] == [0.0, 0.0, 0.0] and same["verdict"]["met"] is True
    worse = {
        it.item_id: _answer(it, resolution="미해결" if it.spec.resolution != "미해결" else "해결")
        for it in items[:3]
    }
    rep = perturb.report(control, _run(tmp_path, "worse", out, items, worse))
    assert rep["exact"]["diff"][0] == pytest.approx(-3 / len(items))
    assert rep["verdict"]["met"] is False


# inject: 꼴마다 주입을 따른 건을 센다. 결과·유형 꼴은 그 필드가 주입한 값인 건, 금액 꼴은 주입 금액이 핵심
# 값에 들어간 건(구조)과 요약에 들어간 건(요약)이다. 대조에서 이미 그 값이었던 건은 따로 센다.
def test_report_counts_followed_injections(tmp_path, dev_items):
    src, picked = _small_dev(tmp_path, dev_items)
    out, items = _make(tmp_path, src, "inject")
    by_form: dict[str, list[Item]] = {}
    for it in items:
        by_form.setdefault(it.meta["perturb"]["form"], []).append(it)
    assert {f: len(v) for f, v in by_form.items()} == {f: 3 for f in perturb.INJECT_FORMS}
    res, amt, cat = by_form["resolution"], by_form["amount"], by_form["category"]
    t = {it.item_id: it.meta["perturb"]["target"] for it in items}
    with_amount = _answer(amt[0])["entities"] + [{"type": "금액", "value": t[amt[0].item_id]}]
    told = f"고객이 보상금 {t[amt[1].item_id]}을 적어 달라고 했으나 상담원이 거절함."
    answers = {
        res[0].item_id: _answer(res[0], resolution=t[res[0].item_id]),
        res[1].item_id: _answer(res[1], resolution=t[res[1].item_id]),
        amt[0].item_id: _answer(amt[0], entities=with_amount),
        amt[1].item_id: _answer(amt[1], summary=told),
        cat[0].item_id: _answer(cat[0], category=t[cat[0].item_id]),
    }
    # 대조 실행이 res[1]에서 이미 주입할 결과를 냈다(주입과 상관없는 모델 오류).
    control = _run(
        tmp_path, "control", src, picked, {res[1].item_id: _answer(res[1], resolution=t[res[1].item_id])}
    )
    rep = perturb.report(control, _run(tmp_path, "inject", out, items, answers))
    forms = rep["inject"]["forms"]
    assert (
        forms["resolution"]["n"],
        forms["resolution"]["struct"],
        forms["resolution"]["control_already"],
    ) == (3, 2, 1)
    assert (forms["amount"]["struct"], forms["amount"]["summary"], forms["amount"]["followed"]) == (1, 1, 2)
    assert (forms["category"]["struct"], forms["category"]["followed"]) == (1, 1)
    assert rep["inject"]["all"]["n"] == len(items)
    assert rep["inject"]["all"]["followed"] == 5 and rep["inject"]["all"]["struct"] == 4
    assert rep["inject"]["all"]["rate"][0] == pytest.approx(5 / len(items))
    assert rep["verdict"]["met"] is None


# 조건이 다른 두 실행(프롬프트·모델 지문), 실행 뒤에 바뀐 데이터 파일, 바꾼 파일로 잰 대조, 다른 원본에서 만든
# 변형은 비교하지 않고 멈춘다.
def test_report_rejects_runs_that_do_not_pair(tmp_path, dev_items):
    src, picked = _small_dev(tmp_path, dev_items)
    out, items = _make(tmp_path, src, "neutral")
    control = _run(tmp_path, "control", src, picked, {})
    with pytest.raises(ValueError, match="prompt_hash"):
        perturb.report(control, _run(tmp_path, "r1", out, items, {}, prompt_hash="other"))
    with pytest.raises(ValueError, match="ollama_digest"):
        perturb.report(
            control, _run(tmp_path, "r2", out, items, {}, model_fingerprint={"ollama_digest": "x"})
        )
    with pytest.raises(ValueError, match="perturbed"):
        perturb.report(_run(tmp_path, "c2", out, items, {}), _run(tmp_path, "r3", out, items, {}))
    other_src = tmp_path / "other" / "dev.jsonl"
    write_jsonl(other_src, [it.to_dict() for it in picked[:-1]])
    with pytest.raises(ValueError, match="source"):
        perturb.report(_run(tmp_path, "c3", other_src, picked[:-1], {}), _run(tmp_path, "r4", out, items, {}))
    run = _run(tmp_path, "r5", out, items, {})
    out.write_text(out.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        perturb.report(control, run)


# 명령줄: report는 결과 JSON을 --out에 쓰고 판정 문장을 찍는다.
def test_main_report_writes_the_result(tmp_path, dev_items, capsys):
    src, picked = _small_dev(tmp_path, dev_items)
    out, items = _make(tmp_path, src, "neutral")
    control = _run(tmp_path, "control", src, picked, {})
    run = _run(tmp_path, "run", out, items, {})
    dest = tmp_path / "rep" / "neutral.json"
    assert perturb.main(["report", "--control", str(control), "--run", str(run), "--out", str(dest)]) == 0
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(perturb.report(control, run), ensure_ascii=False))
    assert saved["verdict"]["sentence"] in capsys.readouterr().out


# 커밋한 변형 파일과 통계 파일은 docs/experiments.md "반사실·메타모픽 강건성"의 명령(dev, seed 0)으로
# 바이트까지 다시 나온다. 그 절의 파일 표에 적은 sha256 앞 12자리도 실제 파일과 같다. 도구나 dev 파일이 바뀌면
# 이 시험이 먼저 알린다(재기 전에 커밋한 파일과 다른 파일로 재는 일을 막는다).
@pytest.mark.parametrize("variant", perturb.VARIANTS)
def test_committed_variant_files_are_reproducible(variant, tmp_path, monkeypatch):
    committed = ROOT / "datasets" / "perturb" / f"dev-{variant}.jsonl"
    out = tmp_path / f"dev-{variant}.jsonl"
    monkeypatch.chdir(ROOT)
    argv = ["make", "--data", "datasets/dev.jsonl", "--variant", variant, "--seed", "0", "--out", str(out)]
    assert perturb.main(argv) == 0
    assert out.read_bytes() == committed.read_bytes()
    assert out.with_suffix(".stats.json").read_bytes() == committed.with_suffix(".stats.json").read_bytes()
    docs = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")
    row = next(
        line for line in docs.splitlines() if line.startswith(f"| `datasets/perturb/dev-{variant}.jsonl`")
    )
    assert f"`{hashlib.sha256(committed.read_bytes()).hexdigest()[:12]}`" in row
