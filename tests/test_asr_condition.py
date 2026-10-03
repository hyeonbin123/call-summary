"""scripts/asr_condition.py: the verbalizer v2 rule and the cache plan. No speech models are loaded here.

The script runs in the sibling project's speech environment, but its text rules and the plan must be
importable in this one (no numpy, scipy or support_agent at import time).
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERBALIZE_PY = ROOT.parent / "support-agent" / "src" / "support_agent" / "voice" / "verbalize.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


asr = _load("asr_condition", ROOT / "scripts" / "asr_condition.py")


def squash(text: str) -> str:  # stands in for verbalize: only the whitespace clean-up
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize(
    "text,spoken",
    [
        ("운송장 번호 6291-7877-5168로 조회할게요", "운송장 번호 육이구일, 칠팔칠칠, 오일육팔 로 조회할게요"),
        ("1000-0203-0040입니다.", "일공공공, 공이공삼, 공공사공 입니다."),
        ("운송장 6291-7877-5168", "운송장 육이구일, 칠팔칠칠, 오일육팔"),
        ("６２９１-７８７７-５１６８요", "육이구일, 칠팔칠칠, 오일육팔 요"),  # full-width digits (NFKC)
    ],
)
def test_v2_reads_tracking_numbers_digit_by_digit(text, spoken):
    assert asr.spoken_form(text, squash, "v2") == spoken


@pytest.mark.parametrize(
    "text",
    [
        "환불은 3-5일 이내에 됩니다",  # a real range keeps the verbalizer's own reading
        "보통 2~3일 걸립니다",
        "010-1234-5678로 연락 주세요",  # phone numbers are the verbalizer's job
        "1234-5678로",  # two groups
        "1234-5678-9012-3456요",  # four groups: not the tracking-number shape, left whole
        "12345-6789-0123요",
        "주문번호 D3928174, 접수번호 R123456",
        "서울 강남구 역삼동 761-4 입니다",
    ],
)
def test_v2_leaves_everything_else_to_the_verbalizer(text):
    assert asr.spoken_form(text, squash, "v2") == asr.spoken_form(text, squash, "v1") == squash(text)


def test_v1_is_the_verbalizer_alone():
    assert asr.spoken_form("6291-7877-5168로", lambda s: f"<{s}>", "v1") == "<6291-7877-5168로>"
    with pytest.raises(ValueError):
        asr.spoken_form("x", squash, "v3")


@pytest.mark.skipif(not VERBALIZE_PY.exists(), reason="support-agent is not checked out next to this repo")
def test_v2_against_the_real_verbalizer():
    verbalize = _load("sa_verbalize", VERBALIZE_PY).verbalize
    v1 = asr.spoken_form("운송장 번호 6291-7877-5168로 조회", verbalize, "v1")
    assert "에서" in v1 and "육천이백구십일" in v1  # the stage-4 artifact: cardinals and a range "에서"
    v2 = asr.spoken_form("운송장 번호 6291-7877-5168로 조회", verbalize, "v2")
    assert v2 == "운송장 번호 육이구일, 칠팔칠칠, 오일육팔 로 조회"
    assert verbalize("010-1234-5678로") == "공일공, 일이삼사, 오육칠팔 로"  # the same comma pattern
    for same in ("환불은 3-5일 이내", "2~3일 내로", "주문번호 D3928174요", "38,900원입니다"):
        assert asr.spoken_form(same, verbalize, "v2") == verbalize(same)


def _items():
    return [
        {
            "item_id": "dev-parcel-00001",
            "turns": [
                {"speaker": "상담원", "text": "새싹택배입니다."},
                {"speaker": "고객", "text": "운송장 6291-7877-5168로 조회해 주세요."},
                {"speaker": "고객", "text": "..."},  # nothing to say: never synthesised
            ],
        }
    ]


def _cache_for(items, identity_key, verbalizer):
    keys = set()
    for item_id, i, speaker, text in asr.utterances(items):
        spoken = asr.spoken_form(text, squash, verbalizer)
        if asr.has_hangul(spoken):
            keys.add(asr.cache_key(identity_key, speaker, asr.seed_of(item_id, i), spoken))
    return keys


def test_plan_after_v1_synthesises_only_the_changed_utterances():
    ident = asr.identity_key({"tts": "melotts/KR", "versions": {"torch": "2.11.0"}})
    v1_cache = _cache_for(_items(), ident, "v1")
    p = asr.plan(_items(), squash, ident, v1_cache, "v2")
    assert p == {
        "utterances": 3,
        "silent": 1,
        "changed": 1,
        "cached": 1,
        "to_synthesise": 1,
        "unchanged_not_cached": 0,
    }
    assert asr.plan(_items(), squash, ident, v1_cache, "v1")["to_synthesise"] == 0


def test_plan_flags_a_changed_speech_environment():
    old = asr.identity_key({"tts": "melotts/KR", "versions": {"torch": "2.11.0"}})
    new = asr.identity_key({"tts": "melotts/KR", "versions": {"torch": "2.12.0"}})
    p = asr.plan(_items(), squash, new, _cache_for(_items(), old, "v1"), "v2")
    assert p["unchanged_not_cached"] == 1 and p["to_synthesise"] == 2


def test_the_cache_key_does_not_carry_the_verbalizer(tmp_path):
    ident = {"tts": "melotts/KR", "versions": {"torch": "2.11.0"}}
    ch1 = asr.Channel(tmp_path / "c.jsonl", verbalizer="v1", verbalize=squash, identity=ident)
    ch2 = asr.Channel(tmp_path / "c.jsonl", verbalizer="v2", verbalize=squash, identity=ident)
    assert ch1._id == ch2._id == asr.identity_key(ident)
    assert ch2.meta(["원문"]) == {**ident, "verbalizer": "v2", "written": ["원문"]}


def test_a_run_refuses_to_resynthesise_unchanged_utterances(tmp_path, monkeypatch, capsys):
    data = tmp_path / "datasets"
    data.mkdir()
    (data / "dev.jsonl").write_text(
        "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in _items()), encoding="utf-8"
    )
    ident = {"tts": "melotts/KR", "versions": {"torch": "2.12.0"}}  # not what the (empty) cache was made with
    monkeypatch.setattr(asr, "speech_identity", lambda whisper_model: ident)
    monkeypatch.setattr(asr, "load_verbalize", lambda: squash)

    def no_models(self):
        raise AssertionError("models loaded")

    monkeypatch.setattr(asr.Channel, "_load_models", no_models)
    out, cache = str(tmp_path / "out"), str(tmp_path / "c")
    args = ["--split", "dev", "--data", str(data), "--out", out, "--cache", cache]
    assert asr.main(args + ["--dry-run"]) == 0
    assert asr.main(args) == 2
    assert "unchanged" in capsys.readouterr().err
    assert not (tmp_path / "out" / "dev.jsonl").exists()
