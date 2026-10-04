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


# --- speech condition v3: pass 1 keeps the audio, pass 2 recognises it again ---------------------------


def test_identifier_turns_are_the_written_order_receipt_and_tracking_numbers():
    assert asr.says_identifier("주문번호는 D4477838입니다.")
    assert asr.says_identifier("접수번호 R123456요")
    assert asr.says_identifier("운송장 5891-6890-2287의 위치")
    assert asr.says_identifier("６２９１-７８７７-５１６８요")  # NFKC
    assert not asr.says_identifier("010-1234-5678로 연락 주세요")
    assert not asr.says_identifier("11월 29일에 도착합니다")


class _FakeAudio(list):
    pass


class _FakeListener:
    def __init__(self):
        self.heard: list[bytes] = []

    def transcribe(self, wav: bytes) -> str:
        self.heard.append(wav)
        return "들림:" + wav.decode()


def _fake_channel(tmp_path, monkeypatch, cache_rows=()):
    cache = tmp_path / "c.jsonl"
    cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cache_rows), encoding="utf-8")
    ident = {"tts": "melotts/KR", "versions": {"torch": "2.11.0"}}
    ch = asr.Channel(cache, verbalizer="v2", verbalize=squash, identity=ident)

    def load(self):
        self.speaker, self.listener = object(), _FakeListener()
        self.wav_bytes = lambda audio, rate: type("A", (), {"wav": "".join(audio).encode()})()

    monkeypatch.setattr(asr.Channel, "_load_models", load)
    monkeypatch.setattr(asr.Channel, "_speak", lambda self, spoken, speaker, seed: _FakeAudio(spoken))
    return ch


def test_pass_one_keeps_the_heard_audio_and_indexes_every_turn(tmp_path, monkeypatch):
    items = [{**it, "domain": "parcel"} for it in _items()]
    first = asr.spoken_form(items[0]["turns"][0]["text"], squash, "v2")
    key0 = asr.cache_key(
        asr.identity_key({"tts": "melotts/KR", "versions": {"torch": "2.11.0"}}),
        "상담원",
        asr.seed_of("dev-parcel-00001", 0),
        first,
    )
    ch = _fake_channel(tmp_path, monkeypatch, [{"key": key0, "spoken": first, "heard": "예전에 들림"}])
    store = asr.WavStore(tmp_path / "wav")
    n = asr.save_wavs(ch, items, store, "dev")
    assert n == {"synthesised": 2, "silent": 1}
    rows = store.split_rows("dev")
    assert [(r["item_id"], r["turn"]) for r in rows] == [("dev-parcel-00001", i) for i in range(3)]
    assert rows[0]["key"] == key0 and rows[0]["cached_heard"] == "예전에 들림"
    assert rows[0]["heard"] == "들림:" + first and rows[0]["domain"] == "parcel"
    assert (store.root / rows[0]["wav"]).read_bytes() == first.encode()  # the bytes the recogniser was given
    assert rows[1]["spoken"] == "운송장 육이구일, 칠팔칠칠, 오일육팔 로 조회해 주세요."
    assert rows[2]["wav"] is None and rows[2]["heard"] == "" and rows[2]["audio_seconds"] == 0.0
    # resumable: a second run synthesises nothing, and a reloaded store has the same rows
    assert asr.save_wavs(ch, items, store, "dev") == {"done": 3}
    assert asr.WavStore(tmp_path / "wav").split_rows("dev") == rows


def test_pass_one_can_keep_only_identifier_turns(tmp_path, monkeypatch):
    ch = _fake_channel(tmp_path, monkeypatch)
    store = asr.WavStore(tmp_path / "wav")
    items = [{**it, "domain": "parcel"} for it in _items()]
    assert asr.save_wavs(ch, items, store, "dev", only_ids=True) == {"synthesised": 1}
    assert [r["turn"] for r in store.split_rows("dev")] == [1]
    # the full run later adds the other turns and keeps the first one
    assert asr.save_wavs(ch, items, store, "dev") == {"done": 1, "synthesised": 1, "silent": 1}


def test_pass_two_recognises_each_kept_wav_once_with_the_domain_context(tmp_path, monkeypatch):
    ch = _fake_channel(tmp_path, monkeypatch)
    store = asr.WavStore(tmp_path / "wav")
    asr.save_wavs(ch, [{**it, "domain": "parcel"} for it in _items()], store, "dev")
    calls = []

    def transcribe(wav: bytes, hotwords: str | None) -> str:
        calls.append(hotwords)
        return f"{wav.decode()}|{hotwords}"

    hyp = tmp_path / "hyp" / "Th.jsonl"
    context = {"parcel": "새싹택배, 전자레인지"}
    assert asr.recognise(store, "dev", "Th", hyp, transcribe, context) == {"recognised": 2}
    assert calls == ["새싹택배, 전자레인지"] * 2
    rows = [json.loads(line) for line in hyp.read_text(encoding="utf-8").splitlines()]
    assert [(r["item_id"], r["turn"], r["arm"]) for r in rows] == [
        ("dev-parcel-00001", 0, "Th"),
        ("dev-parcel-00001", 1, "Th"),
    ]
    assert rows[0]["heard"].endswith("|새싹택배, 전자레인지")
    assert asr.recognise(store, "dev", "Th", hyp, transcribe, context) == {"done": 2}
    assert asr.recognise(store, "dev", "L", tmp_path / "L.jsonl", transcribe, None) == {"recognised": 2}
    assert calls[-1] is None  # only Th and Qc carry the context


def test_the_recogniser_arms_and_their_decoding():
    assert set(asr.RECOGNISERS) == {"T", "Th", "L", "N"}
    assert asr.RECOGNISERS["T"] == ("large-v3-turbo", False)
    assert asr.RECOGNISERS["Th"] == ("large-v3-turbo", True)
    assert asr.RECOGNISERS["L"] == ("large-v3", False)
    assert asr.RECOGNISERS["N"][0].replace("\\", "/").endswith("whisper-ko-ft/outputs/turbo-n/ct2")


def test_the_pass_one_plan_counts_what_it_would_synthesise(tmp_path, monkeypatch):
    ch = _fake_channel(tmp_path, monkeypatch)
    store = asr.WavStore(tmp_path / "wav")
    items = [{**it, "domain": "parcel"} for it in _items()]
    plan = asr.keep_plan(ch, items, store, only_ids=False)
    assert plan == {"turns": 3, "done": 0, "silent": 1, "to_synthesise": 2, "in_text_cache": 0}
    store.check_identity({"a": 1}, write=False)
    assert not (store.root / "identity.json").exists()
    store.check_identity({"a": 1})
    with pytest.raises(SystemExit):
        store.check_identity({"a": 2}, write=False)
    asr.save_wavs(ch, items, store, "dev", only_ids=True)
    assert asr.keep_plan(ch, items, store, only_ids=True)["done"] == 1
