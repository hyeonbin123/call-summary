"""scripts/asr_condition.py: the verbalizer v2 rule and the cache plan. No speech models are loaded here.

The script runs in the sibling project's speech environment, but its text rules and the plan must be
importable in this one (no numpy, scipy or support_agent at import time).
"""

# [테스트 묶음] 음성 조건 데이터를 만드는 scripts/asr_condition.py를 음성 모델 없이 검증한다. (1) 합성에 넘길
# 글을 만드는 규칙(verbalizer v1/v2), (2) 캐시를 보고 새로 합성할 발화를 세는 계획(plan, --dry-run), (3) 음성
# 환경이 캐시와 다르면 전체 재합성을 거절하는 보호, (4) 음성 조건 v3의 1패스(소리 남기기)와 2패스(남긴 소리를
# 다른 인식기로 다시 듣기).
# [왜 중요한가] v1 조건은 운송장번호를 기수와 범위 "에서"로 읽게 한 측정 인공물이 있었다(docs/experiments.md
# "음성 조건 v2"). v2가 운송장번호만 숫자 하나씩 읽고 나머지 발화는 v1과 글자까지 같아야 바뀐 발화만 다시
# 합성하고 v1 결과와 비교할 수 있다. 합성·인식은 GPU 몇 시간짜리라, 다시 돌리기 전에 무엇을 할지 정확히 세는
# 것이 중요하다.

import importlib.util
import json
import re
from pathlib import Path

import pytest

# 스크립트는 support-agent의 음성 환경에서 도는 독립 파일이라 패키지로 import하지 않고 경로로 읽는다(_load).
# VERBALIZE_PY는 옆 저장소 support-agent의 실제 verbalize 파일 경로로, 있을 때만 실제 규칙과 맞춰 본다.
ROOT = Path(__file__).resolve().parents[1]
VERBALIZE_PY = ROOT.parent / "support-agent" / "src" / "support_agent" / "voice" / "verbalize.py"


# 파일 경로로 모듈을 불러오는 도우미. 스크립트를 불러오는 순간 numpy·scipy·support_agent를 import하면 이
# 환경에서 실패하므로, 불러오기 자체가 그런 의존성 없이 되는지도 함께 확인된다(모듈 docstring).
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


asr = _load("asr_condition", ROOT / "scripts" / "asr_condition.py")


# 실제 verbalize 대신 공백만 정리하는 가짜. 그래서 아래 기대값은 이 스크립트가 더하는 규칙(운송장번호 읽기)만
# 보여 준다.
def squash(text: str) -> str:  # stands in for verbalize: only the whitespace clean-up
    return re.sub(r"\s+", " ", text).strip()


# v2는 운송장번호(4-4-4자리)를 숫자 하나씩 한글 이름으로, 묶음 사이에 쉼표를 두고 읽는다. 0은 "공"으로
# 읽고(전화번호를 읽는 verbalize와 같은 방식), 전각 숫자도 NFKC로 바꿔 같은 규칙을 적용한다.
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


# 운송장번호 꼴이 아닌 것은 v2도 손대지 않고 verbalize에 맡겨 v1과 글자까지 같아야 한다: 진짜 범위(3-5일,
# 2~3일), 전화번호, 묶음이 둘·넷인 숫자, 첫 묶음이 다섯 자리인 숫자, 주문·접수번호, 주소 번지. 이 조건이
# 지켜져야 v2에서 다시 합성할 발화가 운송장번호 발화로만 한정된다(규칙: 조건 v2는 이것만 바꾼다).
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


# v1은 verbalize를 그대로 부른 결과이고, 모르는 verbalizer 이름은 조용히 넘기지 않고 ValueError로 거절한다.
def test_v1_is_the_verbalizer_alone():
    assert asr.spoken_form("6291-7877-5168로", lambda s: f"<{s}>", "v1") == "<6291-7877-5168로>"
    with pytest.raises(ValueError):
        asr.spoken_form("x", squash, "v3")


# 옆 저장소의 실제 verbalize로 확인한다(없으면 건너뜀). v1이 4단계의 인공물("육천이백구십일 … 에서")을 그대로
# 재현하고, v2는 숫자 하나씩 읽으며, 이 꼴이 verbalize가 전화번호를 읽는 쉼표 꼴과 같은지, 그 밖의 발화는
# v2에서도 verbalize 결과 그대로인지 본다.
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


# 택배 건 하나: 운송장번호가 없는 발화, 있는 발화, "..."(한글이 없어 합성하지 않는 발화).
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


# 주어진 음성 환경·verbalizer로 이미 합성했다고 치는 캐시 키 집합을 만든다. 키는 (음성 환경, 화자, 발화 seed,
# 읽는 글)의 해시라, 읽는 글이 같으면 v1과 v2가 같은 키를 쓴다.
def _cache_for(items, identity_key, verbalizer):
    keys = set()
    for item_id, i, speaker, text in asr.utterances(items):
        spoken = asr.spoken_form(text, squash, verbalizer)
        if asr.has_hangul(spoken):
            keys.add(asr.cache_key(identity_key, speaker, asr.seed_of(item_id, i), spoken))
    return keys


# v1 캐시가 있을 때 v2 계획은 읽는 글이 바뀐 운송장번호 발화 하나만 새로 합성하고, 그대로인 발화는 캐시에서
# 가져오며, 소리 없는 발화는 따로 센다. v1로 다시 계획하면 합성할 것이 없다(--dry-run이 보여 주는 수).
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


# 음성 환경(여기서는 torch 버전)이 캐시와 다르면 v1과 같게 읽는 발화도 캐시에 없으므로 unchanged_not_cached가
# 0보다 커진다. 이 수가 "환경이 바뀌었다"는 신호다.
def test_plan_flags_a_changed_speech_environment():
    old = asr.identity_key({"tts": "melotts/KR", "versions": {"torch": "2.11.0"}})
    new = asr.identity_key({"tts": "melotts/KR", "versions": {"torch": "2.12.0"}})
    p = asr.plan(_items(), squash, new, _cache_for(_items(), old, "v1"), "v2")
    assert p["unchanged_not_cached"] == 1 and p["to_synthesise"] == 2


# 캐시 키에 verbalizer 이름을 넣지 않는다(키는 음성 환경 + 화자 + seed + 읽는 글, 커밋 82638b5). 그래야 v1과
# 같게 읽는 발화를 v2에서 다시 합성하지 않는다. verbalizer는 각 건의 meta에 남는다.
def test_the_cache_key_does_not_carry_the_verbalizer(tmp_path):
    ident = {"tts": "melotts/KR", "versions": {"torch": "2.11.0"}}
    ch1 = asr.Channel(tmp_path / "c.jsonl", verbalizer="v1", verbalize=squash, identity=ident)
    ch2 = asr.Channel(tmp_path / "c.jsonl", verbalizer="v2", verbalize=squash, identity=ident)
    assert ch1._id == ch2._id == asr.identity_key(ident)
    assert ch2.meta(["원문"]) == {**ident, "verbalizer": "v2", "written": ["원문"]}


# 캐시와 다른 음성 환경으로 실제 실행을 하면 v1과 같게 읽는 발화까지 모두 다시 합성하게 되므로, 스크립트는
# 모델을 올리기 전에 거절하고 종료 코드 2를 돌려야 한다(--allow-full-run 없이는). --dry-run은 계획만 보여 주고
# 0을 돌린다.
# monkeypatch로 음성 환경 정보와 verbalize를 가짜로 바꾸고, 모델 적재 함수가 불리면 실패하게 해 "모델을 올리지
# 않았음"까지 확인한다. 출력 파일도 생기지 않아야 한다.
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


# v3 싼 관문이 고르는 "번호 발화": 쓴 글에 주문번호(D+7자리)·접수번호(R+6자리)·운송장번호(4-4-4) 꼴이 있는
# 발화만이다(전각 숫자 포함). 전화번호와 날짜는 넣지 않는다(docs/experiments.md 음성 조건 v3 "싼 첫 관문").
def test_identifier_turns_are_the_written_order_receipt_and_tracking_numbers():
    assert asr.says_identifier("주문번호는 D4477838입니다.")
    assert asr.says_identifier("접수번호 R123456요")
    assert asr.says_identifier("운송장 5891-6890-2287의 위치")
    assert asr.says_identifier("６２９１-７８７７-５１６８요")  # NFKC
    assert not asr.says_identifier("010-1234-5678로 연락 주세요")
    assert not asr.says_identifier("11월 29일에 도착합니다")


# 가짜 합성 결과: 읽는 글을 그대로 담은 목록. 아래 가짜 wav_bytes가 이것을 바이트로 바꾸므로 "인식기가 받은
# 바이트"를 글로 확인할 수 있다.
class _FakeAudio(list):
    pass


# 가짜 인식기: 받은 바이트를 기록하고 "들림:<글>"을 돌려준다.
class _FakeListener:
    def __init__(self):
        self.heard: list[bytes] = []

    def transcribe(self, wav: bytes) -> str:
        self.heard.append(wav)
        return "들림:" + wav.decode()


# 모델 없이 도는 Channel을 만든다. _load_models와 _speak를 monkeypatch로 가짜로 바꿔 MeloTTS·faster-whisper를
# 올리지 않는다. cache_rows로 글 캐시(v2 캐시)를 미리 채울 수 있다.
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


# 1패스(--keep-audio): 발화마다 다시 합성해 인식기가 받은 WAV 바이트를 그대로 저장하고, index에 키·T의 글·같은
# 키의 글 캐시 값·업종을 한 줄씩 적는지 확인한다. 다른 팔이 모두 이 같은 바이트를 들어야 비교가 공정하다(문서
# "소리" 절).
# 소리 없는 발화도 행을 남기고(wav None), 두 번째 실행은 아무것도 다시 합성하지 않으며(이어 하기), 다시 읽은
# 저장소의 행이 같은지도 본다.
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


# --only-ids는 번호 발화만 합성해 남기고(싼 관문용), 나중에 전체를 돌리면 이미 남긴 발화는 건너뛰고 나머지만
# 더한다.
def test_pass_one_can_keep_only_identifier_turns(tmp_path, monkeypatch):
    ch = _fake_channel(tmp_path, monkeypatch)
    store = asr.WavStore(tmp_path / "wav")
    items = [{**it, "domain": "parcel"} for it in _items()]
    assert asr.save_wavs(ch, items, store, "dev", only_ids=True) == {"synthesised": 1}
    assert [r["turn"] for r in store.split_rows("dev")] == [1]
    # the full run later adds the other turns and keeps the first one
    assert asr.save_wavs(ch, items, store, "dev") == {"done": 1, "synthesised": 1, "silent": 1}


# 2패스(--recognise): 남긴 WAV마다 한 번씩 인식하고 팔 이름과 함께 가설 파일에 적는다. 맥락을 주면(Th 팔) 그
# 건 업종의 맥락 글이 hotwords로 넘어가고, 맥락이 없는 팔(L)은 None이다. 다시 돌리면 이미 적은 발화는
# 건너뛴다(이어 하기). 소리 없는 발화는 인식하지 않는다.
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


# faster-whisper 팔의 모델과 hotwords 여부가 규칙 표(T: turbo, Th: turbo + 맥락, L: large-v3, N: whisper-ko-ft
# turbo-n 변환본)와 같은지 확인한다.
def test_the_recogniser_arms_and_their_decoding():
    assert set(asr.RECOGNISERS) == {"T", "Th", "L", "N"}
    assert asr.RECOGNISERS["T"] == ("large-v3-turbo", False)
    assert asr.RECOGNISERS["Th"] == ("large-v3-turbo", True)
    assert asr.RECOGNISERS["L"] == ("large-v3", False)
    assert asr.RECOGNISERS["N"][0].replace("\\", "/").endswith("whisper-ko-ft/outputs/turbo-n/ct2")


# 1패스 계획(keep_plan)이 발화 수, 이미 한 것, 소리 없음, 합성할 것, 글 캐시에 키가 있는 것을 세는지 본다.
# 저장소의 음성 환경 기록(identity.json)은 write=False면 쓰지 않고, 다른 환경으로 같은 저장소에 더하려 하면
# SystemExit로 거절한다(서로 다른 음성 환경의 소리를 섞지 않게).
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
