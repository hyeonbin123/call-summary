"""scripts/asr_qwen.py: the parts that need no model (and no numpy or torch, except where skipped)."""

# [테스트 묶음] 음성 조건 v3의 Qwen3-ASR 팔(Q, Qc) 2패스 스크립트 scripts/asr_qwen.py 가운데 모델 없이 도는
# 부분을 검증한다: 디코딩된 글에서 받아 적은 글만 꺼내기, 고정한 모델 판·디코딩 설정, 이어 하기를 위한 할 일
# 고르기·배치 나누기, 맥락 없는 Qc 거절, 할 일이 없으면 모델을 올리지 않기, WAV 읽기.
# [왜 중요한가] 규칙은 모델 판(revision)과 탐욕 디코딩·토큰 상한을 측정 전에 고정했다(docs/experiments.md 음성
# 조건 v3 "팔" 표). 설정이 바뀌거나 이어 하기가 이미 한 발화를 다시 인식하면 같은 소리를 다른 조건으로 재게
# 된다.

import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import pytest

# scripts/ 아래 스크립트는 패키지가 아니라서 파일 경로로 불러온다. 스크립트가 옆의 asr_condition.py를
# import하므로 그 경로도 스크립트가 스스로 sys.path에 넣는다.
ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qwen = _load("asr_qwen", ROOT / "scripts" / "asr_qwen.py")


# raw_transcription: 공식 파서처럼 "assistant" 뒤, "<asr_text>" 표시 뒤의 글만 꺼내 앞뒤 공백을 떼되 반복
# 제거는 하지 않는다(whisper-ko-ft 9단계와 같음). 언어를 강제하면 표시 없이 글만 나오는 경우도 그대로 둔다.
# 학생에게 가는 것은 이 반복 제거 전 글(raw)이다(규칙: raw를 쓰고 fixed는 기록만).
@pytest.mark.parametrize(
    "decoded,want",
    [
        ("주문번호는 디 사사칠칠팔삼팔 입니다.", "주문번호는 디 사사칠칠팔삼팔 입니다."),  # language forced
        ("language Korean<asr_text>네, 맞습니다.", "네, 맞습니다."),
        ("system\nassistant\nlanguage Korean<asr_text> 안녕하세요 ", "안녕하세요"),
        ("", ""),
    ],
)
def test_raw_transcription_is_the_text_after_the_marker(decoded, want):
    assert qwen.raw_transcription(decoded) == want


# 모델 이름과 revision, 탐욕 디코딩(샘플링 없음, 빔 1, 새 토큰 최대 256), 팔 표(Qc만 맥락을 쓴다)가 규칙에
# 적은 값 그대로인지 고정한다.
def test_pinned_model_and_greedy_decoding():
    assert qwen.MODEL == "Qwen/Qwen3-ASR-1.7B-hf"
    assert qwen.REVISION == "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    assert qwen.GENERATE == {"do_sample": False, "num_beams": 1, "max_new_tokens": 256}
    assert qwen.ARMS == {"Q": False, "Qc": True}


# 할 일 고르기: 1패스 저장소에서 그 분할의, 소리가 있고(wav가 None이 아님) 가설 파일에 아직 없는 발화만 저장
# 순서대로 고른다. 다른 분할 발화는 섞이지 않고, limit은 앞에서부터 자른다(스모크용). batched는 순서를 지키며
# 크기대로 나눈다(마지막 배치는 작을 수 있음).
def test_rows_to_do_skip_silent_turns_and_finished_keys(tmp_path):
    store = qwen.WavStore(tmp_path)
    for i, wav in enumerate([b"a", None, b"c", b"d"]):
        row = {
            "key": f"k{i}",
            "split": "dev",
            "item_id": "x",
            "turn": i,
            "domain": "shop",
            "audio_seconds": 1.0,
        }
        store.add(row, wav)
    store.add({"key": "t0", "split": "test-a", "item_id": "y", "turn": 0, "domain": "shop"}, b"t")
    rows = qwen.todo_rows(store, "dev", {"k2"})
    assert [r["key"] for r in rows] == ["k0", "k3"]
    assert [r["key"] for r in qwen.todo_rows(store, "dev", set(), limit=1)] == ["k0"]
    assert [[r["key"] for r in b] for b in qwen.batched(qwen.todo_rows(store, "dev", set()), 2)] == [
        ["k0", "k2"],
        ["k3"],
    ]


# Qc는 업종 맥락이 있어야 하는 팔이라 --context 없이 부르면 argparse 오류로 멈춘다. 맥락 없이 돌면 Q와 같은
# 조건이 되어 팔이 뜻을 잃는다.
def test_the_context_arm_refuses_to_run_without_context(tmp_path, capsys):
    with pytest.raises(SystemExit):
        qwen.main(["--arm", "Qc", "--split", "dev", "--audio", str(tmp_path)])
    assert "needs --context" in capsys.readouterr().err


# 할 발화가 0이면 torch·transformers를 불러오거나 모델을 올리지 않고 0으로 끝난다. 표준 오류의 첫 JSON 줄에
# to_do 0이 찍히는지 본다(이어 하기에서 다 끝난 팔을 다시 불러도 GPU를 쓰지 않는다).
def test_nothing_to_do_loads_no_model(tmp_path, capsys):
    assert qwen.main(["--arm", "Q", "--split", "dev", "--audio", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().err.strip())["to_do"] == 0


# read_wav가 16비트 정수 PCM을 32768로 나눈 float32로 그대로 돌려주는지 확인한다(최댓값·최솟값 포함).
# faster-whisper의 decode_audio와 같은 값을 줘야 Qwen 팔과 Whisper 팔이 같은 소리를 듣는다(문서 "소리" 절에서
# 확인한 내용). numpy가 없으면 건너뛴다.
def test_read_wav_gives_the_samples_the_wav_holds(tmp_path):
    np = pytest.importorskip("numpy")
    pcm = np.array([0, 1, -1, 32767, -32768, 1234], dtype="<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm.tobytes())
    path = tmp_path / "x.wav"
    path.write_bytes(buf.getvalue())
    got = qwen.read_wav(path)
    assert got.dtype == np.float32 and list(got) == list(pcm.astype(np.float32) / 32768.0)
