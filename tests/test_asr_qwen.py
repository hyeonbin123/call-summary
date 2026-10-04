"""scripts/asr_qwen.py: the parts that need no model (and no numpy or torch, except where skipped)."""

import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qwen = _load("asr_qwen", ROOT / "scripts" / "asr_qwen.py")


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


def test_pinned_model_and_greedy_decoding():
    assert qwen.MODEL == "Qwen/Qwen3-ASR-1.7B-hf"
    assert qwen.REVISION == "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    assert qwen.GENERATE == {"do_sample": False, "num_beams": 1, "max_new_tokens": 256}
    assert qwen.ARMS == {"Q": False, "Qc": True}


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


def test_the_context_arm_refuses_to_run_without_context(tmp_path, capsys):
    with pytest.raises(SystemExit):
        qwen.main(["--arm", "Qc", "--split", "dev", "--audio", str(tmp_path)])
    assert "needs --context" in capsys.readouterr().err


def test_nothing_to_do_loads_no_model(tmp_path, capsys):
    assert qwen.main(["--arm", "Q", "--split", "dev", "--audio", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().err.strip())["to_do"] == 0


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
