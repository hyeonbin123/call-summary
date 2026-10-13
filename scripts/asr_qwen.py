"""Speech condition v3, pass 2 for the Qwen3-ASR arms: Qwen3-ASR-1.7B on the WAVs kept by pass 1.

It runs in this project's own environment (transformers 5.17 has the model class; numpy and torch from the
`train` group), on the audio scripts/asr_condition.py --keep-audio kept:

    uv run --no-sync python scripts/asr_qwen.py --arm Q --split dev
    uv run --no-sync python scripts/asr_qwen.py --arm Qc --split dev --context outputs/asr3/context.json

Decoding follows whisper-ko-ft stage 9 (`whisper_ko_ft.evaluate_qwen`): the Transformers classes at a pinned
revision, fp16, SDPA attention, greedy, the language forced through the official prompt
(`apply_transcription_request(..., language="Korean")`), at most 256 new tokens, left padding. Qc passes the
domain's context (`python -m call_summary.asr_arms context`) as the official `prompt` (the system message).
Batch 8 and a per-process GPU memory cap (default 9.5 GB) keep it inside the 11 GB card.

Output rows are those of asr_condition's pass 2 ({key, split, item_id, turn, arm, heard}) plus the official
parse with its repetition fix (`fixed`), the generated token count, whether the token limit was hit, and
whether any logit of the utterance was not finite (fp16 overflow). `heard` is the raw text after "<asr_text>";
the student gets it after the identifier rule (call_summary.id_itn), not the fixed one.
"""

# [파일 역할] 음성 조건 v3(인식기 바꾸기)의 2패스 가운데 Qwen3-ASR 팔(Q, Qc)을 돌리는 스크립트다.
# 1패스(asr_condition.py --keep-audio)가 남긴 WAV를 Qwen3-ASR-1.7B로 다시 받아 적어 팔의 hyp 파일(발화마다 한
# 줄)을 쓴다.
# [왜 필요한가] whisper-ko-ft 9단계에서 학습하지 않은 Qwen3-ASR-1.7B가 FLEURS 한국어에서 기준 인식기 turbo보다
# CER이 낮았다. 그래서 이 프로젝트의 전화 조건 음성에서도 나은지 같은 소리로 비교했다(docs/experiments.md
# "음성 조건 v3" 왜). 결과는 번호의 숫자를 빠뜨리거나 한글로 적어 turbo보다 크게 나빴다(관문 Q+I 72, Qc+I 83
# 대 T 107).
# [설계 이유] faster-whisper 팔과 달리 이 프로젝트 환경(transformers 5.17)에서 돈다: Qwen3-ASR 모델 클래스가
# transformers 5.x에 있고, MeloTTS가 쓰는 support-agent 음성 환경은 transformers 4.27이다. 디코딩은
# whisper-ko-ft 9단계 그대로(고정 revision, fp16, SDPA, 탐욕적, 언어 Korean 고정, 새 토큰 최대 256)이고,
# 배치를 16에서 8로 줄이고 프로세스 GPU 메모리 상한(9.5 GB)을 둬 11 GB 카드 안에 머문다(규칙).
# [연결] 출력 hyp 파일은 call_summary.asr_arms(gate, build)가 읽어 팔의 데이터셋을 만들고, 이때 식별자
# 규칙(id_itn)을 씌운다.

from __future__ import annotations

# wave: 1패스가 남긴 16비트 WAV를 표준 라이브러리로 읽어 numpy 배열로 바꾼다(docs/experiments.md 음성 조건 v3
# "소리").
import argparse
import json
import sys
import time
import wave
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

# 같은 scripts/ 폴더의 asr_condition.py에서 1패스 저장소(WavStore)와 JSONL·커밋·hyp 메타 도우미를 가져온다.
# 스크립트 폴더를 import 경로에 넣어야 해서 import가 이 줄 뒤에 온다(그래서 E402 예외 표시가 붙어 있다). 두
# 패스가 같은 기록 형식을 쓰게 하려는 것이다.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_condition import (  # noqa: E402
    DEFAULT_AUDIO,
    LISTEN_RATE,
    WavStore,
    _append_jsonl,
    _git_commit,
    _read_jsonl,
    write_hyp_meta,
)

# MODEL, REVISION: 모델과 고정 revision(whisper-ko-ft 9단계와 같은 가중치를 쓰려고 커밋 해시로 고정).
# LANGUAGE: 언어를 Korean으로 강제한다. MARKER: 공식 출력에서 받아 적은 글이 시작되는 표시.
# GENERATE: 탐욕적 디코딩(샘플링 없음, 빔 1), 새 토큰 최대 256. ARMS: 팔 이름 → 업종 맥락을 프롬프트로
# 줄지(Qc만).
MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"  # downloaded 2026-10-04 (whisper-ko-ft stage 9)
LANGUAGE = "Korean"
MARKER = "<asr_text>"
GENERATE = {"do_sample": False, "num_beams": 1, "max_new_tokens": 256}
ARMS = {"Q": False, "Qc": True}  # arm -> the domain context as the prompt


# 디코딩한 글에서 공식 파서처럼 assistant 응답과 <asr_text> 뒤만 꺼내되, 반복 제거는 하지 않은 글(raw)을
# 돌려준다. 규칙이 학생에게 raw를 주고 반복 제거한 글(fixed)은 기록만 하기로 정했다(docs/experiments.md 음성
# 조건 v3).
def raw_transcription(text: str) -> str:
    """The official parse of one decoded output without its repetition fix (as whisper-ko-ft stage 9)."""
    text = text.strip()
    if "assistant\n" in text:
        text = text.split("assistant\n", 1)[-1]
    if MARKER in text:
        text = text.split(MARKER, 1)[1]
    return text.strip()


# 16비트 모노 16 kHz WAV를 [-1, 1) float32로 읽는다(32768로 나눔). faster-whisper의 decode_audio와 표본 값이
# 같음을 확인했으므로 모든 팔이 같은 소리를 듣는다(docs/experiments.md 음성 조건 v3 "소리"). 형식이 다르면
# 멈춘다.
def read_wav(path: Path):
    """16-bit mono 16 kHz WAV -> float32 in [-1, 1), as faster-whisper's decode_audio gives it."""
    import numpy as np

    with wave.open(str(path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, LISTEN_RATE):
            raise ValueError(f"{path}: not 16-bit mono {LISTEN_RATE} Hz")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


# 그 분할의 index 행 중 소리가 있고(wav가 None이 아님) 아직 hyp에 없는 것. 이어 하기와 --limit(스모크)을 위한
# 것이다.
def todo_rows(store: WavStore, split: str, done: set[str], limit: int | None = None) -> list[dict]:
    rows = [r for r in store.split_rows(split) if r["wav"] is not None and r["key"] not in done]
    return rows[:limit] if limit else rows


# 행 목록을 size개씩 끊어 내준다.
def batched(rows: list[dict], size: int) -> Iterator[list[dict]]:
    for begin in range(0, len(rows), size):
        yield rows[begin : begin + size]


# 생성한 토큰 행마다 (첫 끝 토큰 앞까지의 토큰 수, 끝 토큰 없이 상한에 닿았는지). 토큰 상한에 닿은 발화는 반복
# 등으로 잘렸을 수 있어 2차 지표로 센다(결과: 0개).
def generated_lengths(generated, eos_token_id) -> list[tuple[int, bool]]:
    """(tokens before the first end token, ran into the token limit) for each row."""
    import torch

    eos = torch.tensor(eos_token_id if isinstance(eos_token_id, list | tuple) else [eos_token_id])
    ended = torch.isin(generated.cpu(), eos)
    out = []
    for row in ended:
        hits = row.nonzero()
        out.append((int(hits[0]), False) if len(hits) else (int(row.shape[0]), True))
    return out


# fp16 넘침 감시: lm_head에 forward hook을 걸어, 생성 중 한 번이라도 로짓에 inf/NaN이 나온 배치 행을 표시한다.
# fp16만 되는 GPU(RTX 2080 Ti, bf16 없음)라 넘침이 생기면 출력이 망가질 수 있어 발화마다 기록한다(결과: 0개).
class NonFiniteWatch:
    """Flags the rows of a batch whose logits were ever not finite (an fp16 overflow), through a hook."""

    def __init__(self, head) -> None:
        self.rows = None
        head.register_forward_hook(self._hook)

    # 생성은 토큰마다 lm_head를 다시 부르므로, 행별 이상 여부를 OR로 누적한다.
    def _hook(self, module, inputs, output) -> None:
        import torch

        bad = ~torch.isfinite(output).flatten(1).all(dim=1)
        self.rows = bad if self.rows is None else self.rows | bad

    # 배치 하나가 끝나면 누적값을 꺼내고 초기화한다. hook이 한 번도 불리지 않았으면 모두 False.
    def take(self, n: int) -> list[bool]:
        rows, self.rows = self.rows, None
        return [False] * n if rows is None else [bool(x) for x in rows.tolist()]


# 프로세서와 모델을 고정 revision으로 올린다. cuda에서는 set_per_process_memory_fraction으로 GPU 메모리 상한을
# 걸어, Windows에서 VRAM을 넘겨 공유 메모리로 새어 느려지는 것을 막는다. 어텐션은 SDPA.
def load(device: str, dtype_name: str, max_memory_gb: float | None):
    import torch
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype_name]
    if device == "cuda" and max_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_memory_gb * 2**30 / total))
    processor = AutoProcessor.from_pretrained(MODEL, revision=REVISION)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(device).eval(), processor, dtype


# 배치 하나를 받아 적는다. apply_transcription_request가 소리와 언어(필요하면 맥락 프롬프트)로 공식 입력을
# 만들고, generate 뒤 입력 길이 이후만 새 토큰으로 디코딩한다. 발화마다 raw 글(heard), 공식 반복 제거
# 글(fixed), 새 토큰 수, 상한 도달, fp16 넘침을 돌려준다.
def transcribe(
    model, processor, audio: list, prompts: list[str] | None, device: str, dtype, watch
) -> list[dict]:
    import torch

    inputs = processor.apply_transcription_request(audio, language=LANGUAGE, prompt=prompts).to(device, dtype)
    with torch.inference_mode():
        output = model.generate(**inputs, **GENERATE)
    generated = output[:, inputs["input_ids"].shape[1] :]
    texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
    fixed = processor.extract_transcription(texts)
    lengths = generated_lengths(generated, model.generation_config.eos_token_id)
    bad = watch.take(len(texts))
    return [
        {
            "heard": raw_transcription(t),
            "fixed": f,
            "new_tokens": n,
            "hit_token_limit": hit,
            "nonfinite_logits": b,
        }
        for t, f, (n, hit), b in zip(texts, fixed, lengths, bad, strict=True)
    ]


# 명령줄 진입점. Qc는 --context(asr_arms context가 만든 업종 맥락 JSON)가 꼭 필요하다. CPU 실행은 측정 조건이
# 아니므로 기본 hyp 파일에 섞이지 않게 별도 --hyp를 요구한다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--audio", default=DEFAULT_AUDIO)
    ap.add_argument("--hyp", help="output (default <audio>/hyp/<ARM>.jsonl)")
    ap.add_argument("--context", help="per-domain context JSON (needed by Qc)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-memory-gb", type=float, default=9.5, help="per-process GPU memory cap")
    ap.add_argument("--precision", choices=["fp16", "fp32", "bf16"], default="fp16")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit", type=int, help="only the first N utterances to do (smoke)")
    args = ap.parse_args(argv)

    if ARMS[args.arm] and not args.context:
        ap.error(f"{args.arm} needs --context")
    if args.device == "cpu" and not args.hyp:
        ap.error("a CPU run is not the measured condition; give it its own --hyp")
    store = WavStore(Path(args.audio))
    hyp = Path(args.hyp) if args.hyp else store.root / "hyp" / f"{args.arm}.jsonl"
    # 이미 받아 적은 키는 건너뛴다(이어 하기). 할 일이 없으면 모델을 올리지 않고 끝낸다. torch·transformers는
    # 그 뒤에 불러온다.
    rows = todo_rows(store, args.split, {r["key"] for r in _read_jsonl(hyp)}, args.limit)
    context = json.loads(Path(args.context).read_text(encoding="utf-8")) if ARMS[args.arm] else None
    print(json.dumps({"arm": args.arm, "split": args.split, "to_do": len(rows)}), file=sys.stderr, flush=True)
    if not rows:
        return 0

    import torch
    import transformers

    model, processor, dtype = load(args.device, args.precision, args.max_memory_gb)
    # hyp 옆에 메타(팔, 모델, revision, 디코딩, 정밀도, 메모리 상한, 어텐션, 맥락, 버전, 커밋, 시작 시각)를 한
    # 번 쓴다. 이어 돌릴 때는 write_hyp_meta가 같은 팔·모델·디코딩인지 확인한다.
    watch = NonFiniteWatch(model.lm_head)
    write_hyp_meta(
        hyp,
        {
            "arm": args.arm,
            "model": MODEL,
            "revision": REVISION,
            "hotwords": ARMS[args.arm],
            "decoding": {**GENERATE, "language": LANGUAGE, "batch_size": args.batch_size},
            "precision": args.precision,
            "device": args.device,
            "max_memory_gb": args.max_memory_gb,
            "attention": model.config._attn_implementation,
            "context": context,
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "commit": _git_commit(),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    n: Counter = Counter()
    t0 = time.time()
    audio_seconds = 0.0
    for batch in batched(rows, args.batch_size):
        audio = [read_wav(store.wav_path(r["key"])) for r in batch]
        # Qc는 발화의 업종에 맞는 맥락 글(회사 이름 + 카탈로그 값)을 공식 prompt(시스템 메시지)로 준다.
        prompts = [context[r["domain"]] for r in batch] if context else None
        for row, out in zip(
            batch, transcribe(model, processor, audio, prompts, args.device, dtype, watch), strict=True
        ):
            # asr_condition.py의 2패스와 같은 꼴(key, split, item_id, turn, arm, heard)에 Qwen 전용 확인값을
            # 더해 한 줄씩 바로 쓴다(중간에 멈춰도 이어 할 수 있다).
            keep = {k: row[k] for k in ("key", "split", "item_id", "turn")}
            _append_jsonl(hyp, {**keep, "arm": args.arm, **out})
            n["recognised"] += 1
            n["hit_token_limit"] += out["hit_token_limit"]
            n["nonfinite"] += out["nonfinite_logits"]
            audio_seconds += row["audio_seconds"]
        # 약 25배치마다 진행 상황과 실시간 대비 속도(받아 적은 소리 길이 ÷ 걸린 시간)를 찍는다.
        if n["recognised"] % (args.batch_size * 25) < args.batch_size:
            rate = audio_seconds / max(time.time() - t0, 1e-9)
            print(
                f"[{args.arm} {args.split}] {n['recognised']}/{len(rows)}, {rate:.1f}x real time",
                file=sys.stderr,
                flush=True,
            )
    # 끝나면 건수, 상한 도달·넘침 수, 걸린 시간, 소리 길이, 최대 GPU 메모리(torch 할당 기준)를 표준 오류에
    # 찍는다. 문서의 "토큰 상한 도달 0, fp16 넘침 0, 최대 GPU 메모리 5,268 MB" 같은 수치가 여기서 나왔다.
    summary = {
        "arm": args.arm,
        "split": args.split,
        **n,
        "seconds": round(time.time() - t0, 1),
        "audio_seconds": round(audio_seconds, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
    }
    print(json.dumps(summary), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
