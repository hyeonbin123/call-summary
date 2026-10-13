"""LoRA / QLoRA fine-tuning of a student on (transcript -> after-call record JSON).

The prompt is exactly the evaluation prompt (0-shot, `prompts.build_messages`), and the loss covers only the
answer tokens. fp16 AMP: the RTX 2080 Ti has no bf16. LoRA weights are kept in fp32.
"""

# [파일 역할] 3·4단계 학습 스크립트. 학생 모델(Qwen3-1.7B는 LoRA, Qwen3-4B는 QLoRA 4bit)을 "전사 → 상담 기록
# JSON"으로 미세 조정하고 어댑터를 outputs/train/<이름>/에 남긴다.
# [설계 이유]
# - 프롬프트는 평가와 똑같이 만든다(p1, 0-shot, prompts.build_messages). 학습과 평가·서빙의 입력이 어긋나지
#   않게 하려는 것이다(docs/experiments.md 3단계 규칙).
# - 손실은 답(JSON) 토큰에만 준다. 프롬프트 부분은 IGNORE로 가린다(같은 규칙).
# - RTX 2080 Ti에는 bf16이 없어 fp16 자동 혼합 정밀도(AMP)로 학습하고, 학습하는 LoRA 가중치만 fp32로
#   둔다(docs/plan.md "모델 후보").
# - Windows에서는 VRAM이 넘쳐도 오류가 나지 않고 시스템 메모리로 새어 20배쯤 느려졌다. 그래서 답 구간만 로짓을
#   계산하고(tail_len·tail_loss) PyTorch 할당 상한(cap_vram)을 둔다(docs/experiments.md 3단계 진행 기록).
# - 계획은 trl을 "편하면 쓰고 아니면 직접 루프"로 남겨 두었고(docs/plan.md 스택), 여기서는 직접 짠 짧은 루프를
#   쓴다.
# [흐름] 인수 → 할당 상한·seed → 데이터 고르기(take_balanced)·인코딩(encode) → 바탕 모델 불러오기(QLoRA면
# 4bit) → LoRA 붙이기 → AdamW·코사인 스케줄·GradScaler → 학습 루프(기울기 누적) → epoch마다·마지막에 어댑터
# 저장.
# [연결] 출력의 train_config.json을 export.py(QLoRA 여부)와 evaluate.py(모델 지문)가 읽는다. providers.py가
# cap_vram을 가져다 추론에도 쓴다.

from __future__ import annotations

# torch·peft·transformers는 무거워서 main 안에서 불러온다. 그래서 encode·take_balanced 같은 도우미는 GPU
# 라이브러리 없이 테스트할 수 있다(tests/test_gen.py).
import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# 평가와 같은 프롬프트(build_messages), 정답 JSON 글(record_to_json: 학습 목표 글), 프롬프트 판과
# 지문(train_config.json에 남겨 어떤 프롬프트로 학습했는지 기록).
from .dataset import Item, load_items
from .prompts import PROMPT_VERSION, build_messages, prompt_hash, record_to_json

# 손실에서 뺄 위치의 라벨 값. PyTorch cross_entropy의 ignore_index 기본값(-100)과 같게 둔다.
IGNORE = -100

# 학습 루프 판. train_config.json에 남겨, 같은 인수로 학습한 어댑터라도 어느 루프로 만들었는지
# 가린다(evaluate가 이 파일의 해시를 모델 지문으로 쓴다). 1: 2026-10-13 고치기 전, 기울기 누적 묶음이 epoch
# 경계를 넘어 이어졌다. 2: 묶음이 epoch 안에서 끝난다(accum_plan).
TRAIN_LOOP_VERSION = 2


# 학습 건 중 n건을 업종마다 고르게 고른다. 업종을 이름순으로 돌며 파일 순서대로 하나씩 가져가므로 결정적이다.
# n이 없거나 전체 이상이면 전체를 쓴다.
# 쓰는 곳: --n-train(데이터 크기 곡선: train 500건 대 1,500건, docs/experiments.md 3단계).
def take_balanced(items: list[Item], n: int | None) -> list[Item]:
    """First n items spread evenly over domains, in file order (deterministic)."""
    if n is None or n >= len(items):
        return list(items)
    by_dom: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_dom[it.domain].append(it)
    doms = sorted(by_dom)
    out: list[Item] = []
    i = 0
    while len(out) < n:
        progressed = False
        for d in doms:
            if i < len(by_dom[d]) and len(out) < n:
                out.append(by_dom[d][i])
                progressed = True
        if not progressed:
            break
        i += 1
    return out


# 학습 예시 하나를 토큰 id와 라벨로 바꾼다. 프롬프트는 모델의 채팅 템플릿으로 만들고(생성 시작 표시 포함, 생각
# 모드 끔: Qwen3 템플릿은 빈 think 블록을 넣는다), 목표는 정답 JSON 글 + 끝 토큰이다.
# 프롬프트와 목표를 따로 토큰화해 이어 붙이므로 둘의 경계가 정확히 나뉜다. 프롬프트 위치의 라벨은 모두
# IGNORE라 손실은 답에서만 생긴다. 끝 토큰까지 배워야 추론 때 JSON 뒤에서 멈춘다.
# max_len을 넘으면 None을 돌려 그 예시를 뺀다(main이 뺀 수를 기록).
def encode(tok, item: Item, max_len: int, enable_thinking: bool = False) -> dict | None:
    """Token ids with the prompt masked out. None if the example does not fit in max_len."""
    messages = build_messages(item.domain, item.transcript)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    target = record_to_json(item.gold()) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    ids = p_ids + t_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "labels": [IGNORE] * len(p_ids) + t_ids}


# 배치에서 로짓이 필요한 마지막 위치 수 K를 구한다. 배치 안에서 가장 먼저 시작하는 답 토큰 위치를 찾아, 그
# 바로 앞 위치(첫 답 토큰을 예측하는 위치)부터 끝까지를 센다. model(..., logits_to_keep=K)에 넘긴다.
def tail_len(labels) -> int:
    """How many final positions need logits: from the position before the first target token to the end."""
    import torch

    is_target = labels != IGNORE
    # argmax는 행마다 처음 참인 위치를 준다(답 토큰이 있는 행이라고 가정). 그중 가장 이른 위치를 쓴다.
    first = int(torch.argmax(is_target.int(), dim=1).min())  # earliest target position in the batch
    return labels.shape[1] - first + 1


# 마지막 K 위치의 로짓만으로 답 토큰의 평균 교차 엔트로피를 낸다. 답 토큰이 모두 그 창 안에 있으면 전체 길이
# 손실과 값이 같다(tests/test_train_loss.py가 작은 Qwen3 모델로 확인).
# 왜: 어휘 약 15만 개 × 대화 전체 길이의 로짓이 학습에서 가장 큰 텐서였다. 답 구간만 계산해 메모리를
# 줄였다(docs/experiments.md 3단계 진행 기록: 할당 상한과 함께 넣은 뒤 스텝당 283초 → 21.6초).
def tail_loss(logits, labels):
    """Mean cross-entropy over target tokens when `logits` cover only the last K positions.

    logits[:, j] belongs to position L-K+j and predicts token L-K+j+1, so logits[:, :-1] line up with
    labels[:, L-K+1:]. Equal to the full-sequence loss when all target tokens lie in that window.
    """
    import torch.nn.functional as F

    # 로짓은 다음 토큰을 예측하므로 마지막 로짓(문장 뒤를 예측)은 버리고, 라벨은 창의 둘째 위치부터 맞춘다.
    # 로짓은 손실 계산 전에 fp32로 올려 fp16 오차를 줄인다.
    k = logits.shape[1]
    targets = labels[:, labels.shape[1] - k + 1 :]
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE
    )


# PyTorch 할당기가 이 GPU에서 max_gb 이상 쓰지 못하게 한다(set_per_process_memory_fraction). 상한을 넘으면
# 느려지는 대신 메모리 부족 오류가 나므로 문제를 바로 알 수 있다. 학습 기본값 8.5 GB, 추론은 providers.py가
# 9.5 GB로 부른다(docs/experiments.md 3단계 진행 기록). 0이나 None이면 끈다.
def cap_vram(max_gb: float | None) -> None:
    """Keep PyTorch's allocator under max_gb of this GPU. On Windows, going past the card's memory does
    not fail: the driver spills to system RAM and everything runs many times slower."""
    if not max_gb:
        return
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb * 1024**3 / total), 0)


# 한 epoch의 미니배치 n_micro개를 기울기 누적 묶음으로 나눈다. 미니배치마다 (그 묶음의 크기, 이 미니배치 뒤에
# optimizer 스텝을 하는지)를 돌려준다. 묶음은 grad_accum개씩이고 epoch의 마지막 묶음은 남은 것만 갖는다
# (1,500건, 누적 16이면 16 × 93 + 12). 그래서 한 epoch는 정확히 ceil(n_micro / grad_accum) 스텝이고, epoch<n>
# 체크포인트는 그 epoch까지의 데이터만 본다. 손실을 묶음 크기로 나누면 스텝마다 묶음 안 평균이 된다.
# 왜: 고치기 전 루프는 미니배치 수를 epoch를 넘어 이어 세어, epoch 끝에 남은 미니배치가 다음 epoch 앞
# 미니배치와 한 스텝으로 합쳐졌다(1,500건이면 epoch1 저장에 둘째 epoch 4건, 마지막 저장에 셋째 epoch 8건이
# 들어감, docs/experiments.md "학습 epoch 경계와 지연 백분위 바로잡기").
def accum_plan(n_micro: int, grad_accum: int) -> list[tuple[int, bool]]:
    """(size of the micro-batch's accumulation group, whether the optimizer steps after it), for each
    micro-batch of one epoch. Groups never cross an epoch: the last one takes the remaining micro-batches."""
    plan = []
    for i in range(n_micro):
        start = i - i % grad_accum
        size = min(grad_accum, n_micro - start)
        plan.append((size, i == start + size - 1))
    return plan


# 명령줄 입구. 기본값은 3단계 규칙의 설정이다: LoRA r=16, alpha=32, dropout 0.05, 모든 선형층, 학습률 2e-4
# 코사인, 2 epoch(docs/experiments.md 3단계). 배치 1 × 기울기 누적 16이 한 optimizer 스텝이다.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--n-train", type=int, help="use only this many training items (balanced over domains)")
    ap.add_argument("--out", required=True, help="output dir (adapter, logs)")
    ap.add_argument("--qlora", action="store_true", help="load the base model in 4-bit (nf4)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--max-vram-gb", type=float, default=8.5, help="allocator cap; 0 disables")
    args = ap.parse_args(argv)

    # 학습 라이브러리는 여기서 불러온다. peft: LoRA 어댑터와 4bit 학습 준비, transformers: 바탕
    # 모델·토크나이저·코사인 스케줄.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    # 모델을 올리기 전에 할당 상한을 건다. torch와 데이터 섞기 순서 모두 seed로 고정한다.
    cap_vram(args.max_vram_gb)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 모든 예시를 미리 인코딩하고, 길이 분포(p50, 최대)와 길이 초과로 뺀 수를 찍는다.
    items = take_balanced(load_items(args.train), args.n_train)
    tok = AutoTokenizer.from_pretrained(args.model)
    # 패딩 토큰이 없는 토크나이저는 끝 토큰을 패딩에 쓴다. 패딩 위치는 라벨 IGNORE·마스크 0이라 학습에 영향이
    # 없다.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    examples = []
    dropped = 0
    for it in items:
        ex = encode(tok, it, args.max_len)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    lengths = sorted(len(e["input_ids"]) for e in examples)
    print(
        f"{len(examples)} examples ({dropped} over {args.max_len} tokens dropped); "
        f"length p50={lengths[len(lengths) // 2]} max={lengths[-1]}",
        file=sys.stderr,
    )

    # 바탕 모델은 fp16으로 GPU 0에 통째로 올린다. --qlora면 nf4 4bit로 양자화해 올리고 계산은 fp16으로
    # 한다(double quant 켬). 평가 때 4bit로 불러오는 providers.py는 double quant를 켜지 않아 설정이 조금
    # 다르다(docs/experiments.md 4단계 한계: 모든 4bit 평가가 같은 설정이라 비교에는 영향이 없다). export.py는
    # 학습과 같은 설정으로 4bit 바탕을 풀어 합친다.
    kwargs: dict = {"dtype": torch.float16, "device_map": {"": 0}}
    if args.qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    # 학습 중에는 KV 캐시가 필요 없어 끈다(gradient checkpointing과도 맞지 않는다). gradient checkpointing은
    # 중간 활성값을 저장하지 않고 역전파 때 다시 계산해 메모리를 아낀다.
    # QLoRA는 peft의 prepare_model_for_kbit_training(4bit 학습 준비 + checkpointing)을 쓰고, LoRA는 직접 켠다.
    # enable_input_require_grads는 바탕 가중치가 얼어 있어도 checkpointing 구간에 기울기가 흐르게 한다.
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    # LoRA: 모든 선형층("all-linear")에 순위 r의 저순위 행렬을 붙이고 그것만 학습한다.
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # 학습하는 파라미터(LoRA 가중치)를 fp32로 올린다. fp16 AMP에서 GradScaler는 fp16 파라미터의 기울기를
    # unscale하지 못하고, 작은 업데이트가 fp16에서 사라지는 것도 막는다(docstring: LoRA weights are kept in
    # fp32).
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}", file=sys.stderr)

    # AdamW(가중치 감쇠 없음), 워밍업 3% 뒤 코사인으로 줄어드는 학습률. 스텝 수는 epoch 수 × ceil(예시 수 /
    # (배치 × 누적))이다. GradScaler는 fp16 손실을 키워 기울기가 0으로 사라지지 않게 하고, inf·NaN 기울기가
    # 나온 스텝은 건너뛴다.
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    # 한 epoch의 누적 묶음(accum_plan). 스텝 수가 아래 steps_per_epoch와 같다.
    plan = accum_plan(math.ceil(len(examples) / args.batch), args.grad_accum)
    steps_per_epoch = math.ceil(len(examples) / (args.batch * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    # 학습 설정 전체(인수, 예시 수, 스텝 수, 프롬프트 판·지문, 학습 파라미터 수, 학습 루프 판)를
    # train_config.json으로 남긴다. export.py가 qlora 값을, evaluate.py가 이 파일의 해시를 모델 지문으로
    # 읽는다. 학습 루프 판이 없는 파일은 고치기 전 루프(판 1)로 만든 것이다.
    config = {
        **vars(args),
        "n_examples": len(examples),
        "dropped": dropped,
        "total_steps": total_steps,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "trainable_params": trainable,
        "train_loop": TRAIN_LOOP_VERSION,
    }
    (out / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")

    # 한 epoch의 미니배치를 만든다. 매 epoch 순서를 seed 고정 rng로 섞고, 배치 안에서 가장 긴 예시에 맞춰
    # 오른쪽을 패딩한다(라벨 IGNORE, 어텐션 마스크 0).
    def batches():
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            chunk = [examples[j] for j in order[i : i + args.batch]]
            width = max(len(e["input_ids"]) for e in chunk)
            ids = [e["input_ids"] + [tok.pad_token_id] * (width - len(e["input_ids"])) for e in chunk]
            labels = [e["labels"] + [IGNORE] * (width - len(e["labels"])) for e in chunk]
            mask = [[1] * len(e["input_ids"]) + [0] * (width - len(e["input_ids"])) for e in chunk]
            yield (
                torch.tensor(ids, device="cuda"),
                torch.tensor(labels, device="cuda"),
                torch.tensor(mask, device="cuda"),
            )

    model.train()
    step = 0
    running = 0.0
    running_n = 0
    t0 = time.time()
    epoch = 0
    done = False
    while not done:
        epoch += 1
        # 미니배치마다 누적 묶음의 크기와 스텝 여부를 함께 받는다. 묶음은 epoch마다 새로 시작한다.
        for (ids, labels, mask), (group, step_now) in zip(batches(), plan, strict=True):
            # 순전파는 fp16 autocast로, 로짓은 답 구간만(tail_len) 받는다.
            with torch.autocast("cuda", dtype=torch.float16):
                # Logits only for the answer span: a 150k-token vocabulary over the whole prompt is the
                # largest tensor in training and pushed the process past the card's memory.
                logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=tail_len(labels)).logits
            # 기울기를 묶음 크기(보통 grad_accum, epoch 끝 묶음은 남은 수)만큼 모아 한 스텝으로 쓰므로 손실을
            # 미리 그 크기로 나눈다. 유한하지 않은 손실은 기록만 하고 넘어간다(GradScaler가 그 스텝을
            # 건너뛴다). 3단계 학습에서는 이런 스텝이 없었다(docs/experiments.md 3단계 진행 기록).
            loss = tail_loss(logits, labels) / group
            if not torch.isfinite(loss):
                print(f"non-finite loss at step {step}", file=sys.stderr)
                log.write(json.dumps({"step": step, "event": "nonfinite_loss"}) + "\n")
            scaler.scale(loss).backward()
            running += loss.item() * group
            running_n += 1
            # 묶음이 다 차지 않았으면 다음 미니배치로. 묶음은 epoch 경계를 넘지 않으므로 epoch의 마지막
            # 미니배치 뒤에는 늘 스텝을 한다(accum_plan).
            if not step_now:
                continue
            # 스텝: 기울기를 원래 크기로 되돌린 뒤 노름 1.0으로 자르고, optimizer·scaler·스케줄을 한 칸
            # 진행한다.
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            # log_every 스텝마다 평균 손실, 기울기 노름, 학습률, 손실 배율, 경과 시간, 최대 GPU 메모리를
            # train_log.jsonl과 표준 오류에 남긴다. 백그라운드 학습이 멈추지 않았는지 로그로 지켜볼 수 있다.
            if step % args.log_every == 0 or step == total_steps:
                rec = {
                    "step": step,
                    "epoch": round(step / steps_per_epoch, 3),
                    "loss": round(running / running_n, 5),
                    "grad_norm": round(float(gnorm), 4),
                    "lr": sched.get_last_lr()[0],
                    "scale": scaler.get_scale(),
                    "elapsed_s": round(time.time() - t0, 1),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), file=sys.stderr, flush=True)
                running, running_n = 0.0, 0
            # epoch에 해당하는 스텝 수마다 어댑터를 epoch<n>/에 저장한다(1 epoch 대 2 epoch를 dev에서 비교,
            # docs/experiments.md 3단계 결과). --max-steps에 닿으면 멈춘다.
            if step % steps_per_epoch == 0:
                model.save_pretrained(out / f"epoch{step // steps_per_epoch}")
            if step >= total_steps:
                done = True
                break
    # 마지막 어댑터와 토크나이저를 final/에 저장한다. evaluate.py(--adapter)와 export.py가 이 폴더를 쓴다.
    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    log.write(json.dumps({"event": "done", "steps": step, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    log.close()
    print(f"-> {out / 'final'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
