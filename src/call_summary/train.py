"""LoRA / QLoRA fine-tuning of a student on (transcript -> after-call record JSON).

The prompt is exactly the evaluation prompt (0-shot, `prompts.build_messages`), and the loss covers only the
answer tokens. fp16 AMP: the RTX 2080 Ti has no bf16. LoRA weights are kept in fp32.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from .dataset import Item, load_items
from .prompts import PROMPT_VERSION, build_messages, prompt_hash, record_to_json

IGNORE = -100


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


def tail_len(labels) -> int:
    """How many final positions need logits: from the position before the first target token to the end."""
    import torch

    is_target = labels != IGNORE
    first = int(torch.argmax(is_target.int(), dim=1).min())  # earliest target position in the batch
    return labels.shape[1] - first + 1


def tail_loss(logits, labels):
    """Mean cross-entropy over target tokens when `logits` cover only the last K positions.

    logits[:, j] belongs to position L-K+j and predicts token L-K+j+1, so logits[:, :-1] line up with
    labels[:, L-K+1:]. Equal to the full-sequence loss when all target tokens lie in that window.
    """
    import torch.nn.functional as F

    k = logits.shape[1]
    targets = labels[:, labels.shape[1] - k + 1 :]
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE
    )


def cap_vram(max_gb: float | None) -> None:
    """Keep PyTorch's allocator under max_gb of this GPU. On Windows, going past the card's memory does
    not fail: the driver spills to system RAM and everything runs many times slower."""
    if not max_gb:
        return
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gb * 1024**3 / total), 0)


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

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    cap_vram(args.max_vram_gb)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = take_balanced(load_items(args.train), args.n_train)
    tok = AutoTokenizer.from_pretrained(args.model)
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
    model.config.use_cache = False
    if args.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable:,}", file=sys.stderr)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(examples) / (args.batch * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda")

    config = {
        **vars(args),
        "n_examples": len(examples),
        "dropped": dropped,
        "total_steps": total_steps,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": prompt_hash(),
        "trainable_params": trainable,
    }
    (out / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")

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
    micro = 0
    running = 0.0
    running_n = 0
    t0 = time.time()
    epoch = 0
    done = False
    while not done:
        epoch += 1
        for ids, labels, mask in batches():
            with torch.autocast("cuda", dtype=torch.float16):
                # Logits only for the answer span: a 150k-token vocabulary over the whole prompt is the
                # largest tensor in training and pushed the process past the card's memory.
                logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=tail_len(labels)).logits
            loss = tail_loss(logits, labels) / args.grad_accum
            if not torch.isfinite(loss):
                print(f"non-finite loss at step {step}", file=sys.stderr)
                log.write(json.dumps({"step": step, "event": "nonfinite_loss"}) + "\n")
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            running_n += 1
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
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
            if step % steps_per_epoch == 0:
                model.save_pretrained(out / f"epoch{step // steps_per_epoch}")
            if step >= total_steps:
                done = True
                break
    model.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")
    log.write(json.dumps({"event": "done", "steps": step, "elapsed_s": round(time.time() - t0, 1)}) + "\n")
    log.close()
    print(f"-> {out / 'final'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
