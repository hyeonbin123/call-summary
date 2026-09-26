"""Inference backends behind one interface: chat messages in, raw reply text out.

- ScriptedProvider: canned replies, for tests.
- OllamaProvider: local Ollama over HTTP (baselines, teacher, judge, and GGUF-served students).
- HFProvider: transformers + optional PEFT adapter (students before GGUF export). Heavy imports are lazy.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Reply:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Provider(Protocol):
    name: str

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply: ...


@dataclass
class ScriptedProvider:
    """Returns replies from a function of the messages (or a fixed list, in order)."""

    replies: list[str] | Callable[[list[dict]], str]
    name: str = "scripted"
    calls: list[list[dict]] = field(default_factory=list)

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        self.calls.append(messages)
        if callable(self.replies):
            text = self.replies(messages)
        else:
            text = self.replies[len(self.calls) - 1]
        return Reply(text=text, latency_s=0.0)


@dataclass
class OllamaProvider:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    temperature: float = 0.0
    num_ctx: int = 8192
    num_predict: int = 1024
    num_gpu: int | None = None  # layers on GPU; None lets Ollama decide (it can overfill VRAM on Windows)
    seed: int = 0
    keep_alive: str = "30m"
    timeout_s: float = 600.0
    use_schema: bool = False  # pass the JSON schema as Ollama's `format` (constrained decoding)
    think: bool | None = False  # Qwen3-style thinking models: off by default for a fair, fast baseline

    @property
    def name(self) -> str:
        return f"ollama:{self.model}" + (":schema" if self.use_schema else "")

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        import httpx

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "seed": self.seed,
            },
        }
        if self.num_gpu is not None:
            body["options"]["num_gpu"] = self.num_gpu
        if self.think is not None:
            body["think"] = self.think
        if self.use_schema and json_schema is not None:
            body["format"] = json_schema
        t0 = time.perf_counter()
        r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return Reply(
            text=data["message"]["content"],
            latency_s=time.perf_counter() - t0,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def loaded_models(self) -> list[dict]:
        import httpx

        r = httpx.get(f"{self.base_url}/api/ps", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])


@dataclass
class HFProvider:
    """Greedy decoding with transformers. fp16 on this GPU (no bf16 on Turing)."""

    model_id: str
    adapter: str | None = None
    load_4bit: bool = False
    max_new_tokens: int = 1024
    enable_thinking: bool = False
    _model: object = field(default=None, init=False, repr=False)
    _tok: object = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return (
            f"hf:{self.model_id}"
            + (f"+{self.adapter}" if self.adapter else "")
            + (":4bit" if self.load_4bit else "")
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict = {"dtype": torch.float16, "device_map": "cuda"}
        if self.load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        self._model, self._tok = model, tok

    def generate(self, messages: list[dict], json_schema: dict | None = None) -> Reply:
        return self.generate_batch([messages])[0]

    def generate_batch(self, batch: list[list[dict]]) -> list[Reply]:
        """Greedy decoding of several conversations at once (left padding). Latency is the batch's
        wall time divided evenly, so per-item latency is only comparable between runs with batch size 1."""
        import torch

        self._load()
        tok, model = self._tok, self._model
        prompts = [
            tok.apply_chat_template(  # type: ignore[attr-defined]
                m, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking
            )
            for m in batch
        ]
        tok.padding_side = "left"  # type: ignore[attr-defined]
        if tok.pad_token is None:  # type: ignore[attr-defined]
            tok.pad_token = tok.eos_token  # type: ignore[attr-defined]
        # Same tokenization as training (train.encode): the chat template already holds special tokens.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")  # type: ignore[operator]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(  # type: ignore[attr-defined]
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,  # type: ignore[attr-defined]
            )
        latency = (time.perf_counter() - t0) / len(batch)
        width = enc["input_ids"].shape[1]
        replies = []
        for i in range(len(batch)):
            new = out[i, width:]
            n_new = int((new != tok.pad_token_id).sum())  # type: ignore[attr-defined]
            replies.append(
                Reply(
                    text=tok.decode(new, skip_special_tokens=True),  # type: ignore[attr-defined]
                    latency_s=latency,
                    prompt_tokens=int(enc["attention_mask"][i].sum()),
                    completion_tokens=n_new,
                )
            )
        return replies
