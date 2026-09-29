"""Run a model over a dataset file and score the structured fields.

Output: <out>/<run_id>/manifest.json, items.jsonl (raw reply + parse + per-item score), summary.json.
Official runs (--official) need a clean git tree and write under reports/ so they can be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .dataset import Item, load_items, write_jsonl
from .prompts import PROMPT_VERSION, PROMPT_VERSIONS, build_messages, prompt_hash
from .providers import HFProvider, OllamaProvider, Provider, Reply
from .schema import AfterCallRecord, parse_reply, record_json_schema
from .scoring import METRICS, ItemScore, bootstrap_ci, score_item, summarize

HEADLINE = (
    "schema",
    "exact",
    "category_acc",
    "resolution_acc",
    "entity_f1",
    "hallucination",
    "action_f1",
    "follow_up_acc",
)


def pick_shots(pool: Sequence[Item], domain: str, k: int, seed: int = 0) -> list[tuple[str, AfterCallRecord]]:
    """k fixed examples of one domain, different categories where possible. Needs reference summaries."""
    cands = [it for it in pool if it.domain == domain and it.summary]
    rng = random.Random(f"shots:{domain}:{seed}")
    rng.shuffle(cands)
    chosen: list[Item] = []
    seen_cats: set[str] = set()
    for it in cands:
        if it.spec.category not in seen_cats:
            chosen.append(it)
            seen_cats.add(it.spec.category)
        if len(chosen) == k:
            break
    return [(it.transcript, it.gold()) for it in chosen]


def run_items(
    provider: Provider,
    items: Sequence[Item],
    shots_for: Callable[[str], list[tuple[str, AfterCallRecord]]] | None = None,
    progress: bool = False,
    batch_size: int = 1,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[list[dict], list[ItemScore]]:
    """Rows and scores in the order of `items`. With batch_size > 1 (providers with generate_batch),
    items are grouped by transcript length to cut padding."""
    schema = record_json_schema()
    messages = [
        build_messages(it.domain, it.transcript, shots_for(it.domain) if shots_for else None, prompt_version)
        for it in items
    ]
    replies: list[Reply | None] = [None] * len(items)
    batched = batch_size > 1 and hasattr(provider, "generate_batch")
    order = (
        sorted(range(len(items)), key=lambda i: len(items[i].transcript))
        if batched
        else list(range(len(items)))
    )
    step = batch_size if batched else 1
    done = 0
    for start in range(0, len(order), step):
        idx = order[start : start + step]
        if batched:
            got = provider.generate_batch([messages[i] for i in idx])  # type: ignore[attr-defined]
        else:
            got = [provider.generate(messages[idx[0]], json_schema=schema)]
        for i, r in zip(idx, got, strict=True):
            replies[i] = r
        done += len(idx)
        if progress and (done % 10 < len(idx) or done == len(items)):
            print(f"[{done}/{len(items)}] last={got[-1].latency_s:.1f}s/item", file=sys.stderr, flush=True)

    rows: list[dict] = []
    scores: list[ItemScore] = []
    for it, reply in zip(items, replies, strict=True):
        assert reply is not None
        parsed = parse_reply(reply.text)
        spoken = it.split.endswith("-asr")
        score = score_item(it.item_id, it.domain, it.gold(), parsed, it.transcript, spoken)
        scores.append(score)
        rows.append(
            {
                "item_id": it.item_id,
                "domain": it.domain,
                "reply": reply.text,
                "latency_s": round(reply.latency_s, 3),
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "parse_error": parsed.error,
                "pred": parsed.record.model_dump() if parsed.record else None,
                "score": score.to_dict(),
            }
        )
    return rows, scores


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def adapter_hash(adapter: str | Path) -> str | None:
    """Hash of a PEFT adapter's weights (export.json records the same value); None without a weights file."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (Path(adapter) / name).exists():
            return _file_hash(Path(adapter) / name)
    return None


def model_fingerprint(provider: Provider) -> dict:
    """What identifies the measured weights beyond a path or tag, which retraining or re-export reuses.
    Missing pieces are None; a run never stops for them."""
    fp: dict = {}
    if isinstance(provider, HFProvider):
        if provider.adapter:
            cfg = Path(provider.adapter).parent / "train_config.json"
            fp["adapter_sha"] = adapter_hash(provider.adapter)
            fp["train_config_sha"] = _file_hash(cfg) if cfg.exists() else None
        try:
            from huggingface_hub import try_to_load_from_cache

            # The snapshot commit the cached config resolves to (snapshot_download refuses partial snapshots).
            cfg_path = try_to_load_from_cache(provider.model_id, "config.json")
            fp["base_revision"] = Path(cfg_path).parent.name if isinstance(cfg_path, str) else None
        except Exception:  # noqa: BLE001 - a local path or no hub install
            fp["base_revision"] = None
    elif isinstance(provider, OllamaProvider):
        import httpx

        names = {provider.model, f"{provider.model}:latest"}
        try:
            tags = httpx.get(f"{provider.base_url}/api/tags", timeout=10).json().get("models", [])
            fp["ollama_digest"] = next(
                (m.get("digest") for m in tags if m.get("name") in names or m.get("model") in names), None
            )
        except Exception:  # noqa: BLE001 - no model server: the run fails later with the real error
            fp["ollama_digest"] = None
    return fp


def summary_table(scores: Sequence[ItemScore], n_boot: int = 2000) -> dict:
    out: dict = {"n": len(scores), "point": summarize(scores), "ci95": {}}

    for name in HEADLINE:
        _, lo, hi = bootstrap_ci(scores, METRICS[name], n_boot=n_boot)
        out["ci95"][name] = [lo, hi]
    schema_only = [s for s in scores if s.schema_ok]
    out["schema_ok_only"] = {"n": len(schema_only), "point": summarize(schema_only)}
    by_domain = {}
    for dom in sorted({s.domain for s in scores}):
        sub = [s for s in scores if s.domain == dom]
        by_domain[dom] = {"n": len(sub), "point": summarize(sub)}
    out["by_domain"] = by_domain
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dataset JSONL")
    ap.add_argument("--backend", choices=["ollama", "hf"], default="ollama")
    ap.add_argument("--model", required=True, help="Ollama model name or HF model id/path")
    ap.add_argument("--adapter", help="PEFT adapter dir (hf backend)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--schema", action="store_true", help="constrained JSON decoding (ollama backend)")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--shot-pool", help="JSONL to draw few-shot examples from (needs summaries)")
    ap.add_argument("--batch", type=int, default=1, help="batch size (hf backend)")
    ap.add_argument("--prompt", choices=PROMPT_VERSIONS, default=PROMPT_VERSION)
    ap.add_argument(
        "--num-gpu",
        type=int,
        help="GPU layers (ollama backend); 44 keeps the 14B model from overfilling VRAM",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--label", default="")
    ap.add_argument("--official", action="store_true", help="clean tree required; writes to reports/")
    ap.add_argument("--allow-test", action="store_true", help="needed for test-* splits")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    items = load_items(data_path)
    splits = {it.split for it in items}
    if any(s.startswith("test") for s in splits) and not args.allow_test:
        ap.error(
            f"{data_path} holds test items ({sorted(splits)}); "
            "pass --allow-test once the stage's pick is fixed"
        )
    if args.limit:
        items = items[: args.limit]
    if args.official:
        if args.limit:
            ap.error("--official runs use the whole file")
        if _git("status", "--porcelain", "--untracked-files=no"):
            ap.error("--official needs a clean working tree (commit the rules first)")

    provider: Provider
    if args.backend == "ollama":
        provider = OllamaProvider(
            model=args.model, use_schema=args.schema, num_ctx=4096, num_gpu=args.num_gpu
        )
    else:
        provider = HFProvider(model_id=args.model, adapter=args.adapter, load_4bit=args.load_4bit)

    shots_for: Callable[[str], list[tuple[str, AfterCallRecord]]] | None = None
    if args.shots:
        if not args.shot_pool:
            ap.error("--shots needs --shot-pool")
        pool = load_items(args.shot_pool)
        cache: dict[str, list] = {}

        def _shots(domain: str) -> list[tuple[str, AfterCallRecord]]:
            if domain not in cache:
                cache[domain] = pick_shots(pool, domain, args.shots)
            return cache[domain]

        shots_for = _shots

    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    run_id = f"{stamp}-{safe_model}" + (f"-{args.label}" if args.label else "")
    out_dir = Path("reports" if args.official else "outputs/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "provider": provider.name,
        "model_fingerprint": model_fingerprint(provider),
        "data": str(data_path),
        "data_sha": _file_hash(data_path),
        "n_items": len(items),
        "shots": args.shots,
        "shot_pool": args.shot_pool,
        "schema_constrained": args.schema,
        "batch": args.batch,
        "prompt_version": args.prompt,
        "prompt_hash": prompt_hash(args.prompt),
        "git_commit": _git("rev-parse", "HEAD"),
        "official": args.official,
        "argv": sys.argv[1:] if argv is None else argv,
        "started": stamp,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows, scores = run_items(
        provider, items, shots_for, progress=True, batch_size=args.batch, prompt_version=args.prompt
    )
    write_jsonl(out_dir / "items.jsonl", rows)
    table = summary_table(scores)
    lat = sorted(r["latency_s"] for r in rows)
    table["latency_s"] = {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))]} if lat else {}
    (out_dir / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v, 4) for k, v in table["point"].items()}, ensure_ascii=False))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
