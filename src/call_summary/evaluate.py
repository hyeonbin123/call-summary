"""Run a model over a dataset file and score the structured fields.

Output: <out>/<run_id>/manifest.json, items.jsonl (raw reply + parse + per-item score), summary.json.
Official runs (--official) need a clean git tree and write under reports/ so they can be committed.
"""

# [파일 역할] 이 프로젝트의 평가 실행기다. 데이터셋 파일의 모든 건을 한 모델(Ollama 또는 transformers)에 같은
# 프롬프트로 보내고, 답을 파싱해 구조 필드를 채점한 뒤 실행 기록(manifest.json, items.jsonl, summary.json)을
# 남긴다. 2~5단계와 이후 비교의 모든 수치가 이 기록에서 나왔다.
# [왜 필요한가] 측정 규칙(docs/experiments.md 공통)을 코드로 강제한다: 같은 프롬프트와 채점, 건별 결과를 남겨
# 다시 계산할 수 있게 함, test 분할은 --allow-test 없이는 돌지 않음, 공식 측정(--official)은 깨끗한 작업
# 트리에서만 돌고 reports/에 써서 커밋함. 어떤 가중치·프롬프트·서버 버전으로 잰 것인지 manifest에 남겨 나중에
# 조건을 되짚을 수 있다.
# [흐름] 인자 확인(test 보호, 공식 실행의 깨끗한 트리) → few-shot 예시 고르기 → 실행기 만들기 → manifest 먼저
# 쓰기 → run_items(생성·파싱·채점) → items.jsonl, summary.json(점 추정, 부트스트랩 95% 구간, 지연,
# run_checks).
# [연결] compare.py, judge_run.py, asr_report.py, judge_calib.py 등이 여기서 만든 실행 폴더를 읽는다.
# judge_run·judge_calib·export는 이 파일의 도우미(THINK, _git, model_fingerprint, adapter_hash 등)를 가져다
# 쓴다.

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

# 프롬프트(build_messages, prompt_hash), 실행기(HF/Ollama), 스키마 파싱(parse_reply), 채점(score_item,
# 부트스트랩)을 각 모듈에서 가져온다. 이 파일은 그것들을 순서대로 엮는 역할만 한다.
from .dataset import Item, load_items, write_jsonl
from .prompts import PROMPT_VERSION, PROMPT_VERSIONS, build_messages, prompt_hash
from .providers import HFProvider, OllamaProvider, Provider, Reply
from .schema import AfterCallRecord, parse_reply, record_json_schema
from .scoring import METRICS, ItemScore, bootstrap_ci, score_item, summarize

# 요약 표에 기본으로 넣고 부트스트랩 구간을 계산하는 주요 지표. exact(구조 필드가 모두 맞은 비율)가 주
# 지표이고, 나머지는 어디서 틀렸는지 보는 세부 지표다(docs/experiments.md 공통). compare.py도 이 목록을 표의
# 열로 쓴다.
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


# --think 값 → 요청의 think 필드. off는 생각 모드를 끄고(기본, 공통 규칙 "Qwen3 계열은 생각 모드를 끈다"),
# default는 요청에 넣지 않아 모델 기본값을 따른다.
THINK = {"off": False, "on": True, "default": None}  # --think -> the request's `think` (None: not sent)


# --option KEY=VALUE를 (키, 값)으로 바꾼다. 값은 true/false면 bool, 정수·실수로 읽히면 숫자로 바꾼다. 2026년
# 비교 대상의 태그 기본 벌점을 끄는 데(presence_penalty=0, repeat_penalty=1) 썼다(docs/experiments.md). 형식이
# 틀리면 argparse 오류로 바로 알린다.
def parse_option(text: str) -> tuple[str, object]:
    """KEY=VALUE for an extra Ollama option; the value becomes an int, float or bool when it reads as one."""
    key, sep, raw = text.partition("=")
    if not sep or not key.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    raw = raw.strip()
    value: object = raw
    if raw.lower() in ("true", "false"):
        value = raw.lower() == "true"
    else:
        for cast in (int, float):
            try:
                value = cast(raw)
                break
            except ValueError:
                pass
    return key.strip(), value


# 설정이 지켜지지 않은 답을 센다: num_predict에서 잘린 답(done_reason "length"), 생각 글이 붙어 온 답,
# 프롬프트 + 답 토큰이 컨텍스트를 다 채운 답(이 경우 프롬프트가 잘렸을 수 있다). 잘린 답도 채점한 그대로 두고
# 수만 summary.json에 적는다(docs/experiments.md 2026 비교 대상 규칙). 판정 도구들(judge_run, judge_calib)도
# 같은 함수로 센다.
def run_checks(rows: Sequence[dict], num_ctx: int | None) -> dict:
    """Counts that show a run whose settings did not hold: replies cut at num_predict, replies that came
    with thinking text, and (Ollama) replies whose prompt plus answer filled the context window."""
    full = None
    if num_ctx is not None:
        full = sum(
            1
            for r in rows
            if r.get("prompt_tokens") is not None
            and r.get("completion_tokens") is not None
            and r["prompt_tokens"] + r["completion_tokens"] >= num_ctx
        )
    return {
        "length_stops": sum(1 for r in rows if r.get("done_reason") == "length"),
        "thinking_items": sum(1 for r in rows if r.get("thinking_chars")),
        "context_full": full,
    }


# few-shot 예시를 고른다: 한 업종의 예시 후보(참고 요약이 있는 건)를 업종·seed로 정한 random.Random으로 섞고,
# 유형이 겹치지 않게 k건을 뽑아 (전사, 정답 기록)으로 돌려준다.
# 2단계 기준선의 2-shot과 2026년 비교 대상이 모두 datasets/train.jsonl에서 seed 0으로 고른 같은 예시를
# 썼다(docs/experiments.md 2단계). 예시 정답에도 summary가 들어가야 하므로 요약이 없는 건은 후보에서 뺀다.
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


# 건마다 메시지를 만들어 모델에 보내고, 답을 파싱·채점해 (기록 행 목록, 점수 목록)을 items 순서대로 돌려준다.
# 배치(transformers 전용): batch_size > 1이고 실행기에 generate_batch가 있으면 전사 길이 순으로 정렬해 묶는다.
# 비슷한 길이끼리 묶어야 왼쪽 패딩이 줄어 빠르다. 생성이 끝나면 원래 순서로 되돌려 채점한다.
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
    # 스키마는 Ollama 실행기가 use_schema=True일 때만 쓴다(--schema). 기본은 끈 채 자유 생성한 답을 파싱해
    # 채점한다.
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
        # 10건 경계를 넘을 때마다 진행 상황을 찍는다(배치 크기와 상관없이).
        if progress and (done % 10 < len(idx) or done == len(items)):
            print(f"[{done}/{len(items)}] last={got[-1].latency_s:.1f}s/item", file=sys.stderr, flush=True)

    rows: list[dict] = []
    scores: list[ItemScore] = []
    for it, reply in zip(items, replies, strict=True):
        assert reply is not None
        # 답에서 첫 JSON 객체를 찾아 스키마로 검증한다. 형식이 틀리면 그 건은 모든 필드를 오답으로 친다(공통
        # 규칙).
        # spoken: 음성 조건 분할(이름이 "-asr"로 끝남)이면 환각 검사에서 번호를 더 넓게 찾는다. 인식기가
        # "5,533, 8,050에서 3,287"처럼 번호를 쪼개 적어 원래 있던 값이 지어낸 값으로 잡히는 것을 막으려는
        # 것이다(docs/experiments.md 4단계).
        parsed = parse_reply(reply.text)
        spoken = it.split.endswith("-asr")
        score = score_item(it.item_id, it.domain, it.gold(), parsed, it.transcript, spoken)
        scores.append(score)
        # 건별 기록: 원 답 글, 지연, 토큰 수, 끝난 이유, 생각 글 길이, 파싱 오류, 파싱된 기록, 필드별 점수. 원
        # 답을 남겨 채점 규칙이 바뀌어도 모델을 다시 돌리지 않고 다시 채점할 수 있다(docs/plan.md "리포트").
        rows.append(
            {
                "item_id": it.item_id,
                "domain": it.domain,
                "reply": reply.text,
                "latency_s": round(reply.latency_s, 3),
                "prompt_tokens": reply.prompt_tokens,
                "completion_tokens": reply.completion_tokens,
                "done_reason": reply.done_reason,
                "thinking_chars": len(reply.thinking or ""),
                "parse_error": parsed.error,
                "pred": parsed.record.model_dump() if parsed.record else None,
                "score": score.to_dict(),
            }
        )
    return rows, scores


# git 명령을 실행하고 실패하면 예외를 낸다. --official의 깨끗한 트리 확인처럼 실패를 그냥 넘기면 안 되는
# 곳에서 쓴다.
def _git_strict(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


# git 명령을 실행하되 실패하면 빈 문자열을 돌려준다. 기록용(커밋 해시 등)이라 git이 없어도 실행을 멈추지
# 않는다.
def _git(*args: str) -> str:
    try:
        return _git_strict(*args)
    except (OSError, subprocess.CalledProcessError):
        return ""


# 파일의 sha256 앞 12자리. 데이터 파일(data_sha), 어댑터, 학습 설정의 지문으로 쓴다. .gitattributes가 jsonl을
# LF로 고정해 체크아웃마다 같은 해시가 나온다.
def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# PEFT 어댑터 가중치 파일의 해시. export.py가 export.json에 같은 값을 남기므로, GGUF 서빙 실행과 transformers
# 실행이 같은 어댑터에서 나왔는지 맞춰 볼 수 있다.
def adapter_hash(adapter: str | Path) -> str | None:
    """Hash of a PEFT adapter's weights (export.json records the same value); None without a weights file."""
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        if (Path(adapter) / name).exists():
            return _file_hash(Path(adapter) / name)
    return None


# 잰 가중치를 경로나 태그 이름 이상으로 식별하는 값을 모은다. 다시 학습하거나 다시 내보내면 같은 경로·태그에
# 다른 가중치가 들어갈 수 있기 때문이다(커밋 f89df9f).
# transformers: 어댑터 해시, 학습 설정 해시, 바탕 모델의 HF 스냅샷 커밋. Ollama: 모델 digest와 서버 버전(같은
# 가중치라도 실행기 버전이 다르면 조건이 다르다). 값을 못 구해도 None으로 두고 실행은 멈추지 않는다.
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

            # 캐시된 config.json 경로의 부모 폴더 이름이 스냅샷 커밋이다. snapshot_download는 일부만 받은
            # 스냅샷을 거부해 None이 나와서 이 방법을 쓴다(아래 영어 주석).
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
        try:  # the server version is part of the condition (same weights, different runtime)
            got = httpx.get(f"{provider.base_url}/api/version", timeout=10).json()
            fp["ollama_version"] = got.get("version")
        except Exception:  # noqa: BLE001
            fp["ollama_version"] = None
    return fp


# Ollama /api/ps 결과(모델이 GPU에 다 올라갔는지, 컨텍스트 길이)를 기록용으로 가져온다. 실패해도 결과는 이미
# 써 두었으므로 None으로 넘긴다. judge_run·judge_calib가 쓴다.
def loaded_models(provider) -> list[dict] | None:
    """Ollama /api/ps after a run (size_vram against size, context_length); None when there is none."""
    try:
        return provider.loaded_models()
    except Exception:  # noqa: BLE001 - only a record; the results are already written
        return None


# 실행 요약: 모든 지표의 점 추정, HEADLINE 지표의 건 단위 부트스트랩 95% 구간(2,000번), 스키마를 통과한
# 건만으로 낸 지표("형식 제외 점수", docs/plan.md 평가 방식), 업종별 지표.
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


# 명령줄 진입점. 주요 옵션: --backend(ollama/hf), --adapter·--load-4bit(학습한 학생), --schema(스키마 강제),
# --shots·--shot-pool(few-shot), --batch(hf 배치), --prompt(p1/p2),
# --num-gpu·--num-ctx·--option·--think(Ollama 요청 설정), --official, --allow-test.
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
    # 기본 컨텍스트 4,096은 서빙 조건과 같다. 2026년 비교 대상은 p2 2-shot 프롬프트가 길어 8,192로
    # 쟀다(docs/experiments.md).
    ap.add_argument("--num-ctx", type=int, default=4096, help="context window (ollama backend)")
    ap.add_argument(
        "--option",
        type=parse_option,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra Ollama option, repeatable (e.g. presence_penalty=0)",
    )
    ap.add_argument(
        "--think",
        choices=sorted(THINK),
        default="off",
        help="thinking switch sent to the model (default off; 'default' leaves it to the model)",
    )
    ap.add_argument("--limit", type=int)
    ap.add_argument("--label", default="")
    ap.add_argument("--official", action="store_true", help="clean tree required; writes to reports/")
    ap.add_argument("--allow-test", action="store_true", help="needed for test-* splits")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    items = load_items(data_path)
    # test 분할 보호: 데이터에 test 건이 있으면 --allow-test를 줘야만 돈다. "test 세트는 단계가 끝날 때 고른
    # 것만 한 번 잰다"는 규칙을 실수로 어기지 않게 하는 장치다.
    splits = {it.split for it in items}
    if any(s.startswith("test") for s in splits) and not args.allow_test:
        ap.error(
            f"{data_path} holds test items ({sorted(splits)}); "
            "pass --allow-test once the stage's pick is fixed"
        )
    if args.limit:
        items = items[: args.limit]
    # 공식 실행: 파일 전체를 써야 하고(--limit 금지), 추적 파일이 바뀌지 않은 깨끗한 트리여야 한다. 규칙을
    # 먼저 커밋하고 재는 이 프로젝트의 방식을 강제한다.
    # git 확인 자체가 실패하면 깨끗하다고 보지 않고 멈춘다(fail closed). 예전에는 git 실패가 깨끗한 트리로
    # 통과했다(커밋 6d9defc).
    if args.official:
        if args.limit:
            ap.error("--official runs use the whole file")
        try:  # fail closed: an unreadable tree is not a clean one
            dirty = _git_strict("status", "--porcelain", "--untracked-files=no")
            _git_strict("rev-parse", "HEAD")
        except (OSError, subprocess.CalledProcessError) as exc:
            ap.error(f"--official needs git to verify a clean tree: {exc}")
        if dirty:
            ap.error("--official needs a clean working tree (commit the rules first)")

    shots_for: Callable[[str], list[tuple[str, AfterCallRecord]]] | None = None
    # few-shot이면 데이터에 있는 업종마다 예시를 미리 고른다. 예시가 모자란 업종이 있으면 멈춘다: 예를 들어
    # test-c(카드)는 train에 카드 통화가 없어, 그대로 두면 조용히 0-shot으로 돌게 된다(2026년 비교 대상 규칙과
    # 함께 추가, docs/experiments.md).
    if args.shots:
        if not args.shot_pool:
            ap.error("--shots needs --shot-pool")
        pool = load_items(args.shot_pool)
        shots = {dom: pick_shots(pool, dom, args.shots) for dom in sorted({it.domain for it in items})}
        short = {dom: len(got) for dom, got in shots.items() if len(got) < args.shots}
        if short:  # e.g. test-c (card): train has no card calls, so the run would silently be 0-shot
            ap.error(f"{args.shot_pool} has fewer than {args.shots} examples for {short}")
        shots_for = shots.__getitem__

    # 백엔드에 맞는 실행기를 만든다. --option은 Ollama 요청 옵션이라 hf 백엔드에서는 오류로 막는다. hf에서는
    # --think on일 때만 채팅 템플릿의 생각 모드를 켠다.
    ollama_options = dict(args.option)
    provider: Provider
    if args.backend == "ollama":
        provider = OllamaProvider(
            model=args.model,
            use_schema=args.schema,
            num_ctx=args.num_ctx,
            num_gpu=args.num_gpu,
            think=THINK[args.think],
            extra_options=ollama_options,
        )
    else:
        if ollama_options:
            ap.error("--option is for the ollama backend")
        provider = HFProvider(
            model_id=args.model,
            adapter=args.adapter,
            load_4bit=args.load_4bit,
            enable_thinking=args.think == "on",
        )

    # 실행 id = 시각-모델 이름(경로 문자 /, :를 _로)-라벨. 공식 실행은 커밋할 reports/에, 나머지는 커밋하지
    # 않는 outputs/runs/에 쓴다.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_model = args.model.replace("/", "_").replace(":", "_")
    run_id = f"{stamp}-{safe_model}" + (f"-{args.label}" if args.label else "")
    out_dir = Path("reports" if args.official else "outputs/runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # manifest는 모델을 돌리기 전에 먼저 쓴다. 실행 조건 전체(실행기 이름, 모델 지문, 데이터 경로와 해시,
    # shot, 스키마 강제, 프롬프트 판과 지문, 컨텍스트, 생각 모드, Ollama 옵션, 코드 커밋, 명령줄 인자)를 남겨
    # 결과를 다시 만들 수 있게 한다.
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
        "num_ctx": args.num_ctx if args.backend == "ollama" else None,
        "think": args.think,
        "ollama_options": ollama_options,
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
    # 지연 p50·p95는 정렬한 목록의 위치 값으로 낸다(보간 없음). run_checks의 컨텍스트 검사는 Ollama
    # 백엔드에서만 한다(hf는 num_ctx 개념이 없어 None).
    lat = sorted(r["latency_s"] for r in rows)
    table["latency_s"] = {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))]} if lat else {}
    table["run_checks"] = run_checks(rows, args.num_ctx if args.backend == "ollama" else None)
    (out_dir / "summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: round(v, 4) for k, v in table["point"].items()}, ensure_ascii=False))
    print(f"-> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
