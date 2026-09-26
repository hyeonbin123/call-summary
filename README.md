# call-summary

고객센터 상담 대화 전사를 상담 후처리 기록(요약, 문의 유형, 처리 결과, 후속 조치, 핵심 값)의 JSON으로 바꾸는 소형 LLM을 LoRA/QLoRA로 학습하고, 로컬에서 서빙한다. 학습 전후와 큰 모델 대비 성능을 미리 정한 규칙으로 측정한다.

- 계획: [docs/plan.md](docs/plan.md)
- 측정 규칙과 결과: [docs/experiments.md](docs/experiments.md)
- 상태: 0단계(뼈대). 결과는 측정이 끝나는 대로 여기에 적는다

데이터는 모두 생성한 가상 데이터다. 업종 4개(쇼핑몰, 통신사, 택배사, 카드사)와 회사 이름, 상품, 요금제는 모두 지어낸 것이다.

## 데이터를 만드는 방식

정답을 먼저 만들고 대화를 나중에 만든다.

1. **명세** (`specs.py`): seed를 고정한 난수로 업종, 문의 유형, 핵심 값과 그 말투(`32,000원` / `3만 2천 원`), 상담원이 한 처리, 결과, 대화 중 일어나는 일(번호를 잘못 말했다 고침, 불만, 대기 등)을 뽑는다. 구조 필드의 정답은 명세에서 바로 나온다
2. **대화** (`gen.py`): 로컬 교사 모델이 명세를 대화로 쓴다
3. **자동 검사**: 명세의 값이 글자 그대로 모두 나오는지, 명세에 없는 금액·날짜·시각·번호가 나오지 않는지 코드로 확인하고, 통과하지 못한 대화는 버린다

## 설치와 테스트

```bash
uv sync                    # 기본 + 개발 도구 (테스트는 GPU·모델 없이 돈다)
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

학습·transformers 추론에는 `train` 그룹이 필요하다 (torch cu128). 이 그룹을 설치한 환경에서는 `uv sync`를 그룹 없이 다시 부르면 지워지므로 `uv sync --group train`을 쓰거나 `uv run --no-sync`로 실행한다.

```bash
uv sync --group train
```

## 실행

```bash
# 명세 만들기 (분할: train, dev, test-a, test-b, test-c)
uv run python -m call_summary.build_specs --split dev --per-domain 96 --out data/specs/dev.jsonl

# 교사 모델로 대화·참고 요약 만들기 (Ollama, 이어서 하기 가능)
uv run python -m call_summary.gen --specs data/specs/dev.jsonl --out data/raw/dev.jsonl --model qwen2.5:14b-instruct

# 분할 확정: 업종마다 통과한 앞의 N건, train과 번호가 겹치는 test 건 제외, 거절 비율 보고
uv run python -m call_summary.finalize

# 평가 (Ollama 또는 transformers). 결과는 outputs/runs/<run_id>/, --official이면 reports/
uv run python -m call_summary.evaluate --data data/dev.jsonl --backend ollama --model qwen2.5:7b-instruct
uv run --no-sync python -m call_summary.evaluate --data data/dev.jsonl --backend hf --model Qwen/Qwen3-1.7B --batch 4
uv run --no-sync python -m call_summary.evaluate --data data/dev.jsonl --backend hf --model Qwen/Qwen3-1.7B --adapter outputs/train/<run>/final

# 요약 판정 (판정 모델), 사람 채점 양식, 실행 비교 표
uv run python -m call_summary.judge_run --run outputs/runs/<run_id>
uv run python -c "from call_summary.judge_run import hand_template_main as m; m()" --run outputs/runs/<run_id> --out work/hand.jsonl
uv run python -m call_summary.compare outputs/runs/<a> outputs/runs/<b>
uv run python -m call_summary.compare --paired outputs/runs/<a> outputs/runs/<b>

# 학습 (LoRA, 4B는 --qlora)
uv run --no-sync python -m call_summary.train --model Qwen/Qwen3-1.7B --out outputs/train/qwen3-1.7b

# LoRA 합치기 → GGUF → Ollama 등록 (work/llama.cpp 필요)
uv run --no-sync python -m call_summary.export --base Qwen/Qwen3-1.7B --adapter outputs/train/qwen3-1.7b/final --name call-summary-qwen3-1.7b --out outputs/export/qwen3-1.7b

# 서비스
CALL_SUMMARY_MODEL=call-summary-qwen3-1.7b:q8_0 uv run uvicorn call_summary.service:create_app --factory --port 8072
```
