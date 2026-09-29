# call-summary

고객센터 상담 대화 전사를 상담 후처리 기록(요약, 문의 유형, 처리 결과, 후속 조치, 핵심 값)의 JSON으로 바꾸는 소형 LLM을 LoRA/QLoRA로 학습하고, 로컬에서 서빙한다. 학습 전후와 큰 모델 대비 성능을 미리 정한 규칙으로 측정한다.

- 계획: [docs/plan.md](docs/plan.md)
- 측정 규칙과 결과: [docs/experiments.md](docs/experiments.md)
- 상태: 0~3단계 끝 (데이터, 기준선, 학습). 4단계(음성 인식 오류 조건), 5단계(서빙) 남음

## 결과 요약 (3단계)

상담 기록의 구조 필드(유형, 처리 결과, 핵심 값, 처리 목록, 후속 조치)가 모두 맞은 비율(exact). 괄호는 95% 부트스트랩 구간이고, 모든 모델에 같은 프롬프트를 쓰며 test 세트는 한 번만 쟀다.

| 모델 | test-a (같은 생성기) | test-b (다른 모델이 쓴 대화) | test-c (학습에 없던 업종) |
|---|---|---|---|
| qwen2.5 7B | 9.7 | 13.3 | 3.0 |
| qwen2.5 14B | 26.0 | 30.0 | 14.0 |
| qwen2.5 14B + 기록 기준 프롬프트 | 29.0 | 33.3 | 23.0 |
| Qwen3-4B 학습 전 | 17.3 | 26.7 | 5.5 |
| **Qwen3-4B QLoRA (1,500건 학습)** | **95.7** [93.3, 97.7] | **90.0** [81.7, 96.7] | **55.0** [48.0, 62.0] |

- 학습한 4B는 세 분할 모두에서 학습 전, 7B, 14B, 기록 기준을 적은 14B보다 높다 (짝지은 차이의 95% 구간 하한이 모두 0보다 크다)
- 이득은 생성기가 달라지면 조금(95.7 → 90.0), 업종이 바뀌면 크게(→ 55.0) 줄어든다. 새 업종에서 틀린 것은 대부분 그 업종에만 있는 처리 라벨이다
- 대화에 없는 값을 적는 비율(환각): 학습한 4B 0~1.5%, 14B 0~28% (새 업종에서 빈 값을 채워 넣는 경우가 많음)
- 학습 전 모델들이 틀린 것의 상당수는 프롬프트에 적지 않은 기록 관례(해결된 상담에도 남은 일을 후속 조치로 적는 것 등)였다. 관례를 프롬프트에 적어 주면 14B가 조금 나아지지만 학습과는 거리가 멀다

자세한 규칙, 모든 수치, 한계는 [docs/experiments.md](docs/experiments.md).

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

# 서비스 (Ollama 모델), 모델 없는 오프라인 모드, 부하 시험, 보안 스캔
CALL_SUMMARY_MODEL=<ollama 모델 이름> uv run uvicorn call_summary.service:create_app --factory --port 8072
uv run uvicorn call_summary.service:create_offline_app --factory --port 8072
uv run python -m call_summary.loadtest --data datasets/dev.jsonl --concurrency 1 2 4
APP_ID=<StackHawk application id> hawk scan
```

`POST /summarize` 요청 예: `{"domain": "shop", "turns": [{"speaker": "상담원", "text": "..."}, {"speaker": "고객", "text": "..."}]}`. 모델 답이 스키마를 통과하지 못하거나 업종 목록 밖의 라벨을 쓰면 한 번 다시 묻고, 그래도 안 되면 502를 돌려준다. `GET /stats`는 요청 수와 지연 분포, `GET /domains`는 업종별 라벨 목록.

음성 조건(4단계) 데이터는 `scripts/asr_condition.py`로 만들었다. MeloTTS가 transformers 4.27을 요구해 이 프로젝트 환경이 아니라 support-agent 저장소의 음성 환경에서 돌린다 (스크립트 머리말 참고).
