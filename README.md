# call-summary

고객센터 상담 대화 전사를 상담 후처리 기록(요약, 문의 유형, 처리 결과, 후속 조치, 핵심 값)의 JSON으로 바꾸는 소형 LLM을 LoRA/QLoRA로 학습하고, 로컬에서 서빙한다. 학습 전후와 큰 모델 대비 성능을 미리 정한 규칙으로 측정한다.

- 계획: [docs/plan.md](docs/plan.md)
- 측정 규칙과 결과: [docs/experiments.md](docs/experiments.md)
- 상태: 0~5단계 끝 (데이터, 기준선, 학습, 음성 인식 오류 조건, 서빙). 2026-10-04 음성 조건을 v2로 고쳐 다시 잼, 서비스에 확인이 필요한 번호 표시를 더함

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

## 결과 요약 (4단계: 음성 인식을 거친 전사)

test-a의 발화를 합성 음성(MeloTTS) → 전화 음질(8 kHz μ-law) → Whisper로 다시 받아 적은 test-d에서 잰 exact. 보고하는 음성 조건은 **v2**(2026-10-04, 운송장번호를 숫자 하나씩 읽음)이고, 처음 잰 v1(2026-09-29) 값을 함께 둔다.

| 모델 | test-a (글 전사) | test-d v1 (2026-09-29) | **test-d v2** (2026-10-04) |
|---|---|---|---|
| qwen2.5 14B | 26.0 | 14.7 | 재지 않음 |
| Qwen3-4B, 글 전사로만 학습 | 95.7 | 51.7 | 재지 않음 |
| **Qwen3-4B, 학습 데이터 절반을 음성 인식 전사로 바꿔 학습** | **95.7** | 80.7 [76.0, 85.3] | **79.3** [74.7, 84.0] |

- v1은 학습한 모델 그대로(transformers, 4bit + LoRA), v2는 서빙하는 GGUF q8_0(Ollama 0.35.1)으로 쟀다. 조건만 바꾼 효과는 같은 서빙 모델의 dev 음성 조건에서 −1.7%p [−5.8, +2.1]이고, 차이는 모두 운송장번호가 든 통화에서 났다
- 음성 인식을 거치면 정답 값이 전사에 온전히 남는 통화가 v2에서 82%(v1 84%)뿐이다 (상품·기기 이름과 주문번호, 운송장번호가 잘 깨진다). 글 전사로만 학습한 모델은 들린 대로 옮겨 적어 44%p 떨어진다 (v1)
- 같은 계산량에서 학습 데이터 절반을 음성 인식 전사로 바꾸면 음성 조건이 29%p 오르고 글 전사 성능은 그대로다 (v1)
- 이 모델은 학습 때 본 이름으로 인식 오류를 바로잡지만("은하택구" → "은하 탭 9"), 빠진 번호의 숫자를 짐작해 채우기도 한다. 번호는 서비스에서 따로 검증해야 한다. 그래서 서비스는 형식이 맞지 않거나 전사에 없는 번호를 "확인이 필요한 번호"로 표시한다 (아래 5단계)
- v1 조건에는 측정 인공물이 있었다: 합성 전에 발음형으로 풀어 쓰는 단계가 운송장번호 `6291-7877-5168`을 범위로 보고 "육천이백구십일 칠천팔백칠십칠에서 오천백육십팔"로 읽게 했다 (test-d 300건 중 87건). v2는 운송장번호만 숫자 하나씩 읽게 고친 것이다. v2에서 가짜 "에서"는 사라졌지만 열두 자리를 하나씩 읽으면 숫자가 빠지거나, 더해지거나, 바뀌거나, 한글 음절로 적히는 일("612호 7692 4794")이 생겨 운송장번호가 전사에 남는 비율은 오히려 낮아졌다 (test-d 90.8 → 85.1%). 규칙과 결과는 docs/experiments.md "음성 조건 v2"

## 결과 요약 (5단계: 서빙)

음성 조건까지 학습한 4B 모델을 GGUF로 바꿔 Ollama에 올리고 FastAPI 서비스로 감쌌다 (dev 240건).

| 형태 | exact | 파일 | 요청당 지연 p50 |
|---|---|---|---|
| 학습한 모델 그대로 (transformers, 4bit + LoRA) | 94.6 | - | - |
| **GGUF q8_0, 서비스 경유** | **94.6** | 4.3 GB | 2.7초 |
| GGUF q4_K_M | 66.2 | 2.5 GB | 2.0초 |

- 서비스를 거쳐도 품질이 그대로다 (dev 0.0%p, 음성 조건(v1) dev −0.4%p). 1,200건 동안 재시도·거절 0
- 처음 서빙에서는 두 가지로 크게 떨어졌다: (1) JSON 스키마 강제 디코딩이 키 순서를 바꿔 학습한 모델이 필드를 빠뜨림 → 강제를 끄고 파싱·검증·재시도로 대신, (2) QLoRA 어댑터를 원래 fp16 바탕에 합쳐 약 7%p 손실 → 학습 때의 4bit 바탕을 fp16으로 풀어낸 뒤 합침. 원인을 dev에서 진단한 과정은 docs/experiments.md 5단계
- 확인이 필요한 번호 표시 (2026-10-04): 기록은 그대로 두고, 번호(주문번호, 운송장번호 등) 중 형식이 맞지 않거나 전사에서 찾을 수 없는 것을 응답에 따로 적는다. 기록된 답에 대 보면 dev 음성 조건(v2)에서 맞은 번호 98개는 하나도 표시하지 않고 틀린 번호 33개 중 22개를 표시해, 표시 없이 나가는 틀린 번호가 예측한 번호의 25.2%에서 8.4%로 준다 (test-d v2에 한 번: 맞은 125개 중 0, 틀린 39개 중 22개, 23.8 → 10.4%). 정확도 지표는 그대로다. 형식이 맞고 전사에도 있는 한 글자 바뀜(대부분 1과 2)은 잡지 못한다
- 한 대의 GPU에서 Ollama가 요청을 하나씩 처리해 처리량은 약 0.4요청/초 (동시 요청을 늘리면 지연만 늘어남)
- 메모리와 속도 (2026-10-03, dev를 한 건씩): Ollama가 보고하는 VRAM은 q8_0 4,731 MiB, f16 8,328 MiB, q4_K_M 3,031 MiB (4,096 컨텍스트의 KV 캐시 576 MiB 포함), 생성 속도는 58, 37, 81토큰/초. 지연은 표의 값(2026-09-30)보다 약 1.45배 길었다 (답은 글자까지 같음). 이 측정 때는 배경화면 프로그램이 GPU를 함께 쓰고 있었고 Ollama도 0.34.2에서 0.35.0으로 바뀌어 있었는데, 둘 중 무엇 때문에 느려졌는지는 가르지 못했다
- 보안 스캔(HawkScan, 모델 없는 오프라인 모드): High·Medium·Low 모두 0. 2026-10-03 18:38 스캔까지는 `/summarize` 요청이 모두 입력 검사(422)에서 걸려 검사를 통과한 뒤의 처리 경로를 시험하지 못했다. OpenAPI 설명이 받는 발화자 값을 정규식으로만 적어 스캔 엔진이 맞는 값을 만들지 못했고, enum으로 고친 뒤에는 엔진이 한글 값을 "???"로 바꿔 보냈다. 그래서 한글을 JSON `\u` 이스케이프로 적은 요청(`hawk/summarize.har`)을 씨앗으로 넣었다. 그 뒤 스캔(2026-10-03 18:47)에서는 `/summarize` 1,793건 중 658건이 처리기에 닿았고 결과는 그대로 0건이다. 번호 표시를 넣은 뒤 다시 스캔(2026-10-04)해도 0건이다 (처리기에 닿은 656건) (docs/experiments.md "보안 스캔 기록")

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

# LoRA 합치기 → GGUF → llama-quantize → Ollama 등록 (work/llama.cpp, work/llama-bin 필요. QLoRA 어댑터는 4bit 바탕을 풀어 합침)
uv run --no-sync python -m call_summary.export --base Qwen/Qwen3-4B --adapter outputs/train/qwen3-4b-qlora-r16-mixed/final --name call-summary-4b-dq --out outputs/export/qwen3-4b-mixed-dq --num-ctx 4096

# 서비스 (Ollama 모델), 모델 없는 오프라인 모드, 부하 시험, 메모리·속도 측정(한 건씩), 보안 스캔
CALL_SUMMARY_MODEL=call-summary-4b-dq:q8_0 uv run uvicorn call_summary.service:create_app --factory --port 8072
uv run uvicorn call_summary.service:create_offline_app --factory --port 8072
uv run python -m call_summary.loadtest --data datasets/dev.jsonl --concurrency 1 2 4
uv run --no-sync python -m call_summary.bench --data datasets/dev.jsonl --model call-summary-4b-dq:q8_0 --out reports/s5c-bench-q8_0.json
APP_ID=<StackHawk application id> hawk scan
```

`POST /summarize` 요청 예: `{"domain": "shop", "turns": [{"speaker": "상담원", "text": "..."}, {"speaker": "고객", "text": "..."}]}`. 전사가 4,000자를 넘으면 모델 컨텍스트(4,096 토큰)에 답까지 들어가지 않으므로 413을 돌려준다 (데이터셋 전사는 가장 긴 것이 1,485자). 모델 답이 스키마를 통과하지 못하거나 업종 목록 밖의 라벨을 쓰면 한 번 다시 묻고, 그래도 안 되면 502를 돌려준다. 응답의 `needs_confirmation`은 기록의 번호(주문번호, 운송장번호, 접수번호, 카드 끝자리) 가운데 배포 형식(정규화한 값이 `D\d{7}`, 숫자 12자리, `R\d{6}`, 숫자 4자리)에 맞지 않거나 전사에서 찾을 수 없는 것을 `{type, value, reasons}`로 적는다 (`reasons`: `format`, `not_in_transcript`). 기록은 바꾸지 않으므로, 표시된 번호는 상담원이 고객에게 다시 확인하라는 뜻이다. 형식이 맞고 전사에도 있는 틀린 번호(인식기가 잘못 들은 숫자를 그대로 옮긴 것)는 표시되지 않는다. 첫 답은 탐욕 디코딩이라 같은 요청을 되풀이하면 같은 답이 나오므로, 다시 물을 때는 temperature 0.3과 다른 seed로 샘플링한다. `GET /stats`는 요청 수와 지연 분포, `GET /domains`는 업종별 라벨 목록. 처리기에 닿은 `/summarize` 요청마다 로거 `call_summary.service`가 JSON 한 줄을 남긴다 (요청 id, 업종, 발화 수·글자 수, 결과(`ok`, `retried_ok`, `rejected`, `model_unavailable`, `unknown_domain`, `too_long`), HTTP 상태, 시도 횟수, 지연, 시도를 합친 프롬프트·생성 토큰 수, 확인이 필요한 번호 수). 요청 본문이 형식 검사(빠진 필드, 허용되지 않은 발화자, 빈 `turns` 등)에서 걸려 FastAPI가 422를 돌려준 요청은 처리기에 닿지 않으므로 줄이 남지 않는다 (목록에 없는 업종의 422는 처리기가 돌려주므로 남는다). 전사와 기록 내용, 표시한 번호의 값은 개인정보가 들어 있어 남기지 않는다. 모델 없는 오프라인 모드의 고정 답에는 업종마다 정해 둔 번호 하나가 들어 있어 번호 검사가 늘 돈다. 기록된 평가 실행에 같은 검사를 대 보는 도구: `uv run --no-sync python -m call_summary.verify reports/<run_id> --show`.

음성 조건(4단계) 데이터는 `scripts/asr_condition.py`로 만들었다. MeloTTS가 transformers 4.27을 요구해 이 프로젝트 환경이 아니라 support-agent 저장소의 음성 환경에서 돌린다 (스크립트 머리말 참고). 2026-10-03부터 기본은 조건 v2(`--verbalizer v2`, 운송장번호를 숫자 하나씩 읽음, 결과는 `data/asr-v2/`)다. `--dry-run`은 모델 없이 새로 합성할 발화 수를 보여 주고, 음성 환경이 캐시와 달라 v1과 같게 읽는 발화까지 다시 만들어야 하면 스크립트가 멈춘다. 모델 없이 도는 분석: `uv run --no-sync python -m call_summary.asr_report survival datasets/dev-asr.jsonl` (유형별 정답 값 보존율), `... asr_report unfound reports/<run_id>` (전사에 없는 예측 값을 복원과 지어낸 값으로 나눔), `... asr_report tracking-forms datasets/test-d-v2.jsonl --show-lost` (인식기가 운송장번호를 적은 꼴과 찾지 못한 번호마다 들린 꼴).
