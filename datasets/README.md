# datasets

모두 생성한 가상 데이터다. 회사 이름(도토리마켓, 별빛텔레콤, 새싹택배, 구름카드), 상품, 요금제, 사람 이름, 번호는 모두 지어낸 것이다.

| 파일 | 건수 | 업종 | 대화를 쓴 것 |
|---|---|---|---|
| `train.jsonl` | 1,500 | 쇼핑몰·통신사·택배사 각 500 | qwen2.5 14B instruct |
| `dev.jsonl` | 240 | 세 업종 각 80 | qwen2.5 14B instruct |
| `test-a.jsonl` | 300 | 세 업종 각 100 | qwen2.5 14B instruct |
| `test-b.jsonl` | 60 | 세 업종 각 20 | Claude (다른 모델 계열, 생성기 차이 시험) |
| `test-c.jsonl` | 200 | 카드사 (학습에 없는 업종) | qwen2.5 14B instruct |

한 줄이 한 건이다: `item_id`, `split`, `domain`, `spec`(명세: 유형, 결과, 핵심 값과 말투, 처리, 후속 조치, 대화 중 사건, 요약에 들어가야 할 사실), `turns`(상담원·고객 발화), `summary`(참고 요약), `source`, `meta`.

음성 조건 파일(4단계와 "음성 조건 v2", docs/experiments.md)은 같은 건의 `turns`를 합성 음성 → 전화 음질 → 음성 인식으로 다시 받아 적은 것이고 원래 글은 `meta.asr.written`에 있다. v1(2026-09-29): `dev-asr.jsonl` 240, `test-d.jsonl` 300(test-a의 음성 조건판), `train-half-asr.jsonl` 750. v2(2026-10-04, 운송장번호를 숫자 하나씩 읽음, 지금 보고하는 조건): `dev-asr-v2.jsonl`, `test-d-v2.jsonl`, `train-half-asr-v2.jsonl` (건수 같음).

**정답 상담 기록은 파일에 따로 없다.** `call_summary.specs.Spec.gold()`가 명세에서 만든다. 만드는 방법과 검사, 거절 비율, 사람 점검 결과는 [docs/experiments.md](../docs/experiments.md) 1단계, 분할별 수치는 `finalize_report.json`.
