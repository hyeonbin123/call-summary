# [테스트 묶음] 학습의 메모리 절약 장치 train.tail_len·tail_loss가 손실 값을 바꾸지 않는지 검증한다. 답 구간의
# 로짓만 계산해 얻은 손실이 전체 길이 로짓으로 계산한 손실과 같아야 한다(docs/experiments.md 3단계 진행 기록:
# 전체 길이 손실과 같은 값임을 작은 모델로 테스트한다).
# [왜 중요한가] 이 장치는 학습 속도 문제를 고치려고 넣었고, 학습 설정(손실의 정의)은 그대로라고 문서에 적었다.
# 그 주장의 근거가 이 테스트다.

import pytest

# torch·transformers는 학습 환경(train 그룹)에만 있으므로 없으면 이 파일을 건너뛴다. CI(GitHub Actions)는
# train 그룹 없이 설치하므로 거기서는 건너뛰고, 학습 환경에서만 돈다.
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from call_summary.train import IGNORE, tail_len, tail_loss  # noqa: E402


# 무작위 가중치의 아주 작은 Qwen3 모델(어휘 97, 2층)로 두 행의 배치를 만든다. 행마다 답이 시작하는 위치가
# 다르고(12, 9), 둘째 행은 오른쪽 패딩(17 이후 IGNORE)까지 있다. 모델의 내장 손실(전체 길이)과, 가장 이른 답
# 위치의 바로 앞부터 끝까지 K = 20 − 9 + 1 = 12 위치의 로짓만 받아 계산한 tail_loss가 오차 1e-5 안에서 같아야
# 한다.
def test_tail_loss_matches_full_sequence_loss():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    model = transformers.Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, 97, (2, 20))
    labels = ids.clone()
    labels[0, :12] = IGNORE
    labels[1, :9] = IGNORE
    labels[1, 17:] = IGNORE  # 오른쪽 패딩
    with torch.no_grad():
        full = model(input_ids=ids, labels=labels).loss
        k = tail_len(labels)
        assert k == 20 - 9 + 1
        tail = tail_loss(model(input_ids=ids, logits_to_keep=k).logits, labels)
    assert torch.allclose(full, tail, atol=1e-5)
