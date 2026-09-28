import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from call_summary.train import IGNORE, tail_len, tail_loss  # noqa: E402


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
    labels[1, 17:] = IGNORE  # right padding
    with torch.no_grad():
        full = model(input_ids=ids, labels=labels).loss
        k = tail_len(labels)
        assert k == 20 - 9 + 1
        tail = tail_loss(model(input_ids=ids, logits_to_keep=k).logits, labels)
    assert torch.allclose(full, tail, atol=1e-5)
