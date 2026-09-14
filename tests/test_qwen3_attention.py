from __future__ import annotations

from types import SimpleNamespace

import torch


class _Identity(torch.nn.Module):
    def forward(self, values):
        return values


class _TinyGroupedAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 8, bias=True)
        self.k_proj = torch.nn.Linear(4, 4, bias=True)
        self.q_norm = _Identity()
        self.k_norm = _Identity()

    def forward(self, hidden_states, position_embeddings):
        del position_embeddings
        return hidden_states


def test_qwen_head_replay_uses_the_grouped_key_head_and_rotary_inputs():
    from qwen3_attention import replay_qwen_qk_head

    torch.manual_seed(3)
    attention = _TinyGroupedAttention()
    hidden = torch.randn(1, 3, 4)
    cosine = torch.ones(1, 3, 2)
    sine = torch.zeros(1, 3, 2)

    x, query, key = replay_qwen_qk_head(attention, hidden, (cosine, sine), head=3)
    expected_q = attention.q_proj(hidden).view(1, 3, 4, 2).transpose(1, 2)[0, 3]
    expected_k = attention.k_proj(hidden).view(1, 3, 2, 2).transpose(1, 2)[0, 1]

    assert torch.equal(x, hidden[0])
    assert torch.allclose(query, expected_q)
    assert torch.allclose(key, expected_k)


def test_qwen_head_replay_rejects_cross_sequence_routing_edges():
    from qwen3_attention import replay_qwen_qk_head

    attention = _TinyGroupedAttention()
    hidden = torch.randn(2, 3, 4)
    cosine = torch.ones(2, 3, 2)
    sine = torch.zeros(2, 3, 2)

    try:
        replay_qwen_qk_head(attention, hidden, (cosine, sine), head=0)
    except ValueError as error:
        assert "one sequence" in str(error)
    else:
        raise AssertionError("expected a single-sequence routing probe error")


def test_qwen_route_filter_bypasses_exactly_at_zero_strength():
    from qwen3_attention import qwen_route_head_corrections

    torch.manual_seed(14)
    x, query, key = torch.randn(3, 4), torch.randn(3, 2), torch.randn(3, 2)
    query_learning = torch.randn(4, 4)
    key_learning = torch.randn(2, 4)
    query_correction, key_correction, diagnostics = qwen_route_head_corrections(
        x,
        query,
        key,
        torch.tensor([1, 2]),
        query_learning,
        key_learning,
        query_head=1,
        key_head=0,
        head_dim=2,
        edges_per_row=2,
        mixture=0.05,
        rho=0.0,
        generator=torch.Generator().manual_seed(15),
    )

    assert torch.equal(query_correction, torch.zeros_like(query_learning))
    assert torch.equal(key_correction, torch.zeros_like(key_learning))
    assert diagnostics["update_norm_before"] > 0


def test_attention_capture_replays_one_sequence_without_storing_the_full_batch():
    from qwen3_attention import QwenAttentionReplayCapture, replay_qwen_qk_head

    torch.manual_seed(18)
    attention = _TinyGroupedAttention()
    hidden = torch.randn(3, 4, 4)
    cosine = torch.ones(3, 4, 2)
    sine = torch.zeros(3, 4, 2)
    with QwenAttentionReplayCapture(attention, sequence_index=1) as capture:
        attention(hidden, (cosine, sine))
        replay = capture.replay(head=2)
    expected = replay_qwen_qk_head(attention, hidden[1:2], (cosine[1:2], sine[1:2]), head=2)

    assert all(torch.equal(actual, target) for actual, target in zip(replay, expected))
    assert capture.hidden_states.shape[0] == 1
