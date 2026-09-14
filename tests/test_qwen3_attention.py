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
