"""Qwen grouped-query attention replay used by routing-curvature probes."""

from __future__ import annotations

import torch


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    """Match the Qwen rotary embedding half rotation."""
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@torch.no_grad()
def replay_qwen_qk_head(
    attention: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    *,
    head: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(X, Q_h, K_group(h))`` for one causal sequence.

    The routing metric constructs individual causal edges only within one
    sequence.  Qwen uses grouped-query attention, so a query head maps to a
    key/value head through integer replication rather than by sharing a
    flattened projection slice.
    """
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("a routing replay requires exactly one sequence")
    config = attention.config
    query_heads = int(config.num_attention_heads)
    key_heads = int(config.num_key_value_heads)
    if not 0 <= head < query_heads or query_heads % key_heads:
        raise ValueError("invalid grouped-query head layout")
    head_dim = int(attention.head_dim)
    shape = (*hidden_states.shape[:-1], -1, head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states).view(shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden_states).view(shape)).transpose(1, 2)
    cosine, sine = position_embeddings
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    query = query * cosine + _rotate_half(query) * sine
    key = key * cosine + _rotate_half(key) * sine
    key_head = head // (query_heads // key_heads)
    return hidden_states.detach()[0], query.detach()[0, head], key.detach()[0, key_head]
