"""Qwen grouped-query attention replay used by routing-curvature probes."""

from __future__ import annotations

import torch

from optimizer_v2.attention import edge_factors, filter_routing_increment


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


@torch.no_grad()
def qwen_route_head_corrections(
    x: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rows: torch.Tensor,
    query_learning: torch.Tensor,
    key_learning: torch.Tensor,
    *,
    query_head: int,
    key_head: int,
    head_dim: int,
    edges_per_row: int,
    mixture: float,
    rho: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Filter one Qwen Q/K head learning increment with the routing metric.

    Qwen linear weights are stored `[out, in]`; the routing equations use
    `[in, head]`.  The conversion is explicit so a Q/K correction cannot be
    silently applied to a transposed or grouped-query-incompatible slice.
    """
    if head_dim <= 0 or query.shape != key.shape or query.shape[-1] != head_dim:
        raise ValueError("Q/K replay tensors must share the requested head dimension")
    if query_learning.ndim != 2 or key_learning.ndim != 2 or x.shape[-1] != query_learning.shape[1]:
        raise ValueError("Q/K learning increments have incompatible projection shapes")
    q_slice = slice(query_head * head_dim, (query_head + 1) * head_dim)
    k_slice = slice(key_head * head_dim, (key_head + 1) * head_dim)
    if q_slice.stop > query_learning.shape[0] or k_slice.stop > key_learning.shape[0]:
        raise ValueError("requested Q/K head is outside its projection matrix")
    factors, probabilities = edge_factors(
        x.float(),
        query.float(),
        key.float(),
        rows,
        edges_per_row=edges_per_row,
        mixture=mixture,
        generator=generator,
    )
    learning_q = query_learning[q_slice].T.float()
    learning_k = key_learning[k_slice].T.float()
    filtered_q, filtered_k, diagnostics = filter_routing_increment(factors, learning_q, learning_k, rho)
    query_correction = torch.zeros_like(query_learning)
    key_correction = torch.zeros_like(key_learning)
    query_correction[q_slice] = (filtered_q - learning_q).T.to(query_learning.dtype)
    key_correction[k_slice] = (filtered_k - learning_k).T.to(key_learning.dtype)
    diagnostics.update(
        query_head=query_head,
        key_head=key_head,
        sample_probability_min=min(probabilities),
        sample_probability_max=max(probabilities),
        sampled_edges=len(probabilities),
    )
    return query_correction, key_correction, diagnostics
