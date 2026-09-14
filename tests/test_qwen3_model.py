from __future__ import annotations

import torch
from torch import nn


class _TinyQwenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.self_attn.k_proj = nn.Linear(4, 2, bias=False)
        self.self_attn.v_proj = nn.Linear(4, 2, bias=False)
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(4, 8, bias=False)
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)
        self.mlp.down_proj = nn.Linear(8, 4, bias=False)
        self.input_layernorm = nn.Parameter(torch.ones(4))


class _TinyQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(11, 4)
        self.model.layers = nn.ModuleList(_TinyQwenLayer() for _ in range(28))
        self.lm_head = nn.Linear(4, 11, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def test_qwen_routing_uses_only_interior_linear_matrices():
    from qwen3_model import qwen_muon_parameter_names

    selected = qwen_muon_parameter_names(_TinyQwen())

    assert "model.layers.1.self_attn.q_proj.weight" in selected
    assert "model.layers.26.mlp.down_proj.weight" in selected
    assert "model.layers.0.self_attn.q_proj.weight" not in selected
    assert "model.layers.27.mlp.down_proj.weight" not in selected
    assert all("embed_tokens" not in name and "lm_head" not in name for name in selected)


def test_muown_preserves_independent_direction_and_gain_rates_for_qwen():
    from qwen3_model import build_qwen_optimizers

    optimizers = build_qwen_optimizers(
        _TinyQwen(),
        "muown",
        direction_lr=0.02,
        gain_lr=0.001,
        auxiliary_lr=0.0003,
        weight_decay=0.01,
    )

    assert optimizers["muown"].param_groups[0]["direction_lr"] == 0.02
    assert optimizers["muown"].param_groups[0]["gain_lr"] == 0.001
    assert optimizers["adamw_aux"].param_groups[0]["lr"] == 0.0003


def test_proposal_notch_uses_muon_matrix_route_and_unchanged_adamw_auxiliary_route():
    from qwen3_model import build_qwen_optimizers
    from qwen3_proposals import QwenProposalNotchOptimizer

    optimizers = build_qwen_optimizers(
        _TinyQwen(),
        "proposal_notch_v1",
        learning_rate=0.02,
        auxiliary_lr=0.0003,
        weight_decay=0.01,
    )

    assert isinstance(optimizers["proposal_notch_v1"], QwenProposalNotchOptimizer)
    assert optimizers["adamw_aux"].param_groups[0]["lr"] == 0.0003


def test_routing_resistance_uses_the_same_muon_matrix_and_adamw_auxiliary_split():
    from qwen3_model import build_qwen_optimizers
    from qwen3_proposals import QwenRoutingResistanceOptimizer

    optimizers = build_qwen_optimizers(
        _TinyQwen(),
        "routing_resistance_v1",
        learning_rate=0.02,
        auxiliary_lr=0.0003,
        weight_decay=0.01,
        routing_rho=0.5,
        routing_interval=3,
        routing_query_rows=2,
        routing_edges_per_row=3,
    )

    assert isinstance(optimizers["routing_resistance_v1"], QwenRoutingResistanceOptimizer)
    assert optimizers["routing_resistance_v1"].rho == 0.5
    assert optimizers["routing_resistance_v1"].interval == 3
    assert optimizers["routing_resistance_v1"].query_rows == 2
    assert optimizers["routing_resistance_v1"].edges_per_row == 3
    assert optimizers["adamw_aux"].param_groups[0]["lr"] == 0.0003
