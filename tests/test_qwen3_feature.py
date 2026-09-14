from __future__ import annotations

from types import SimpleNamespace

import torch


class _TinyQwenFeatureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(13, 4)
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)
        self.model.layers = torch.nn.ModuleList([layer])
        self.lm_head = torch.nn.Linear(4, 13, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        hidden = self.model.embed_tokens(input_ids)
        hidden = self.model.layers[0].mlp.down_proj(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_qwen_factor_probe_matches_the_dense_weight_gradient_without_populating_grad():
    from qwen3_feature import collect_qwen_dense_factors

    torch.manual_seed(12)
    model = _TinyQwenFeatureModel()
    ids = torch.tensor([[1, 2, 3, 4]])
    factors = collect_qwen_dense_factors(model, ids, ["model.layers.0.mlp.down_proj"])
    x, error = factors["model.layers.0.mlp.down_proj"]

    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(
        model(input_ids=ids).logits[:, :-1].reshape(-1, 13), ids[:, 1:].reshape(-1)
    )
    loss.backward()
    expected = model.model.layers[0].mlp.down_proj.weight.grad.T

    assert torch.allclose(x.T @ error, expected)
    model.zero_grad(set_to_none=True)
    collect_qwen_dense_factors(model, ids, ["model.layers.0.mlp.down_proj"])
    assert model.model.layers[0].mlp.down_proj.weight.grad is None


def test_qwen_cohort_update_remaps_only_the_historical_pytorch_weight_buffer():
    from qwen3_feature import qwen_cohort_momentum_step

    historical = torch.ones(2, 2)  # PyTorch [out, in]
    fresh = torch.zeros_like(historical)
    gradient = torch.zeros_like(historical)
    left = [2 * torch.eye(2)]
    right = [torch.eye(2)]

    momentum, next_historical, next_fresh = qwen_cohort_momentum_step(
        historical, fresh, gradient, beta=0.5, maps=(left, right), refresh=True
    )

    assert torch.equal(momentum, torch.ones_like(momentum))
    assert torch.equal(next_historical, momentum)
    assert torch.equal(next_fresh, torch.zeros_like(fresh))


def test_qwen_feature_prediction_diagnostic_accepts_an_exact_held_out_drift_map():
    from qwen3_feature import qwen_feature_prediction_diagnostics

    torch.manual_seed(30)
    old_fit = (torch.randn(12, 4, dtype=torch.float64), torch.randn(12, 4, dtype=torch.float64))
    old_check = (torch.randn(9, 4, dtype=torch.float64), torch.randn(9, 4, dtype=torch.float64))
    left = torch.diag(torch.tensor([1.05, 0.97, 1.02, 0.98], dtype=torch.float64))
    right = torch.diag(torch.tensor([0.96, 1.03, 0.99, 1.04], dtype=torch.float64))
    result = qwen_feature_prediction_diagnostics(
        {"module": old_fit},
        {"module": (old_fit[0] @ left, old_fit[1] @ right)},
        {"module": old_check},
        {"module": (old_check[0] @ left, old_check[1] @ right)},
        block_size=4,
    )

    assert result["module"]["accepted"] is True
    assert result["module"]["map_error"] < result["module"]["raw_error"]
