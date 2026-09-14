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
