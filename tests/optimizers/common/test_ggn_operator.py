import torch
from torch import nn

from kronecker_ggn_common.curvature_operator import (
    FunctionalCurvatureBatch,
    GGNLinearOperator,
)
from kronecker_ggn_common.layer_registry import LayerRegistry


def test_matrix_free_ggn_matches_explicit_jacobian_and_double_autograd():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    inputs = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    registry = LayerRegistry(model)
    batch = FunctionalCurvatureBatch(
        (inputs,),
        lambda output: torch.nn.functional.cross_entropy(output, targets),
    )
    operator = GGNLinearOperator(model, registry, batch)
    layer_id = "0"
    vector = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.1, 0.2]], dtype=torch.float64)

    actual = operator.matvec(layer_id, vector)
    reference = operator.double_autograd_matvec(layer_id, vector)
    dense = operator.explicit_matrix_for_testing(layer_id)

    assert torch.allclose(actual, reference, atol=1.0e-10)
    assert torch.allclose(actual.reshape(-1), dense @ vector.reshape(-1), atol=1.0e-10)
    assert torch.allclose(dense, dense.T, atol=1.0e-10)
    assert torch.linalg.eigvalsh(dense).min().item() >= -1.0e-10


def test_linear_mse_ggn_has_exact_reduction_scaling():
    model = nn.Linear(2, 1, bias=False).double()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    registry = LayerRegistry(model)
    summed = GGNLinearOperator(
        model,
        registry,
        FunctionalCurvatureBatch(
            (inputs,), lambda output: 0.5 * (output - targets).square().sum()
        ),
    )
    meaned = GGNLinearOperator(
        model,
        registry,
        FunctionalCurvatureBatch(
            (inputs,), lambda output: 0.5 * (output - targets).square().mean()
        ),
    )
    vector = torch.tensor([[0.2, -0.1]], dtype=torch.float64)

    assert torch.allclose(
        summed.matvec("<root>", vector), torch.tensor([[0.6, 0.8]], dtype=torch.float64)
    )
    assert torch.allclose(
        meaned.matvec("<root>", vector), summed.matvec("<root>", vector) / 2
    )


def test_tiny_transformer_linear_registration_and_token_cross_entropy_ggn():
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(7, 4)
            self.qkv = nn.Linear(4, 12, bias=False)
            self.projection = nn.Linear(4, 4, bias=False)
            self.head = nn.Linear(4, 7, bias=False)

        def forward(self, tokens):
            hidden = self.embedding(tokens)
            query, key, value = self.qkv(hidden).chunk(3, dim=-1)
            attention = (query @ key.transpose(-1, -2) / 2.0).softmax(dim=-1)
            return self.head(hidden + self.projection(attention @ value))

    model = TinyTransformer().double()
    tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    registry = LayerRegistry(model)

    assert [layer.layer_id for layer in registry.supported] == [
        "head",
        "projection",
        "qkv",
    ]
    fallback = {
        name: reason for name, _parameter, reason in registry.fallback_parameters
    }
    assert fallback["embedding.weight"] == "embedding"

    operator = GGNLinearOperator(
        model,
        registry,
        FunctionalCurvatureBatch(
            (tokens,),
            lambda logits: torch.nn.functional.cross_entropy(
                logits.reshape(-1, 7), targets.reshape(-1)
            ),
        ),
    )
    vector = torch.randn_like(model.projection.weight)

    assert torch.allclose(
        operator.matvec("projection", vector),
        operator.double_autograd_matvec("projection", vector),
        atol=1.0e-9,
    )
