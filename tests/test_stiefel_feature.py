import importlib.util

import pytest
import torch


def test_source_polar_update_preserves_column_orthogonality_and_reports_budget():
    assert importlib.util.find_spec("stiefel_feature") is not None
    from stiefel_feature import feature_polar_update

    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    gradient = torch.tensor([[.2, -.1], [.1, .3], [.4, -.2]])
    covariance = torch.tensor([[2.0, .25], [.25, 1.0]])
    updated, diagnostics = feature_polar_update(weight, gradient, covariance, alpha=.1)

    torch.testing.assert_close(updated.T @ updated, torch.eye(2), atol=2e-6, rtol=2e-6)
    assert diagnostics.feature_budget > 0
    assert diagnostics.orthogonality_error < 2e-6


def test_source_polar_update_rejects_wide_weights():
    from stiefel_feature import feature_polar_update

    with pytest.raises(ValueError, match="tall or square"):
        feature_polar_update(torch.eye(2, 3), torch.ones(2, 3), torch.eye(3), alpha=.1)


def test_feature_selection_uses_only_native_tall_or_square_gpt_linears():
    from stiefel_feature import stiefel_feature_parameter_names

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(2, 6, bias=False)
            self.projection = torch.nn.Linear(2, 2, bias=False)
            self.feedforward_in = torch.nn.Linear(2, 8, bias=False)
            self.feedforward_out = torch.nn.Linear(8, 2, bias=False)

    assert stiefel_feature_parameter_names(Model()) == {
        "qkv.weight", "projection.weight", "feedforward_in.weight"
    }


def test_input_covariance_collector_aggregates_token_features_without_graph():
    from stiefel_feature import InputCovarianceCollector

    linear = torch.nn.Linear(2, 3, bias=False)
    collector = InputCovarianceCollector({"linear.weight": linear})
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    second = torch.tensor([[2.0, 0.0]], requires_grad=True)
    linear(first).sum().backward()
    linear(second).sum().backward()

    covariance = collector.consume("linear.weight")

    values = torch.cat((first.detach(), second.detach())).double()
    expected = values.T @ values / 3
    torch.testing.assert_close(covariance, expected)
    assert not covariance.requires_grad
    assert collector.consume("linear.weight") is None
    collector.close()


def test_input_covariance_collector_uses_fp64_for_its_autocast_gram_matrix():
    from stiefel_feature import InputCovarianceCollector

    linear = torch.nn.Linear(2, 3, bias=False)
    collector = InputCovarianceCollector({"linear.weight": linear})
    features = torch.tensor([[1.1, 2.2], [3.3, 4.4]], requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        linear(features).sum().backward()

    covariance = collector.consume("linear.weight")

    assert covariance.dtype == torch.float64
    torch.testing.assert_close(
        covariance,
        features.detach().double().T @ features.detach().double() / features.shape[0],
    )
    collector.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SVD numerics")
def test_column_polar_keeps_large_ill_conditioned_cuda_factor_orthogonal():
    from stiefel_feature import column_polar

    torch.manual_seed(0)
    columns = 512
    left, _ = torch.linalg.qr(torch.randn(columns, columns, device="cuda"))
    right, _ = torch.linalg.qr(torch.randn(columns, columns, device="cuda"))
    singular_values = torch.logspace(0, -7, columns, device="cuda")
    matrix = left @ torch.diag(singular_values) @ right.T

    polar = column_polar(matrix)

    assert torch.linalg.vector_norm(polar.T @ polar - torch.eye(columns, device="cuda")) < 1.0e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA SVD numerics")
def test_feature_polar_update_keeps_a_stiefel_point_when_gradient_is_zero():
    from stiefel_feature import feature_polar_update

    torch.manual_seed(1)
    rows, columns = 768, 512
    reference_weight, _ = torch.linalg.qr(
        torch.randn(rows, columns, device="cuda", dtype=torch.float64)
    )
    basis, _ = torch.linalg.qr(torch.randn(columns, columns, device="cuda", dtype=torch.float64))
    covariance = basis @ torch.diag(
        torch.logspace(0, -7, columns, device="cuda", dtype=torch.float64)
    ) @ basis.T
    weight = reference_weight.float()

    updated, diagnostics = feature_polar_update(weight, torch.zeros_like(weight), covariance, alpha=.1)

    assert torch.linalg.vector_norm(updated.double() - reference_weight) < 1.0e-4
    assert diagnostics.feature_budget < 1.0e-4
