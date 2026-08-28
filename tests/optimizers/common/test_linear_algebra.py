import pytest
import torch
from torch import nn

from kronecker_ggn_common.kronecker_factors import (
    KroneckerFactorEstimator,
    update_factor_ema,
)
from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator
from kronecker_ggn_common.layer_registry import LayerRegistry
from kronecker_ggn_common.loss_hessian import softmax_cross_entropy_hessian_matvec
from kronecker_ggn_common.vector_space import (
    frobenius_inner,
    normalize_frobenius,
    project_frobenius,
    reconstruct_frobenius,
)


def dense_action(operator, function, dtype=torch.float64):
    rows, columns = operator.matrix_shape
    identity = torch.eye(rows * columns, dtype=dtype)
    return torch.stack(
        [
            function(identity[:, index].reshape(rows, columns)).reshape(-1)
            for index in range(rows * columns)
        ],
        dim=1,
    )


def test_matrix_vector_convention_and_frobenius_projection():
    activation = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    output = torch.tensor([[1.5, -0.2], [-0.2, 0.7]], dtype=torch.float64)
    value = torch.tensor([[0.2, -0.4], [0.5, 0.1]], dtype=torch.float64)
    operator = KroneckerSpectralOperator(activation, output, 0.2)

    expected = (
        torch.kron(output, activation) + 0.2 * torch.eye(4, dtype=torch.float64)
    ) @ value.reshape(-1)

    assert torch.allclose(operator.matvec(value).reshape(-1), expected, atol=1.0e-12)
    basis = torch.stack(
        (
            normalize_frobenius(value),
            normalize_frobenius(
                torch.tensor([[0.4, 0.2], [-0.2, 0.4]], dtype=torch.float64)
            ),
        )
    )
    coefficients = project_frobenius(basis, value)
    assert torch.allclose(
        reconstruct_frobenius(basis, coefficients),
        basis.reshape(2, -1).T.mv(coefficients).reshape_as(value),
    )
    assert frobenius_inner(value, value).item() == pytest.approx(
        value.square().sum().item()
    )


def test_joint_spectral_actions_match_dense_eigendecomposition():
    activation = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    output = torch.tensor([[1.5, -0.2], [-0.2, 0.7]], dtype=torch.float64)
    operator = KroneckerSpectralOperator(
        activation, output, 0.2, eigenvalue_floor=1.0e-12
    )
    dense = dense_action(operator, operator.matvec)
    eigenvalues, eigenvectors = torch.linalg.eigh(dense)
    inverse = eigenvectors @ torch.diag(eigenvalues.reciprocal()) @ eigenvectors.T
    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    square_root = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T

    assert torch.allclose(
        dense_action(operator, operator.apply_inverse), inverse, atol=1.0e-11
    )
    assert torch.allclose(
        dense_action(operator, operator.apply_inverse_sqrt), inverse_sqrt, atol=1.0e-11
    )
    assert torch.allclose(
        dense_action(operator, operator.apply_sqrt), square_root, atol=1.0e-11
    )
    assert torch.allclose(square_root @ square_root, dense, atol=1.0e-11)


def test_cross_entropy_output_hessian_matches_autograd():
    logits = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.1, 0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    vector = torch.tensor([[0.1, 0.5, -0.2], [0.4, -0.3, 0.2]], dtype=torch.float64)
    expected = torch.autograd.functional.hvp(
        lambda value: torch.nn.functional.cross_entropy(value, torch.tensor([2, 1])),
        logits,
        vector,
    )[1]

    actual = softmax_cross_entropy_hessian_matvec(logits, vector, reduction="mean")

    assert torch.allclose(actual, expected, atol=1.0e-12)


def test_factor_ema_symmetrizes_and_serializes_values():
    estimate = torch.tensor([[2.0, 0.4], [0.2, 1.0]], dtype=torch.float64)
    first = update_factor_ema(None, estimate, 0.9)
    second = update_factor_ema(first, 2 * estimate, 0.5)

    assert torch.equal(first, first.T)
    assert torch.allclose(second, 1.5 * first)


def test_hooked_empirical_fisher_capture_does_not_modify_parameter_gradients():
    model = nn.Linear(2, 2, bias=False).double()
    inputs = torch.tensor([[1.0, 2.0], [0.5, -0.2]], dtype=torch.float64)
    targets = torch.tensor([[0.3, -0.4], [0.1, 0.2]], dtype=torch.float64)
    estimator = KroneckerFactorEstimator(torch.float64)

    factors = estimator.capture_from_loss(
        LayerRegistry(model),
        lambda: 0.5 * (model(inputs) - targets).square().sum(),
        curvature_mode="empirical_fisher",
    )

    assert model.weight.grad is None
    assert factors["<root>"].activation.shape == (2, 2)
    assert factors["<root>"].output.shape == (2, 2)
    with pytest.raises(ValueError, match="exact_ggn"):
        estimator.capture_from_loss(
            LayerRegistry(model),
            lambda: model(inputs).square().sum(),
            curvature_mode="exact_ggn",
        )
