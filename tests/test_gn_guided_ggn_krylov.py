from dataclasses import replace

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from gn_guided_adam.blocks import BlockRegistry
from gn_guided_adam.config import GNConfig
from gn_guided_adam.ggn_operator import GGNBlockOperator, softmax_cross_entropy_hvp
from gn_guided_adam.krylov import KrylovError, build_krylov_basis, capture_metrics, reduced_gn_solve
from gn_guided_adam.oracle import GNOracle, run_oracle_probe
from gn_guided_adam.types import FunctionalBatch


def linear_operator(reduction="sum"):
    model = nn.Linear(2, 1, bias=False).double()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    loss = lambda output: 0.5 * (output - targets).square().sum() if reduction == "sum" else 0.5 * (output - targets).square().mean()
    block = BlockRegistry(model, replace(GNConfig(), min_block_numel=1)).enabled[0]
    return model, inputs, GGNBlockOperator(model, block, FunctionalBatch((inputs,), loss))


def test_matrix_free_ggn_matches_explicit_reference_and_identities():
    _, _, operator = linear_operator()
    matrix = operator.explicit_matrix_for_testing()
    first = torch.tensor([0.2, -0.1], dtype=torch.float64)
    second = torch.tensor([-0.3, 0.4], dtype=torch.float64)
    left = operator.matvec(2 * first - 0.5 * second)

    assert torch.allclose(left, 2 * operator.matvec(first) - 0.5 * operator.matvec(second), atol=1.0e-10)
    assert torch.allclose(matrix, torch.tensor([[10.0, 14.0], [14.0, 20.0]], dtype=torch.float64))
    assert torch.dot(first, operator.matvec(second)) == pytest.approx(torch.dot(second, operator.matvec(first)).item())
    assert torch.dot(first, operator.matvec(first)).item() >= -1.0e-10


def test_ggn_uses_loss_reduction_scaling_exactly():
    _, _, summed = linear_operator("sum")
    _, _, meaned = linear_operator("mean")
    vector = torch.tensor([0.5, -0.25], dtype=torch.float64)

    assert torch.allclose(meaned.matvec(vector), summed.matvec(vector) / 2, atol=1.0e-10)


def test_ggn_excludes_model_output_second_derivatives():
    class Nonlinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([[1.0]], dtype=torch.float64))

        def forward(self, value):
            return value * self.weight.square()

    model = Nonlinear()
    block = BlockRegistry(model, replace(GNConfig(), min_block_numel=1)).enabled[0]
    operator = GGNBlockOperator(
        model,
        block,
        FunctionalBatch((torch.ones(1, 1, dtype=torch.float64),), lambda output: 0.5 * output.square().sum()),
    )

    assert operator.matvec(torch.ones(1, dtype=torch.float64)).item() == pytest.approx(4.0)
    assert operator.matvec(torch.ones(1, dtype=torch.float64)).item() != pytest.approx(6.0)


def test_operator_does_not_mutate_parameters_gradients_modes_or_rng():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Dropout(0.5), nn.Linear(2, 1, bias=False)).double()
    model.train()
    model[0].weight.grad = torch.full_like(model[0].weight, 7.0)
    block = BlockRegistry(model, replace(GNConfig(), min_block_numel=1)).enabled[0]
    batch = FunctionalBatch((torch.ones(2, 2, dtype=torch.float64),), lambda output: output.square().mean())
    operator = GGNBlockOperator(model, block, batch)
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    gradient = model[0].weight.grad.detach().clone()
    rng = torch.random.get_rng_state().clone()

    operator.matvec(torch.ones(block.numel, dtype=torch.float64))

    assert model.training and model[1].training
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(model[0].weight.grad, gradient)
    assert all(torch.equal(value, parameters[name]) for name, value in model.named_parameters())


def test_softmax_cross_entropy_hvp_matches_autograd_with_mask_and_mean():
    logits = torch.tensor([[[0.2, -0.1, 0.7], [0.3, 0.4, -0.2]]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[2, 0]])
    mask = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    vector = torch.tensor([[[0.5, -0.3, 0.2], [0.1, 0.2, -0.4]]], dtype=torch.float64)

    def masked_loss(value):
        losses = functional.cross_entropy(value.reshape(-1, 3), targets.reshape(-1), reduction="none").reshape_as(mask)
        return (losses * mask).sum() / mask.sum()

    gradient = torch.autograd.grad(masked_loss(logits), logits, create_graph=True)[0]
    explicit = torch.autograd.grad((gradient * vector).sum(), logits)[0]

    assert torch.allclose(
        softmax_cross_entropy_hvp(logits, vector, mask=mask, reduction="mean"),
        explicit,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_krylov_basis_reduced_matrix_solve_and_oracle_match_dense_problem():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    gradient = torch.tensor([2.0, -1.0], dtype=torch.float64)
    krylov = build_krylov_basis(lambda value: matrix @ value, gradient, rank=2)
    solve = reduced_gn_solve(krylov.basis, krylov.reduced_matrix, gradient, damping=0.5)
    expected = torch.linalg.solve(matrix + 0.5 * torch.eye(2, dtype=torch.float64), -gradient)

    assert torch.allclose(krylov.basis.T @ krylov.basis, torch.eye(2, dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(krylov.reduced_matrix, krylov.basis.T @ matrix @ krylov.basis, atol=1.0e-10)
    assert torch.allclose(solve.direction, expected, atol=1.0e-10)
    assert solve.residual_norm < 1.0e-10

    _, _, operator = linear_operator()
    oracle_gradient = operator.gradient()
    oracle = GNOracle(operator, damping=0.5, max_iterations=20, relative_tolerance=1.0e-12).solve(oracle_gradient)
    dense = operator.explicit_matrix_for_testing()
    expected_oracle = torch.linalg.solve(dense + 0.5 * torch.eye(2, dtype=torch.float64), -oracle_gradient)
    assert torch.allclose(oracle.direction, expected_oracle, atol=1.0e-9)
    metrics = capture_metrics(solve, expected, solve.predicted_reduction)
    assert metrics["capture"] == pytest.approx(1.0)
    assert metrics["cosine"] == pytest.approx(1.0)


def test_krylov_rejects_significant_negative_curvature():
    with pytest.raises(KrylovError, match="negative eigenvalue"):
        build_krylov_basis(lambda value: -value, torch.tensor([1.0, 0.0]), rank=1)


def test_tiny_mlp_cross_entropy_ggn_is_explicitly_psd():
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    inputs = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)
    targets = torch.tensor([0, 1])
    block = BlockRegistry(model, replace(GNConfig(), min_block_numel=1)).enabled[0]
    operator = GGNBlockOperator(
        model,
        block,
        FunctionalBatch((inputs,), lambda output: functional.cross_entropy(output, targets)),
    )
    matrix = operator.explicit_matrix_for_testing()

    assert torch.allclose(matrix, matrix.T, atol=1.0e-9, rtol=1.0e-9)
    assert torch.linalg.eigvalsh(matrix).min().item() >= -1.0e-9


def test_oracle_probe_merges_selected_blocks_and_global_line_searches_without_mutation():
    model = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(1.0)
    inputs = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    targets = 2 * inputs
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum())
    block = BlockRegistry(model, replace(GNConfig(), min_block_numel=1)).enabled[0]
    operator = GGNBlockOperator(model, block, batch)
    gradient = operator.gradient()
    before = model.weight.detach().clone()

    probe = run_oracle_probe(
        model,
        {"weight": operator},
        {"weight": gradient},
        {"weight": -1.0e-3 * gradient.reshape_as(model.weight)},
        batch,
        damping=1.0e-2,
        max_iterations=20,
        relative_tolerance=1.0e-12,
        lr=1.0,
        weight_decay=0.0,
    )

    assert probe.accepted
    assert probe.actual_reduction > 0
    assert probe.oracle_loss < probe.adam_loss
    assert torch.equal(model.weight, before)
