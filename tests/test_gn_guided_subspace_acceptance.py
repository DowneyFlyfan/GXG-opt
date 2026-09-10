from dataclasses import replace

import pytest
import torch
from torch import nn

from gn_guided_adam.adam_state import AdamStateBank
from gn_guided_adam.acceptance import compare_candidates_statelessly
from gn_guided_adam.config import AdamWConfig, GNConfig
from gn_guided_adam.execution import candidate_loss
from gn_guided_adam.krylov import ReducedSolveResult
from gn_guided_adam.staleness import staleness_weight
from gn_guided_adam.tensor_ops import project, project_complement, subspace_overlap
from gn_guided_adam.trust_region import apply_trust_limits, decrease_damping, increase_damping
from gn_guided_adam.types import FunctionalBatch


def test_implicit_projectors_match_explicit_matrix_and_components_are_orthogonal():
    basis, _ = torch.linalg.qr(torch.tensor([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]]))
    basis = basis[:, :1]
    vector = torch.tensor([1.0, 2.0, 3.0])
    explicit = basis @ basis.T
    projected = project(basis, vector)
    complement = project_complement(basis, vector)

    assert torch.allclose(projected, explicit @ vector)
    assert torch.allclose(complement, (torch.eye(3) - explicit) @ vector)
    assert torch.dot(projected, complement).item() == pytest.approx(0.0, abs=1.0e-6)


def test_trust_limits_enforce_curvature_and_relative_parameter_caps():
    reduced = torch.diag(torch.tensor([4.0, 1.0]))
    solve = ReducedSolveResult(
        direction=torch.tensor([10.0, 0.0]),
        coordinates=torch.tensor([10.0, 0.0]),
        projected_gradient=torch.tensor([-40.0, 0.0]),
        residual_norm=0.0,
        method="test",
        predicted_reduction=1.0,
    )
    config = replace(GNConfig(), trust_radius=1.0, max_relative_block_update=0.1, alpha_max=1.0)

    trusted = apply_trust_limits(solve, reduced, torch.ones(2), config)

    assert trusted.curvature_norm * trusted.alpha <= config.trust_radius + 1.0e-6
    assert trusted.relative_update <= config.max_relative_block_update + 1.0e-6
    assert increase_damping(1.0, config) == pytest.approx(10.0)
    assert decrease_damping(1.0, config) == pytest.approx(0.5)


def test_staleness_age_drift_and_overlap_are_deterministic():
    snapshot = torch.ones(4)

    fresh = staleness_weight(snapshot, snapshot, step=5, refresh_step=5, max_age=10, max_drift=0.1)
    aged = staleness_weight(snapshot, snapshot, step=10, refresh_step=5, max_age=10, max_drift=0.1)
    drifted = staleness_weight(2 * snapshot, snapshot, step=6, refresh_step=5, max_age=10, max_drift=0.1)

    assert fresh.weight == pytest.approx(1.0)
    assert aged.weight == pytest.approx(0.5)
    assert not drifted.valid and drifted.weight == 0
    basis = torch.eye(4)[:, :2]
    assert subspace_overlap(basis, basis) == pytest.approx(1.0)
    assert subspace_overlap(torch.eye(4)[:, 2:], basis) == pytest.approx(0.0)


def test_stateless_acceptance_restores_modes_buffers_rng_and_parameters():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Dropout(0.5), nn.Linear(2, 1))
    model.train()
    inputs = torch.ones(4, 2)
    targets = torch.ones(4, 1)
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().mean())
    direction = {name: -0.01 * torch.ones_like(parameter) for name, parameter in model.named_parameters()}
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rng = torch.random.get_rng_state().clone()

    decision = compare_candidates_statelessly(
        model,
        batch,
        direction,
        direction,
        predicted_hybrid_reduction=1.0,
        rho_min=0.0,
        acceptance_margin=0.0,
        lr=0.1,
        weight_decay=0.0,
    )

    assert decision.reason in {"adam_candidate_better", "poor_reduction_ratio"}
    assert model.training and model[1].training and model[2].training
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(torch.equal(value, parameters[name]) for name, value in model.named_parameters())
    assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())


def test_candidate_loss_applies_weight_decay_once_without_mutation():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
    batch = FunctionalBatch((torch.ones(1, 1),), lambda output: output.square().sum())
    direction = {"weight": torch.zeros_like(model.weight)}

    loss = candidate_loss(model, batch, direction, lr=0.1, weight_decay=0.5)

    assert loss == pytest.approx(1.9**2)
    assert model.weight.item() == pytest.approx(2.0)


def test_momentum_subspace_bridge_changes_only_first_moment():
    model = nn.Linear(2, 1, bias=False)
    bank = AdamStateBank(model, AdamWConfig())
    gradient = {"weight": torch.tensor([[1.0, 2.0]])}
    bank.update(gradient)
    variance = bank.state["weight"]["v"].clone()
    first_before = bank.state["weight"]["m"].clone()
    basis = torch.tensor([[1.0], [0.0]])

    bank.remove_first_moment_subspace("weight", basis, fraction=0.5)

    assert not torch.equal(bank.state["weight"]["m"], first_before)
    assert torch.equal(bank.state["weight"]["v"], variance)
