import copy
import math

import torch

from multi_step_spectral_geometry import (
    apply_projected_curvature,
    fit_projected_curvature,
    reduced_schatten_direction,
    rollout_geometry_candidates,
    schatten_direction,
)
from multi_step_spectral_geometry import MultiStepSpectralOptimizer, SpectralPolicyConfig


def test_schatten_endpoints_energy_ratios_and_equivariance():
    matrix = torch.diag(torch.tensor([8.0, 2.0, 1.0]))
    p2 = schatten_direction(matrix, 2.0)
    pinf = schatten_direction(matrix, math.inf)
    assert math.isclose(float(torch.linalg.vector_norm(p2)), math.sqrt(3), rel_tol=1e-6)
    assert math.isclose(float(torch.linalg.vector_norm(pinf)), math.sqrt(3), rel_tol=1e-6)
    torch.testing.assert_close(p2 / p2[0, 0], matrix / matrix[0, 0])
    torch.testing.assert_close(pinf, torch.eye(3))
    p4 = schatten_direction(matrix, 4.0)
    assert math.isclose(float(p4[0, 0] / p4[1, 1]), 4.0 ** (1.0 / 3.0), rel_tol=1e-5)
    left = torch.linalg.qr(torch.randn(3, 3)).Q
    right = torch.linalg.qr(torch.randn(3, 3)).Q
    torch.testing.assert_close(
        schatten_direction(left @ matrix @ right.T, 4.0),
        left @ p4 @ right.T,
        atol=2e-5,
        rtol=2e-5,
    )


def test_projected_curvature_recovers_full_psd_operator():
    hessian = torch.tensor([[3.0, 1.0], [1.0, 2.0]])
    steps = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    changes = [hessian @ value for value in steps]
    model = fit_projected_curvature(
        steps, changes, basis_rank=2, ridge=1e-8,
        min_curvature=0.0, max_curvature=10.0, perpendicular_curvature=0.0,
    )
    vector = torch.tensor([0.3, -0.8])
    torch.testing.assert_close(apply_projected_curvature(model, vector), hessian @ vector, atol=1e-5, rtol=1e-5)
    assert bool(torch.all(model["eigenvalues"] >= 0.0))


def test_rollout_matches_explicit_quadratic_and_does_not_mutate_inputs():
    hessian = torch.eye(4)
    model = {
        "basis": torch.eye(4),
        "curvature": hessian,
        "perpendicular": 0.0,
    }
    gradient = torch.tensor([[1.0, 0.0], [0.0, 0.5]])
    momentum = gradient.clone()
    gradient_before = gradient.clone()
    momentum_before = momentum.clone()
    scores, increments, _ = rollout_geometry_candidates(
        gradient, momentum, model, [2.0, math.inf],
        learning_rates=[0.1, 0.1], horizon_weights=[1.0, 1.0],
        beta=0.9, matrix_scale=1.0, selected_p=2.0, switch_penalty=0.0,
    )
    assert scores[2.0] == sum(increments[2.0])
    torch.testing.assert_close(gradient, gradient_before)
    torch.testing.assert_close(momentum, momentum_before)


def test_fixed_mode_checkpoint_round_trip():
    config = SpectralPolicyConfig(
        lr=0.01, momentum=0.8, candidates=(4.0,), initial_p=4.0,
        transform_backend="exact_svd", fit_interval=1,
    )
    first_parameter = torch.nn.Parameter(torch.randn(3, 2))
    first = MultiStepSpectralOptimizer(
        [{"params": [first_parameter], "use_spectral": True}], config
    )
    first_parameter.grad = torch.randn_like(first_parameter)
    first.step()
    checkpoint = copy.deepcopy(first.state_dict())
    second_parameter = torch.nn.Parameter(first_parameter.detach().clone())
    second = MultiStepSpectralOptimizer(
        [{"params": [second_parameter], "use_spectral": True}], config
    )
    second.load_state_dict(checkpoint)
    gradient = torch.randn_like(first_parameter)
    first_parameter.grad = gradient.clone()
    second_parameter.grad = gradient.clone()
    first.step()
    second.step()
    torch.testing.assert_close(first_parameter, second_parameter)


def test_reduced_candidate_transforms_are_validated_against_exact_svd():
    matrix = torch.randn(16, 12, generator=torch.Generator().manual_seed(2))
    for candidate in (2.0, 4.0, 8.0, math.inf):
        exact = schatten_direction(matrix, candidate)
        reduced = reduced_schatten_direction(matrix, candidate)
        residual = torch.linalg.vector_norm(exact - reduced) / torch.linalg.vector_norm(exact)
        assert float(residual) < 0.12
