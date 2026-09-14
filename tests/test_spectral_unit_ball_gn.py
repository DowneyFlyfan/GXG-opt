from __future__ import annotations

import torch

from spectral_unit_ball_gn import (
    SpectralUnitBallConfig,
    project_spectral_unit_ball,
    solve_projected_ggn,
    spectral_norms,
)
from spectral_unit_ball_gn_experiment import _batch


def test_projection_clips_each_matrix_singular_value_independently():
    vector = torch.tensor([3.0, 0.0, 0.0, 2.0, 4.0, 0.0], dtype=torch.float64)
    projected = project_spectral_unit_ball(vector, (torch.Size((2, 2)), torch.Size((1, 2))))

    norms = spectral_norms(projected, (torch.Size((2, 2)), torch.Size((1, 2))))
    assert max(norms) <= 1.0 + 1.0e-12
    torch.testing.assert_close(projected[:4], torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64))


def test_projected_solver_uses_joint_cross_block_curvature():
    # q(x) = g^T x + 1/2 x^T B x, with B_12 != 0.  A block-diagonal
    # surrogate would return a different second coordinate after one PG step.
    gradient = torch.tensor([1.0, 1.0], dtype=torch.float64)
    curvature = torch.tensor([[1.0, 0.75], [0.75, 1.0]], dtype=torch.float64)
    delta, curvature_delta, _, iterations, _ = solve_projected_ggn(
        gradient,
        lambda vector: curvature @ vector,
        (torch.Size((1, 1)), torch.Size((1, 1))),
        SpectralUnitBallConfig(inner_iterations=4, initial_beta=4.0),
    )

    assert iterations == 4
    assert torch.dot(gradient, delta) + 0.5 * torch.dot(delta, curvature_delta) < 0
    assert max(spectral_norms(delta, (torch.Size((1, 1)), torch.Size((1, 1))))) <= 1.0
    assert not torch.allclose(delta[1:2], torch.tensor([-1.0], dtype=torch.float64))


def test_language_model_output_hvp_preserves_logit_shape():
    logits = torch.randn(1, 2, 3)
    tangent = torch.randn_like(logits)
    output_hvp = _batch(torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long)).output_hvp_fn

    assert output_hvp is not None
    assert output_hvp(logits, tangent).shape == logits.shape
