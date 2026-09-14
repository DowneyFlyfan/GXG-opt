"""Projected full-network GGN updates over a product of spectral-norm balls.

The solver deliberately owns only the selected matrix blocks.  Its curvature
operator is therefore a restricted, *full* generalized Gauss--Newton (GGN)
operator: JVP/VJP products retain every cross-block term among those blocks.
No block-diagonal or per-layer curvature replacement is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpectralUnitBallConfig:
    """Numerical controls for the fixed-batch projected-GGN subproblem."""

    inner_iterations: int = 2
    initial_beta: float = 1.0
    beta_growth: float = 2.0
    maximum_backtracks: int = 5
    armijo_coefficient: float = 1.0e-4
    maximum_loss_backtracks: int = 5
    loss_backtrack_decay: float = 0.5
    minimum_predicted_reduction: float = 0.0


@dataclass(frozen=True)
class SpectralUnitBallStep:
    """One solved projected-GGN trial and its measured acceptance evidence."""

    accepted: bool
    beta: float
    inner_iterations: int
    model_value: float
    predicted_reduction: float
    initial_loss: float
    final_loss: float
    step_scale: float
    maximum_spectral_norm: float
    curvature_matvecs: int


def _slices(shapes: tuple[torch.Size, ...]) -> tuple[slice, ...]:
    offset = 0
    result = []
    for shape in shapes:
        count = int(torch.Size(shape).numel())
        result.append(slice(offset, offset + count))
        offset += count
    return tuple(result)


def spectral_norms(vector: Tensor, shapes: tuple[torch.Size, ...]) -> tuple[float, ...]:
    """Return every matrix block's exact largest singular value."""
    return tuple(
        float(torch.linalg.matrix_norm(vector[part].reshape(shape), ord=2).item())
        for part, shape in zip(_slices(shapes), shapes, strict=True)
    )


def project_spectral_unit_ball(vector: Tensor, shapes: tuple[torch.Size, ...]) -> Tensor:
    """Frobenius projection onto ``||X_l||_2 <= 1`` by singular-value clipping."""
    projected = torch.empty_like(vector)
    for part, shape in zip(_slices(shapes), shapes, strict=True):
        matrix = vector[part].reshape(shape)
        left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
        projected[part] = (
            (left * singular_values.clamp(max=1.0).unsqueeze(0)) @ right
        ).reshape(-1)
    return projected


def quadratic_value(gradient: Tensor, delta: Tensor, curvature_delta: Tensor) -> Tensor:
    return torch.dot(gradient, delta) + 0.5 * torch.dot(delta, curvature_delta)


def solve_projected_ggn(
    gradient: Tensor,
    curvature_matvec: Callable[[Tensor], Tensor],
    shapes: tuple[torch.Size, ...],
    config: SpectralUnitBallConfig,
) -> tuple[Tensor, Tensor, float, int, int]:
    """Run monotone projected gradient on the fixed GGN quadratic.

    ``beta`` is increased until the actual fixed-quadratic value decreases.
    This is stronger than assuming an unchecked curvature bound and leaves the
    model/curvature batch fixed throughout every inner iteration.
    """
    if config.inner_iterations <= 0 or config.initial_beta <= 0:
        raise ValueError("inner_iterations and initial_beta must be positive")
    if config.beta_growth <= 1.0:
        raise ValueError("beta_growth must exceed one")
    delta = torch.zeros_like(gradient)
    curvature_delta = torch.zeros_like(gradient)
    value = torch.zeros((), device=gradient.device, dtype=gradient.dtype)
    beta = config.initial_beta
    matvecs = 0
    completed = 0
    for _ in range(config.inner_iterations):
        residual = gradient + curvature_delta
        accepted = False
        for _ in range(config.maximum_backtracks + 1):
            candidate = project_spectral_unit_ball(delta - residual / beta, shapes)
            candidate_curvature = curvature_matvec(candidate)
            matvecs += 1
            candidate_value = quadratic_value(gradient, candidate, candidate_curvature)
            # Monotonicity is a direct check of this exact, fixed quadratic;
            # it does not assume that a power-iteration estimate is an upper
            # bound on ||B||_2.
            if torch.isfinite(candidate_value) and candidate_value <= value:
                delta, curvature_delta, value = candidate, candidate_curvature, candidate_value
                accepted = True
                completed += 1
                break
            beta *= config.beta_growth
        if not accepted:
            break
    return delta, curvature_delta, beta, completed, matvecs


@torch.no_grad()
def spectral_unit_ball_step(
    operator,
    loss_closure: Callable[[], Tensor],
    *,
    config: SpectralUnitBallConfig,
) -> SpectralUnitBallStep:
    """Solve the constrained GGN model then accept only a feasible loss decrease."""
    shapes = tuple(parameter.shape for parameter in operator.parameters)
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError("spectral-unit-ball GGN requires matrix parameter blocks")
    gradient = operator.gradient()
    initial_loss = float(loss_closure().item())
    delta, curvature_delta, beta, completed, matvecs = solve_projected_ggn(
        gradient, operator.matvec, shapes, config
    )
    model_value = float(quadratic_value(gradient, delta, curvature_delta).item())
    predicted_reduction = -model_value
    maximum_norm = max(spectral_norms(delta, shapes), default=0.0)
    if maximum_norm > 1.0 + 1.0e-5:
        raise RuntimeError("spectral projection produced an infeasible matrix update")
    originals = tuple(parameter.detach().clone() for parameter in operator.parameters)

    def assign(scale: float) -> None:
        offset = 0
        for parameter, original in zip(operator.parameters, originals, strict=True):
            size = parameter.numel()
            parameter.copy_(
                original + scale * delta[offset : offset + size].reshape_as(parameter).to(parameter)
            )
            offset += size

    if completed == 0 or predicted_reduction <= config.minimum_predicted_reduction:
        return SpectralUnitBallStep(
            False, beta, completed, model_value, predicted_reduction, initial_loss,
            initial_loss, 0.0, maximum_norm, matvecs,
        )
    scale = 1.0
    final_loss = initial_loss
    accepted = False
    directional_derivative = float(torch.dot(gradient, delta).item())
    for _ in range(config.maximum_loss_backtracks + 1):
        assign(scale)
        candidate_loss = float(loss_closure().item())
        if candidate_loss <= initial_loss + config.armijo_coefficient * scale * directional_derivative:
            final_loss = candidate_loss
            accepted = True
            break
        scale *= config.loss_backtrack_decay
    if not accepted:
        assign(0.0)
    return SpectralUnitBallStep(
        accepted, beta, completed, model_value, predicted_reduction, initial_loss,
        final_loss, scale if accepted else 0.0, maximum_norm * (scale if accepted else 0.0), matvecs,
    )
