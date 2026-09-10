"""Condition-capped spectral-norm descent for matrix parameters.

This is the finite-precision implementation of ``Low-Spectral-Variance.md``.
The documented dual solve supplies a tangent polar direction.  When the
extreme singular values are simple, the finite direct Cayley curve avoids a
second singular-value decomposition.  At a repeated extreme singular value the
source's singular-vector velocity is undefined, so the implementation uses an
exact spectrum projection for that update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LowSpectralVarianceDiagnostics:
    """Observable per-update geometry quantities."""

    top_tangent_inner_product: float
    minimum_tangent_inner_product: float
    condition_number: float
    boundary_active: bool


def _logical_orientation(matrix: Tensor) -> tuple[Tensor, bool]:
    if matrix.ndim != 2:
        raise ValueError("Low-Spectral-Variance requires matrix parameters")
    if min(matrix.shape) < 3:
        raise ValueError("Low-Spectral-Variance requires both matrix dimensions >= 3")
    return (matrix, False) if matrix.shape[0] >= matrix.shape[1] else (matrix.T, True)


@torch.no_grad()
def low_spectral_variance_parameter_names(
    model: nn.Module,
    *,
    condition_limit: float,
    candidates: set[str] | None = None,
) -> set[str]:
    """Return matrices already feasible for the requested spectral cap.

    ``candidates`` lets an experiment retain the project's Muon eligibility
    exclusions for embeddings, output heads, and small convolutional layers.
    """
    if condition_limit <= 1:
        raise ValueError("condition limit must exceed one")
    selected: set[str] = set()
    for name, parameter in model.named_parameters():
        if candidates is not None and name not in candidates:
            continue
        try:
            logical, _ = _logical_orientation(parameter)
        except ValueError:
            continue
        singular_values = torch.linalg.svdvals(logical.float())
        if (
            torch.isfinite(singular_values).all()
            and singular_values[-1] > 0
            and singular_values[0] / singular_values[-1] <= condition_limit
        ):
            selected.add(name)
    return selected


def _newton_schulz_polar_factor(matrix: Tensor, *, iterations: int = 8) -> Tensor:
    """Approximate the compact polar factor with the source's Newton-Schulz map.

    For a tall logical matrix ``Z``, the recurrence is
    ``X <- X @ (3I - X.T @ X) / 2`` after Frobenius normalization.  It is the
    finite numerical approximation explicitly allowed by the derivation; the
    subsequent singular-value projection remains the exact finite feasibility
    safeguard.
    """
    if iterations <= 0:
        raise ValueError("Newton-Schulz iterations must be positive")
    value = matrix.float()
    norm = torch.linalg.vector_norm(value)
    if norm == 0:
        return torch.zeros_like(value)
    value = value / norm
    identity = torch.eye(value.shape[1], dtype=value.dtype, device=value.device)
    for _ in range(iterations):
        value = 0.5 * value @ (3.0 * identity - value.T @ value)
    return value


def _polar_factor(matrix: Tensor) -> Tensor:
    """Return the documented finite-iteration polar approximation."""
    return _newton_schulz_polar_factor(matrix)


def _project_condition_capped_spectrum(matrix: Tensor, condition_limit: float) -> Tensor:
    """Project onto ``sigma_1=1`` and ``sigma_m>=1/condition_limit``."""
    left, singular_values, right_transpose = torch.linalg.svd(
        matrix.float(), full_matrices=False
    )
    capped = singular_values.clamp(min=1.0 / condition_limit, max=1.0)
    capped[0] = 1.0
    return (left * capped.unsqueeze(0)) @ right_transpose


def _rotation_velocity(
    weight: Tensor,
    direction: Tensor,
    left: Tensor,
    singular_values: Tensor,
    right_transpose: Tensor,
    index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Invert the source's singular-vector velocity equations at one extreme."""
    singular_value = singular_values[index]
    left_vector = left[:, index]
    right_vector = right_transpose[index]
    derivative = torch.sum(left_vector * (direction @ right_vector))
    residual = (
        singular_value * (direction.T @ left_vector)
        + weight.T @ (direction @ right_vector)
        - 2.0 * singular_value * derivative * right_vector
    )
    denominators = singular_value.square() - singular_values.square()
    coefficients = right_transpose @ residual
    other_indices = torch.ones_like(singular_values, dtype=torch.bool)
    other_indices[index] = False
    if torch.any(denominators[other_indices].abs() <= 1.0e-6):
        raise ValueError("Low-Spectral-Variance requires simple extreme singular values")
    coefficients = torch.where(
        other_indices,
        coefficients / torch.where(other_indices, denominators, torch.ones_like(denominators)),
        torch.zeros_like(coefficients),
    )
    right_velocity = right_transpose.T @ coefficients
    left_velocity = (
        direction @ right_vector + weight @ right_velocity - derivative * left_vector
    ) / singular_value
    return derivative, left_velocity, right_velocity


def _cayley_factors(vectors: Tensor, velocities: Tensor) -> tuple[Tensor, Tensor]:
    """Return the rank-at-most-four skew generator factors from the source."""
    count = vectors.shape[1]
    connection = vectors.T @ velocities
    identity = torch.eye(count, dtype=vectors.dtype, device=vectors.device)
    zeros = torch.zeros_like(identity)
    skew_core = torch.cat(
        (
            torch.cat((-connection, -identity), dim=1),
            torch.cat((identity, zeros), dim=1),
        ),
        dim=0,
    )
    return torch.cat((vectors, velocities), dim=1), skew_core


def _apply_low_rank_cayley_left(matrix: Tensor, factors: Tensor, skew_core: Tensor, step_size: float) -> Tensor:
    """Apply ``C_h(F J F.T)`` using only a small linear solve."""
    count = factors.shape[1]
    identity = torch.eye(count, dtype=matrix.dtype, device=matrix.device)
    system = identity - 0.5 * step_size * skew_core @ (factors.T @ factors)
    correction = torch.linalg.solve(system, skew_core @ (factors.T @ matrix))
    return matrix + step_size * factors @ correction


def _direct_feasible_curve(
    weight: Tensor,
    direction: Tensor,
    left: Tensor,
    singular_values: Tensor,
    right_transpose: Tensor,
    *,
    condition_limit: float,
    step_size: float,
) -> tuple[Tensor, float]:
    """Take the source's direct low-rank Cayley feasible update.

    The supplied direction is ``D=-Phi``.  The gap-based clipping condition is
    the sufficient finite-step bound in `Low-Spectral-Variance.md`; it avoids a
    second full singular-value decomposition solely to retract the candidate.
    """
    minimum_index = singular_values.numel() - 1
    top_derivative, top_left_velocity, top_right_velocity = _rotation_velocity(
        weight, direction, left, singular_values, right_transpose, 0
    )
    minimum_derivative, minimum_left_velocity, minimum_right_velocity = _rotation_velocity(
        weight, direction, left, singular_values, right_transpose, minimum_index
    )
    extreme_left = torch.stack((left[:, 0], left[:, minimum_index]), dim=1)
    extreme_right = torch.stack((right_transpose[0], right_transpose[minimum_index]), dim=1)
    left_factors, left_skew = _cayley_factors(
        extreme_left, torch.stack((top_left_velocity, minimum_left_velocity), dim=1)
    )
    right_factors, right_skew = _cayley_factors(
        extreme_right, torch.stack((top_right_velocity, minimum_right_velocity), dim=1)
    )
    left_generator_weight = left_factors @ (left_skew @ (left_factors.T @ weight))
    right_generator = right_factors @ (right_skew @ right_factors.T)
    residual = direction - left_generator_weight + weight @ right_generator
    extreme_derivatives = torch.stack((top_derivative, minimum_derivative))
    residual_orthogonal = residual - (
        extreme_left * extreme_derivatives.unsqueeze(0)
    ) @ extreme_right.T
    gap = torch.minimum(1.0 - singular_values[1], singular_values[-2] - singular_values[-1])
    if gap <= 1.0e-6:
        raise ValueError("Low-Spectral-Variance requires separated extreme singular values")
    denominator = residual_orthogonal.norm() + minimum_derivative.abs()
    admissible = 0.45 * gap / denominator.clamp_min(1.0e-8)
    lower_bound = 1.0 / condition_limit
    if minimum_derivative < 0:
        admissible = torch.minimum(
            admissible,
            0.9 * (singular_values[-1] - lower_bound) / (-minimum_derivative),
        )
    used_step = min(step_size, float(admissible))
    if used_step <= 0:
        raise ValueError("direct feasible curve has no positive admissible step")
    candidate = weight + used_step * residual
    candidate = _apply_low_rank_cayley_left(candidate, left_factors, left_skew, used_step)
    candidate = _apply_low_rank_cayley_left(
        candidate.T, right_factors, right_skew, used_step
    ).T
    return candidate, used_step


def _dual_tangent_direction(
    gradient: Tensor,
    top: Tensor,
    minimum: Tensor,
    *,
    boundary_active: bool,
    dual_steps: int,
) -> Tensor:
    """Approximately solve the dual, then enforce its primal tangent constraints."""
    multiplier_top = gradient.new_zeros(())
    multiplier_minimum = gradient.new_zeros(())
    for iteration in range(dual_steps):
        shifted = gradient + multiplier_top * top - multiplier_minimum * minimum
        polar = _polar_factor(shifted)
        rate = 0.5 / (iteration + 1) ** 0.5
        multiplier_top = multiplier_top - rate * torch.sum(top * polar)
        if boundary_active:
            multiplier_minimum = (multiplier_minimum + rate * torch.sum(minimum * polar)).clamp_min(0)
    direction = _polar_factor(gradient + multiplier_top * top - multiplier_minimum * minimum)
    direction = direction - torch.sum(top * direction) * top
    if boundary_active:
        minimum_component = torch.sum(minimum * direction)
        if minimum_component > 0:
            direction = direction - minimum_component * minimum
    return direction


def low_spectral_variance_update(
    weight: Tensor,
    gradient: Tensor,
    *,
    condition_limit: float,
    step_size: float,
    dual_steps: int,
    boundary_tolerance: float = 1.0e-4,
) -> tuple[Tensor, LowSpectralVarianceDiagnostics]:
    """Take one condition-capped spectral-norm descent update.

    The caller supplies a matrix normalized to unit spectral norm.  Wide stored
    weights are handled as their transposes so that the mathematical matrix has
    full columns as required by the source derivation.
    """
    if gradient.shape != weight.shape:
        raise ValueError("weight and gradient must have equal shapes")
    if condition_limit <= 1 or step_size <= 0 or dual_steps <= 0:
        raise ValueError("condition limit, step size, and dual steps must be positive")
    logical_weight, transposed = _logical_orientation(weight)
    logical_gradient = gradient.T if transposed else gradient
    left, singular_values, right_transpose = torch.linalg.svd(
        logical_weight.float(), full_matrices=False
    )
    if not torch.isfinite(singular_values).all() or singular_values[-1] <= 0:
        raise ValueError("Low-Spectral-Variance requires a finite full-column-rank matrix")
    if not torch.isclose(singular_values[0], torch.ones_like(singular_values[0]), rtol=2.0e-4, atol=2.0e-4):
        raise ValueError("Low-Spectral-Variance weights must be normalized to unit spectral norm")
    condition_number = singular_values[0] / singular_values[-1]
    if condition_number > condition_limit * (1.0 + boundary_tolerance):
        raise ValueError("matrix is outside the requested condition-number cap")
    top = torch.outer(left[:, 0], right_transpose[0])
    minimum = torch.outer(left[:, -1], right_transpose[-1])
    boundary_active = bool(
        singular_values[-1] <= (1.0 / condition_limit) * (1.0 + boundary_tolerance)
    )
    phi = _dual_tangent_direction(
        logical_gradient.float(),
        top,
        minimum,
        boundary_active=boundary_active,
        dual_steps=dual_steps,
    )
    direction = -phi
    try:
        candidate, _ = _direct_feasible_curve(
            logical_weight.float(),
            direction,
            left,
            singular_values,
            right_transpose,
            condition_limit=condition_limit,
            step_size=step_size,
        )
    except ValueError as error:
        if "simple extreme singular values" not in str(error) and (
            "separated extreme singular values" not in str(error)
        ):
            raise
        # The paper's rotation-velocity formula divides by spectral gaps.  A
        # repeated extreme has no unique singular-vector derivative, but the
        # tangent direction remains defined; take the finite step and retract
        # its spectrum exactly instead of aborting the whole training run.
        candidate = _project_condition_capped_spectrum(
            logical_weight.float() + step_size * direction,
            condition_limit,
        )
    candidate = candidate.to(weight.dtype)
    diagnostics = LowSpectralVarianceDiagnostics(
        top_tangent_inner_product=float(torch.sum(top * phi)),
        minimum_tangent_inner_product=float(torch.sum(minimum * phi)),
        condition_number=float(condition_number),
        boundary_active=boundary_active,
    )
    return (candidate.T if transposed else candidate), diagnostics


class LowSpectralVariance(torch.optim.Optimizer):
    """Momentum optimizer using condition-capped spectral-norm directions."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        condition_limit: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        dual_steps: int = 8,
    ) -> None:
        if lr <= 0 or condition_limit <= 1 or not 0 <= momentum < 1 or dual_steps <= 0:
            raise ValueError("Low-Spectral-Variance hyperparameters are invalid")
        super().__init__(
            params,
            dict(
                lr=lr,
                condition_limit=condition_limit,
                momentum=momentum,
                nesterov=nesterov,
                dual_steps=dual_steps,
            ),
        )
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    logical, _ = _logical_orientation(parameter)
                    scale = torch.linalg.svdvals(logical.float()).max()
                    if not torch.isfinite(scale) or scale <= 0:
                        raise ValueError("Low-Spectral-Variance requires finite nonzero matrix scales")
                    normalized = logical / scale
                    condition = torch.linalg.svdvals(normalized.float())
                    if condition[-1] <= 0 or condition[0] / condition[-1] > group["condition_limit"]:
                        raise ValueError("parameter does not satisfy the requested condition-number cap")
                    self.state[parameter]["spectral_scale"] = float(scale)

    @staticmethod
    def scaled_lr(lr: float, rows: int, columns: int) -> float:
        return lr * 0.2 * max(rows, columns) ** 0.5

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                buffer.mul_(group["momentum"]).add_(parameter.grad)
                gradient = (
                    parameter.grad.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                scale = float(state["spectral_scale"])
                updated, diagnostics = low_spectral_variance_update(
                    parameter / scale,
                    gradient / scale,
                    condition_limit=group["condition_limit"],
                    step_size=self.scaled_lr(group["lr"], *parameter.shape),
                    dual_steps=group["dual_steps"],
                )
                parameter.copy_(scale * updated)
                state["diagnostics"] = diagnostics
        return loss
