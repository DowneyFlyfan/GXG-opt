"""Core linear solver for matrix-free generalized Gauss--Newton updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConjugateGradientResult:
    direction: Tensor
    iterations: int
    residual_norm: float
    candidates: tuple[Tensor, ...]


@dataclass(frozen=True)
class FullGGNConfig:
    maximum_cg_iterations: int = 4
    relative_cg_tolerance: float = 1.0e-8
    cg_warm_start_decay: float = 0.95
    armijo_coefficient: float = 1.0e-4
    line_search_decay: float = 0.5
    maximum_line_search_steps: int = 8
    damping_increase: float = 1.5
    damping_decrease: float = 2.0 / 3.0
    minimum_damping: float = 1.0e-6
    maximum_damping: float = 1.0e4


@dataclass
class FullGGNState:
    damping: float
    previous_direction: Tensor | None = None


@dataclass(frozen=True)
class FullGGNStepResult:
    accepted: bool
    initial_loss: float
    final_loss: float
    step_scale: float
    reduction_ratio: float
    cg_iterations: int
    predicted_reduction: float


def conjugate_gradient(
    curvature_matvec: Callable[[Tensor], Tensor],
    right_hand_side: Tensor,
    *,
    damping: float,
    maximum_iterations: int,
    initial_direction: Tensor | None = None,
    relative_tolerance: float = 1.0e-8,
) -> ConjugateGradientResult:
    """Solve ``(G + damping I) direction = right_hand_side`` by CG."""
    if damping <= 0:
        raise ValueError("damping must be positive")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    direction = (
        torch.zeros_like(right_hand_side)
        if initial_direction is None
        else initial_direction.detach().to(right_hand_side).clone()
    )

    def system_matvec(vector: Tensor) -> Tensor:
        return curvature_matvec(vector) + damping * vector

    residual = right_hand_side - system_matvec(direction)
    search = residual.clone()
    squared_residual = torch.dot(residual, residual)
    initial_norm = squared_residual.sqrt().clamp_min(torch.finfo(residual.dtype).eps)
    candidates: list[Tensor] = []
    for iteration in range(1, maximum_iterations + 1):
        system_search = system_matvec(search)
        denominator = torch.dot(search, system_search)
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("Damped GGN system is not positive definite")
        step = squared_residual / denominator
        direction = direction + step * search
        # The Hessian-free backtracking policy needs every CG iterate, but
        # retaining 54M-parameter copies on the GPU can exhaust otherwise
        # usable curvature-batch memory.  Keep only the active solve on GPU.
        candidates.append(direction.detach().to("cpu", copy=True))
        residual = residual - step * system_search
        next_squared_residual = torch.dot(residual, residual)
        residual_norm = next_squared_residual.sqrt()
        if not torch.isfinite(residual_norm):
            raise RuntimeError("Conjugate gradient residual is non-finite")
        if residual_norm <= relative_tolerance * initial_norm:
            return ConjugateGradientResult(
                direction, iteration, float(residual_norm.item()), tuple(candidates)
            )
        search = residual + (next_squared_residual / squared_residual) * search
        squared_residual = next_squared_residual
    return ConjugateGradientResult(
        direction,
        maximum_iterations,
        float(squared_residual.sqrt().item()),
        tuple(candidates),
    )


def full_ggn_step(
    operator,
    loss_closure: Callable[[], Tensor],
    *,
    state: FullGGNState,
    config: FullGGNConfig,
) -> FullGGNStepResult:
    """Apply one damped full-GGN step with Armijo acceptance and LM damping."""
    if not state.damping > 0:
        raise ValueError("state damping must be positive")
    with torch.no_grad():
        initial_loss = float(loss_closure().item())
    gradient = operator.gradient()
    initial_direction = (
        None
        if state.previous_direction is None
        else state.previous_direction * config.cg_warm_start_decay
    )
    cg = conjugate_gradient(
        operator.matvec,
        -gradient,
        damping=state.damping,
        maximum_iterations=config.maximum_cg_iterations,
        initial_direction=initial_direction,
        relative_tolerance=config.relative_cg_tolerance,
    )
    originals = tuple(parameter.detach().clone() for parameter in operator.parameters)

    def assign(direction: Tensor, scale: float) -> None:
        offset = 0
        with torch.no_grad():
            for parameter, size, original in zip(
                operator.parameters, operator._sizes, originals, strict=True
            ):
                parameter.copy_(
                    original
                    + scale
                    * direction[offset : offset + size]
                    .reshape_as(parameter)
                    .to(parameter)
                )
                offset += size

    candidate_direction = cg.direction
    candidate_loss = float("inf")
    for candidate in cg.candidates:
        assign(candidate, 1.0)
        with torch.no_grad():
            loss = float(loss_closure().item())
        if loss < candidate_loss:
            candidate_loss = loss
            candidate_direction = candidate
    direction = candidate_direction.to(gradient)
    curvature_direction = operator.matvec(direction)
    gradient_dot_direction = float(torch.dot(gradient, direction).item())
    curvature_quadratic = float(torch.dot(direction, curvature_direction).item())
    accepted = False
    final_loss = initial_loss
    scale = 1.0
    for _ in range(config.maximum_line_search_steps):
        assign(direction, scale)
        with torch.no_grad():
            candidate_loss = float(loss_closure().item())
        if candidate_loss <= initial_loss + config.armijo_coefficient * scale * gradient_dot_direction:
            accepted = True
            final_loss = candidate_loss
            break
        scale *= config.line_search_decay
    if not accepted:
        assign(direction, 0.0)
        state.damping = min(
            config.maximum_damping, state.damping * config.damping_increase
        )
        return FullGGNStepResult(
            False,
            initial_loss,
            initial_loss,
            0.0,
            0.0,
            cg.iterations,
            0.0,
        )

    predicted_reduction = -(
        scale * gradient_dot_direction
        + 0.5 * scale * scale * curvature_quadratic
    )
    actual_reduction = initial_loss - final_loss
    reduction_ratio = actual_reduction / max(predicted_reduction, torch.finfo(direction.dtype).eps)
    if reduction_ratio < 0.25:
        state.damping = min(
            config.maximum_damping, state.damping * config.damping_increase
        )
    elif reduction_ratio > 0.75:
        state.damping = max(
            config.minimum_damping, state.damping * config.damping_decrease
        )
    state.previous_direction = cg.direction.detach().clone()
    return FullGGNStepResult(
        True,
        initial_loss,
        final_loss,
        scale,
        reduction_ratio,
        cg.iterations,
        predicted_reduction,
    )
