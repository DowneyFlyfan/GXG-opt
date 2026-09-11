"""Certified finite steps that retain effective rank at least one half."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CertifiedEffectiveRankStep:
    """A spectrally bounded update with its retained effective-rank evidence."""

    weight: Tensor
    direction: Tensor
    scale: float
    accepted: bool
    effective_rank: float


class EffectiveRankHalf(torch.optim.Optimizer):
    """Momentum partial-polar updates constrained to effective rank one half."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
    ) -> None:
        if lr <= 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("effective-rank-half hyperparameters are invalid")
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, "momentum": momentum})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                matrix = parameter.reshape(parameter.shape[0], -1)
                gradient = parameter.grad.reshape_as(matrix)
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(matrix))
                buffer.mul_(group["momentum"]).add_(gradient)
                direction_gradient = gradient.add(buffer, alpha=group["momentum"])
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                maximum_step = float(
                    torch.linalg.matrix_norm(matrix, ord="fro") / math.sqrt(min(matrix.shape))
                )
                if maximum_step == 0.0 or effective_rank(matrix) < 0.5 - 1.0e-6:
                    state["skipped_steps"] = int(state.get("skipped_steps", 0)) + 1
                    continue
                scaled_step = group["lr"] * 0.2 * math.sqrt(max(matrix.shape))
                update = certified_effective_rank_step(
                    matrix,
                    direction_gradient,
                    step_size=min(scaled_step, 0.99 * maximum_step),
                )
                parameter.copy_(update.weight.reshape_as(parameter))
                state["last_scale"] = update.scale
                state["accepted_steps"] = int(state.get("accepted_steps", 0)) + int(update.accepted)
        return loss


def effective_rank(weight: Tensor) -> float:
    """Return the Frobenius-norm effective-rank ratio for a nonzero matrix."""
    if weight.ndim != 2:
        raise ValueError("effective-rank certification requires a matrix")
    rank_bound = min(weight.shape)
    frobenius_squared = torch.sum(weight.square())
    if float(frobenius_squared.detach()) == 0.0:
        return 0.0
    gram_squared = torch.sum((weight.transpose(-2, -1) @ weight).square())
    return float((frobenius_squared.square() / (rank_bound * gram_squared)).detach())


def _constraint(weight: Tensor) -> Tensor:
    """Return the equivalent half-effective-rank constraint."""
    rank_bound = min(weight.shape)
    frobenius_squared = torch.sum(weight.square())
    gram_norm = torch.linalg.matrix_norm(weight.transpose(-2, -1) @ weight, ord="fro")
    return math.sqrt(rank_bound / 2.0) * gram_norm - frobenius_squared


def _is_effective_rank_half_feasible(weight: Tensor, tolerance: float) -> bool:
    """Check the scale-invariant constraint within the matrix dtype's precision."""
    numerical_tolerance = tolerance
    if weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
        numerical_tolerance = max(numerical_tolerance, 1.0e-6)
    return effective_rank(weight) >= 0.5 - numerical_tolerance


def _partial_polar(gradient: Tensor) -> Tensor:
    """Return a rank-aware partial polar factor with spectral norm at most one."""
    left, singular_values, right_transpose = torch.linalg.svd(gradient, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        return torch.zeros_like(gradient)
    threshold = torch.finfo(singular_values.dtype).eps * max(gradient.shape) * singular_values[0]
    rank = int((singular_values > threshold).sum().item())
    return left[:, :rank] @ right_transpose[:rank, :]


def certified_effective_rank_step(
    weight: Tensor,
    gradient: Tensor,
    *,
    step_size: float,
    tolerance: float = 1.0e-10,
    bisection_steps: int = 48,
) -> CertifiedEffectiveRankStep:
    """Take the largest feasible partial-polar step along the supplied gradient.

    The scalar search stays inside the spectral-norm ball because it scales a
    partial polar factor.  It starts from zero, which is feasible whenever
    the input already has effective rank at least one half, and accepts only
    candidates satisfying the finite-step constraint within the matrix dtype's
    numerical precision.
    """
    if weight.shape != gradient.shape or weight.ndim != 2:
        raise ValueError("weight and gradient must be equally shaped matrices")
    if step_size <= 0 or bisection_steps <= 0:
        raise ValueError("step_size and bisection_steps must be positive")
    rank_bound = min(weight.shape)
    maximum_step = float(torch.linalg.matrix_norm(weight, ord="fro") / math.sqrt(rank_bound))
    if not step_size < maximum_step:
        raise ValueError("step_size must be smaller than ||W||_F / sqrt(r)")
    if not _is_effective_rank_half_feasible(weight, tolerance):
        raise ValueError("input matrix is not effective-rank-half feasible")

    direction = _partial_polar(gradient)
    if not torch.any(direction):
        return CertifiedEffectiveRankStep(weight, direction, 0.0, False, effective_rank(weight))
    candidate = weight - step_size * direction
    if _is_effective_rank_half_feasible(candidate, tolerance):
        return CertifiedEffectiveRankStep(candidate, direction, 1.0, True, effective_rank(candidate))

    lower, upper = 0.0, 1.0
    for _ in range(bisection_steps):
        midpoint = (lower + upper) / 2.0
        candidate = weight - step_size * midpoint * direction
        if _is_effective_rank_half_feasible(candidate, tolerance):
            lower = midpoint
        else:
            upper = midpoint
    candidate = weight - step_size * lower * direction
    return CertifiedEffectiveRankStep(
        candidate,
        direction,
        lower,
        lower > 0.0,
        effective_rank(candidate),
    )
