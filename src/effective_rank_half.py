"""Certified finite steps that retain a configured effective-rank bound."""

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

    minimum_effective_rank = 0.5
    constraint_name = "effective-rank-half"

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
                if maximum_step == 0.0 or not _is_effective_rank_feasible(
                    matrix, 1.0e-6, self.minimum_effective_rank
                ):
                    state["skipped_steps"] = int(state.get("skipped_steps", 0)) + 1
                    continue
                scaled_step = group["lr"] * 0.2 * math.sqrt(max(matrix.shape))
                update = certified_effective_rank_step(
                    matrix,
                    direction_gradient,
                    step_size=min(scaled_step, 0.99 * maximum_step),
                    minimum_effective_rank=self.minimum_effective_rank,
                )
                parameter.copy_(update.weight.reshape_as(parameter))
                state["last_scale"] = update.scale
                state["accepted_steps"] = int(state.get("accepted_steps", 0)) + int(update.accepted)
        return loss


class EffectiveRankThird(EffectiveRankHalf):
    """Momentum partial-polar updates constrained to effective rank one third."""

    minimum_effective_rank = 1.0 / 3.0
    constraint_name = "effective-rank-third"


class EffectiveRankLinear(EffectiveRankHalf):
    """Certified partial-polar updates with a linearly increasing rank floor.

    The floor starts at 0.2 and reaches 0.8 after ``schedule_steps`` updates.
    A step is accepted only when its *current* scheduled floor is feasible, so
    the existing finite-step certificate is never weakened by the scheduler.
    """

    constraint_name = "effective-rank-linear-02-to-08"

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        schedule_steps: int,
        start_effective_rank: float = 0.2,
        end_effective_rank: float = 0.8,
        projection_margin: float = 0.05,
    ) -> None:
        if (
            schedule_steps <= 0
            or not 0 < start_effective_rank <= end_effective_rank <= 1
            or not 0 <= projection_margin < 1
        ):
            raise ValueError("effective-rank linear schedule is invalid")
        self.start_effective_rank = start_effective_rank
        self.end_effective_rank = end_effective_rank
        self.schedule_steps = schedule_steps
        self.projection_margin = projection_margin
        self.schedule_step = 0
        super().__init__(params, lr=lr, weight_decay=weight_decay, momentum=momentum)

    @property
    def minimum_effective_rank(self) -> float:
        # There are ``schedule_steps`` certified updates, including both
        # endpoints: the first uses 0.2 and the final uses 0.8.
        progress = min(self.schedule_step / max(self.schedule_steps - 1, 1), 1.0)
        return self.start_effective_rank + progress * (
            self.end_effective_rank - self.start_effective_rank
        )

    @torch.no_grad()
    def step(self, closure=None):
        target = min(self.minimum_effective_rank + self.projection_margin, 1.0)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                matrix = parameter.reshape(parameter.shape[0], -1)
                projection_norm = _project_effective_rank(matrix, target)
                if projection_norm == 0.0:
                    continue
                state = self.state[parameter]
                state["projected_steps"] = int(state.get("projected_steps", 0)) + 1
                state["projection_frobenius"] = float(
                    state.get("projection_frobenius", 0.0)
                ) + projection_norm
        loss = super().step(closure)
        self.schedule_step += 1
        return loss

    def state_dict(self) -> dict:
        """Persist the global schedule alongside ordinary optimizer state."""
        state = super().state_dict()
        state["effective_rank_linear_schedule"] = {
            "schedule_step": self.schedule_step,
            "schedule_steps": self.schedule_steps,
            "start_effective_rank": self.start_effective_rank,
            "end_effective_rank": self.end_effective_rank,
            "projection_margin": self.projection_margin,
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore only a schedule with the same declared endpoints."""
        state_dict = dict(state_dict)
        schedule = state_dict.pop("effective_rank_linear_schedule", None)
        if schedule is not None:
            expected = {
                "schedule_steps": self.schedule_steps,
                "start_effective_rank": self.start_effective_rank,
                "end_effective_rank": self.end_effective_rank,
                "projection_margin": self.projection_margin,
            }
            actual = {key: schedule[key] for key in expected}
            if actual != expected:
                raise ValueError("effective-rank linear checkpoint has a different schedule")
            self.schedule_step = int(schedule["schedule_step"])
        super().load_state_dict(state_dict)


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


def _constraint(weight: Tensor, minimum_effective_rank: float = 0.5) -> Tensor:
    """Return the effective-rank lower-bound constraint for a matrix."""
    rank_bound = min(weight.shape)
    frobenius_squared = torch.sum(weight.square())
    gram_norm = torch.linalg.matrix_norm(weight.transpose(-2, -1) @ weight, ord="fro")
    return math.sqrt(rank_bound * minimum_effective_rank) * gram_norm - frobenius_squared


def _is_effective_rank_feasible(
    weight: Tensor, tolerance: float, minimum_effective_rank: float
) -> bool:
    """Check the scale-invariant constraint within the matrix dtype's precision."""
    numerical_tolerance = tolerance
    if weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
        numerical_tolerance = max(numerical_tolerance, 1.0e-6)
    return effective_rank(weight) >= minimum_effective_rank - numerical_tolerance


@torch.no_grad()
def _project_effective_rank(weight: Tensor, minimum_effective_rank: float) -> float:
    """Minimally equalize singular values until the requested floor is feasible.

    The interpolation preserves singular vectors and continuously moves the
    spectrum toward equal nonzero singular values, whose normalized effective
    rank is one.  Bisection therefore finds a feasible finite projection when
    the scheduled floor rises above the matrix's current rank.
    """
    if _is_effective_rank_feasible(weight, 1.0e-6, minimum_effective_rank):
        return 0.0
    left, singular_values, right_transpose = torch.linalg.svd(weight, full_matrices=False)
    if singular_values.numel() == 0 or float(singular_values.max()) == 0.0:
        return 0.0
    equal_spectrum = torch.full_like(singular_values, singular_values.mean())

    rank_bound = singular_values.numel()

    def spectrum_is_feasible(mixing: float) -> bool:
        spectrum = torch.lerp(singular_values, equal_spectrum, mixing)
        squared = spectrum.square().sum()
        ratio = squared.square() / (rank_bound * spectrum.pow(4).sum())
        return float(ratio) >= minimum_effective_rank - 1.0e-6

    lower, upper = 0.0, 1.0
    for _ in range(48):
        midpoint = (lower + upper) / 2.0
        if spectrum_is_feasible(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    projected_spectrum = torch.lerp(singular_values, equal_spectrum, upper)
    projected = (left * projected_spectrum.unsqueeze(0)) @ right_transpose
    distance = float(torch.linalg.matrix_norm(projected - weight, ord="fro"))
    weight.copy_(projected)
    return distance


def _constraint_name(minimum_effective_rank: float) -> str:
    if minimum_effective_rank == 0.5:
        return "effective-rank-half"
    if minimum_effective_rank == 1.0 / 3.0:
        return "effective-rank-third"
    return "effective-rank"


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
    minimum_effective_rank: float = 0.5,
) -> CertifiedEffectiveRankStep:
    """Take the largest feasible partial-polar step along the supplied gradient.

    The scalar search stays inside the spectral-norm ball because it scales a
    partial polar factor.  It starts from zero, which is feasible whenever
    the input already has at least the requested effective rank, and accepts only
    candidates satisfying the finite-step constraint within the matrix dtype's
    numerical precision.
    """
    if weight.shape != gradient.shape or weight.ndim != 2:
        raise ValueError("weight and gradient must be equally shaped matrices")
    if step_size <= 0 or bisection_steps <= 0 or not 0 < minimum_effective_rank <= 1:
        raise ValueError("step_size and bisection_steps must be positive")
    rank_bound = min(weight.shape)
    maximum_step = float(torch.linalg.matrix_norm(weight, ord="fro") / math.sqrt(rank_bound))
    if not step_size < maximum_step:
        raise ValueError("step_size must be smaller than ||W||_F / sqrt(r)")
    if not _is_effective_rank_feasible(weight, tolerance, minimum_effective_rank):
        raise ValueError(f"input matrix is not {_constraint_name(minimum_effective_rank)} feasible")

    direction = _partial_polar(gradient)
    if not torch.any(direction):
        return CertifiedEffectiveRankStep(weight, direction, 0.0, False, effective_rank(weight))
    candidate = weight - step_size * direction
    if _is_effective_rank_feasible(candidate, tolerance, minimum_effective_rank):
        return CertifiedEffectiveRankStep(candidate, direction, 1.0, True, effective_rank(candidate))

    lower, upper = 0.0, 1.0
    for _ in range(bisection_steps):
        midpoint = (lower + upper) / 2.0
        candidate = weight - step_size * midpoint * direction
        if _is_effective_rank_feasible(candidate, tolerance, minimum_effective_rank):
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
