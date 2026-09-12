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


@dataclass(frozen=True)
class JointNewtonEffectiveRankStep(CertifiedEffectiveRankStep):
    """A certified active-constraint step and its joint-solver evidence."""

    solver: str
    multiplier: float
    residual: float
    certificate_gap: float


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

    def _effective_rank_update(
        self, matrix: Tensor, direction_gradient: Tensor, scaled_step: float
    ) -> CertifiedEffectiveRankStep:
        """Produce one finite-step-certified matrix update."""
        return certified_effective_rank_step(
            matrix,
            direction_gradient,
            step_size=scaled_step,
            minimum_effective_rank=self.minimum_effective_rank,
        )

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
                update = self._effective_rank_update(
                    matrix, direction_gradient, min(scaled_step, 0.99 * maximum_step)
                )
                parameter.copy_(update.weight.reshape_as(parameter))
                state["last_scale"] = update.scale
                state["accepted_steps"] = int(state.get("accepted_steps", 0)) + int(update.accepted)
                if isinstance(update, JointNewtonEffectiveRankStep):
                    state["last_solver"] = update.solver
                    state[f"{update.solver}_steps"] = int(
                        state.get(f"{update.solver}_steps", 0)
                    ) + 1
        return loss


class EffectiveRankThird(EffectiveRankHalf):
    """Momentum partial-polar updates constrained to effective rank one third."""

    minimum_effective_rank = 1.0 / 3.0
    constraint_name = "effective-rank-third"


class EffectiveRankJointNewton(EffectiveRankHalf):
    """Effective-rank-half optimizer with a smooth joint-Newton active solver."""

    constraint_name = "effective-rank-half-joint-newton"

    def _effective_rank_update(
        self, matrix: Tensor, direction_gradient: Tensor, scaled_step: float
    ) -> JointNewtonEffectiveRankStep:
        update = joint_newton_effective_rank_step(
            matrix,
            direction_gradient,
            step_size=scaled_step,
            minimum_effective_rank=self.minimum_effective_rank,
        )
        return update


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


class EffectiveRankLinearJointNewton(EffectiveRankLinear):
    """0.2-to-0.8 effective-rank schedule with certified joint-Newton solves.

    Each scheduled floor is still enforced by ``EffectiveRankLinear`` before
    the active solve.  The joint solver therefore accelerates only the
    per-matrix constrained direction; it cannot relax the schedule.
    """

    constraint_name = "effective-rank-linear-02-to-08-joint-newton"

    def _effective_rank_update(
        self, matrix: Tensor, direction_gradient: Tensor, scaled_step: float
    ) -> JointNewtonEffectiveRankStep:
        return joint_newton_effective_rank_step(
            matrix,
            direction_gradient,
            step_size=scaled_step,
            minimum_effective_rank=self.minimum_effective_rank,
        )


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


def _polar_newton_schulz(
    matrix: Tensor, directions: tuple[Tensor, ...] = (), iterations: int = 12
) -> tuple[Tensor, tuple[Tensor, ...], float]:
    """Approximate a full-column-rank polar factor and Frechet directions.

    The iteration uses only matrix products.  It intentionally returns a
    residual so callers can reject a nonsmooth or insufficiently converged
    branch rather than treating it as a polar factor.
    """
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("Newton-Schulz polar iteration requires a tall matrix")
    scale = torch.linalg.matrix_norm(matrix, ord="fro")
    if not torch.isfinite(scale) or float(scale) == 0.0:
        raise ValueError("polar matrix must be finite and nonzero")
    iterate = matrix / scale
    tangent = tuple(
        direction / scale - matrix * (torch.sum(matrix * direction) / scale.pow(3))
        for direction in directions
    )
    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
    for _ in range(iterations):
        gram = iterate.transpose(-2, -1) @ iterate
        correction = 3.0 * identity - gram
        next_tangent = tuple(
            0.5
            * (
                direction @ correction
                - iterate
                @ (direction.transpose(-2, -1) @ iterate + iterate.transpose(-2, -1) @ direction)
            )
            for direction in tangent
        )
        iterate = 0.5 * iterate @ correction
        tangent = next_tangent
    residual = float(torch.linalg.matrix_norm(iterate.transpose(-2, -1) @ iterate - identity, ord="fro"))
    return iterate, tangent, residual


def _polar_residual_tolerance(matrix: Tensor) -> float:
    """Return a dtype-aware orthogonality gate for a computed polar factor.

    Newton--Schulz stagnates at the working precision; an absolute ``1e-8``
    gate is below float32 roundoff for GPT projection matrices.  The accepted
    factor is subsequently scaled by this residual, preserving the spectral
    bound used by the finite-step certificate.
    """
    return max(
        1.0e-8,
        16.0 * torch.finfo(matrix.dtype).eps * math.sqrt(min(matrix.shape)),
    )


def _joint_constraint_terms(
    weight: Tensor, direction: Tensor, step_size: float, minimum_effective_rank: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return the relaxation residual, normal, candidate weight, Gram, and norm."""
    rank_bound = min(weight.shape)
    chi = math.sqrt(rank_bound * minimum_effective_rank)
    candidate = weight - step_size * direction
    gram = candidate.transpose(-2, -1) @ candidate
    gram_norm = torch.linalg.matrix_norm(gram, ord="fro")
    if float(gram_norm) == 0.0:
        raise ValueError("effective-rank candidate has zero Gram norm")
    residual = (
        chi * gram_norm
        - torch.sum(weight.square())
        + 2.0 * step_size * torch.sum(weight * direction)
        - step_size**2 * rank_bound
    )
    normal = 2.0 * (chi * candidate @ gram / gram_norm - weight)
    return residual, normal, candidate, gram, gram_norm


def _joint_constraint_hessian(candidate: Tensor, gram: Tensor, gram_norm: Tensor, tangent: Tensor) -> Tensor:
    """Apply the effective-rank constraint Hessian without materializing it."""
    candidate_gram = candidate @ gram
    gram_tangent = tangent.transpose(-2, -1) @ candidate + candidate.transpose(-2, -1) @ tangent
    return (
        2.0 * (tangent @ gram + candidate @ gram_tangent) / gram_norm
        - 4.0
        * candidate_gram
        * torch.sum(candidate_gram * tangent)
        / gram_norm.pow(3)
    )


def _joint_richardson_solve(
    right_hand_side: Tensor,
    operator,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
) -> Tensor | None:
    """Solve a near-identity joint Newton system by residual-corrected steps."""
    solution = right_hand_side.clone()
    scale = max(float(torch.linalg.matrix_norm(right_hand_side, ord="fro")), 1.0)
    for _ in range(maximum_iterations):
        residual = right_hand_side - operator(solution)
        if float(torch.linalg.matrix_norm(residual, ord="fro")) <= relative_tolerance * scale:
            return solution
        solution.add_(residual)
    residual = right_hand_side - operator(solution)
    if float(torch.linalg.matrix_norm(residual, ord="fro")) <= relative_tolerance * scale:
        return solution
    return None


@torch.no_grad()
def joint_newton_effective_rank_step(
    weight: Tensor,
    gradient: Tensor,
    *,
    step_size: float,
    tolerance: float = 1.0e-10,
    minimum_effective_rank: float = 0.5,
    maximum_newton_steps: int = 6,
    polar_iterations: int = 12,
    linear_iterations: int = 16,
) -> JointNewtonEffectiveRankStep:
    """Solve a smooth active effective-rank step with a coupled Newton update.

    This is deliberately conservative: the Newton--Schulz polar residual,
    Schur complement, original finite-step constraint, and a computable dual
    gap must all pass.  Any failed smooth-branch check uses the existing
    bisection-certified update instead of applying an unverified direction.
    """
    if weight.shape != gradient.shape or weight.ndim != 2:
        raise ValueError("weight and gradient must be equally shaped matrices")
    if (
        step_size <= 0
        or not 0 < minimum_effective_rank <= 1
        or maximum_newton_steps <= 0
        or polar_iterations <= 0
        or linear_iterations <= 0
    ):
        raise ValueError("joint Newton effective-rank hyperparameters are invalid")
    # Newton--Schulz is written below for a tall matrix.  The objective,
    # spectral bound, and effective-rank constraint are transpose invariant,
    # so wide GPT projection matrices use the same solve on their transpose.
    if weight.shape[0] < weight.shape[1]:
        transposed = joint_newton_effective_rank_step(
            weight.transpose(-2, -1),
            gradient.transpose(-2, -1),
            step_size=step_size,
            tolerance=tolerance,
            minimum_effective_rank=minimum_effective_rank,
            maximum_newton_steps=maximum_newton_steps,
            polar_iterations=polar_iterations,
            linear_iterations=linear_iterations,
        )
        return JointNewtonEffectiveRankStep(
            transposed.weight.transpose(-2, -1),
            transposed.direction.transpose(-2, -1),
            transposed.scale,
            transposed.accepted,
            transposed.effective_rank,
            transposed.solver,
            transposed.multiplier,
            transposed.residual,
            transposed.certificate_gap,
        )
    maximum_step = float(torch.linalg.matrix_norm(weight, ord="fro") / math.sqrt(min(weight.shape)))
    if not step_size < maximum_step:
        raise ValueError("step_size must be smaller than ||W||_F / sqrt(r)")
    if not _is_effective_rank_feasible(weight, tolerance, minimum_effective_rank):
        raise ValueError(f"input matrix is not {_constraint_name(minimum_effective_rank)} feasible")

    def fallback() -> JointNewtonEffectiveRankStep:
        certified = certified_effective_rank_step(
            weight, gradient, step_size=step_size, tolerance=tolerance,
            minimum_effective_rank=minimum_effective_rank,
        )
        return JointNewtonEffectiveRankStep(
            certified.weight, certified.direction, certified.scale, certified.accepted,
            certified.effective_rank, "certified_fallback", 0.0, math.inf, math.inf,
        )

    try:
        direction, _, polar_residual = _polar_newton_schulz(
            gradient, iterations=polar_iterations
        )
    except ValueError:
        return fallback()
    if polar_residual > _polar_residual_tolerance(gradient):
        return fallback()
    direction = direction / math.sqrt(1.0 + polar_residual)
    if _is_effective_rank_feasible(
        weight - step_size * direction, tolerance, minimum_effective_rank
    ):
        return JointNewtonEffectiveRankStep(
            weight - step_size * direction, direction, 1.0, True,
            effective_rank(weight - step_size * direction), "unconstrained", 0.0, polar_residual, 0.0,
        )

    multiplier = 0.0
    rank_bound = min(weight.shape)
    chi = math.sqrt(rank_bound * minimum_effective_rank)
    margin = 512.0 * torch.finfo(weight.dtype).eps * torch.sum(weight.square())
    initial_normal_scale: float | None = None
    accepted_residual = math.inf
    for _ in range(maximum_newton_steps):
        try:
            constraint, normal, candidate, gram, gram_norm = _joint_constraint_terms(
                weight, direction, step_size, minimum_effective_rank
            )
            corrected_gradient = gradient + multiplier * normal
            polar, polar_directions, polar_residual = _polar_newton_schulz(
                corrected_gradient, (normal,), polar_iterations
            )
        except ValueError:
            return fallback()
        if polar_residual > _polar_residual_tolerance(corrected_gradient):
            return fallback()
        polar_normal = polar_directions[0]
        residual_direction = direction - polar
        scalar_residual = (constraint + margin) / step_size
        if initial_normal_scale is None:
            initial_normal_scale = max(float(torch.linalg.matrix_norm(normal, ord="fro")), 1.0)
        combined_residual = math.sqrt(
            float(torch.sum(residual_direction.square()))
            + float(scalar_residual.square()) / initial_normal_scale**2
        )
        newton_tolerance = max(
            1.0e-8,
            16.0 * torch.finfo(weight.dtype).eps * math.sqrt(weight.numel()),
        )
        if combined_residual <= newton_tolerance:
            accepted_residual = combined_residual
            break

        def polar_apply(tangent: Tensor) -> Tensor:
            try:
                _, tangents, residual = _polar_newton_schulz(
                    corrected_gradient, (tangent,), polar_iterations
                )
            except ValueError:
                return torch.full_like(tangent, math.nan)
            if residual > _polar_residual_tolerance(corrected_gradient):
                return torch.full_like(tangent, math.nan)
            return tangents[0]

        def operator(tangent: Tensor) -> Tensor:
            hessian = _joint_constraint_hessian(candidate, gram, gram_norm, tangent)
            return tangent + multiplier * step_size * chi * polar_apply(hessian)

        solve_tolerance = max(1.0e-10, min(0.25, 0.1 * combined_residual))
        residual_solution = _joint_richardson_solve(
            residual_direction, operator, maximum_iterations=linear_iterations,
            relative_tolerance=solve_tolerance,
        )
        normal_solution = _joint_richardson_solve(
            polar_normal, operator, maximum_iterations=linear_iterations,
            relative_tolerance=solve_tolerance,
        )
        if residual_solution is None or normal_solution is None:
            return fallback()
        denominator = torch.sum(normal * normal_solution)
        if not torch.isfinite(denominator) or float(denominator) <= 1.0e-14:
            return fallback()
        multiplier_step = (scalar_residual + torch.sum(normal * residual_solution)) / denominator
        direction_step = -residual_solution + normal_solution * multiplier_step
        accepted = False
        for line_search in range(12):
            fraction = 0.5**line_search
            next_multiplier = multiplier + fraction * float(multiplier_step)
            if next_multiplier < 0.0:
                continue
            next_direction = direction + fraction * direction_step
            try:
                next_constraint, next_normal, _, _, _ = _joint_constraint_terms(
                    weight, next_direction, step_size, minimum_effective_rank
                )
                next_polar, _, next_polar_residual = _polar_newton_schulz(
                    gradient + next_multiplier * next_normal, iterations=polar_iterations
                )
            except ValueError:
                continue
            if next_polar_residual > _polar_residual_tolerance(
                gradient + next_multiplier * next_normal
            ):
                continue
            next_residual_direction = next_direction - next_polar
            next_scalar_residual = (next_constraint + margin) / step_size
            next_combined = math.sqrt(
                float(torch.sum(next_residual_direction.square()))
                + float(next_scalar_residual.square()) / initial_normal_scale**2
            )
            if math.isfinite(next_combined) and next_combined < combined_residual:
                direction = next_direction
                multiplier = next_multiplier
                accepted_residual = next_combined
                accepted = True
                break
        if not accepted:
            return fallback()
    else:
        return fallback()

    try:
        constraint, normal, _, _, _ = _joint_constraint_terms(
            weight, direction, step_size, minimum_effective_rank
        )
        corrected_gradient = gradient + multiplier * normal
        polar, _, polar_residual = _polar_newton_schulz(
            corrected_gradient, iterations=polar_iterations
        )
    except ValueError:
        return fallback()
    orthogonality_error = torch.linalg.matrix_norm(
        polar.transpose(-2, -1) @ polar
        - torch.eye(rank_bound, dtype=polar.dtype, device=polar.device), ord="fro"
    )
    feasible_direction = polar / torch.sqrt(1.0 + orthogonality_error)
    updated = weight - step_size * feasible_direction
    if polar_residual > _polar_residual_tolerance(corrected_gradient) or not _is_effective_rank_feasible(
        updated, tolerance, minimum_effective_rank
    ):
        return fallback()
    nuclear_upper = torch.sum(corrected_gradient * polar) + math.sqrt(rank_bound) * torch.linalg.matrix_norm(
        corrected_gradient - corrected_gradient @ (polar.transpose(-2, -1) @ polar), ord="fro"
    )
    objective = torch.sum(gradient * feasible_direction)
    upper = objective + nuclear_upper - torch.sum(corrected_gradient * feasible_direction) - multiplier * constraint / step_size
    certificate_gap = max(float(upper - objective), 0.0)
    objective_scale = max(abs(float(objective)), abs(float(upper)), 1.0)
    if certificate_gap / objective_scale > 1.0e-5:
        return fallback()
    return JointNewtonEffectiveRankStep(
        updated, feasible_direction, 1.0, True, effective_rank(updated), "joint_newton",
        multiplier, accepted_residual, certificate_gap,
    )
