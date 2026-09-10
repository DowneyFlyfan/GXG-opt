"""Source-derived strict-feasible spectral-band optimizer.

Weights are stored with a fixed per-matrix scale.  Their normalized form lives
in the spectral band ``[sqrt((1-rho)/(1+rho)), 1]``.  The matrix update first
solves the active-boundary tangent problem approximately with the polar factor,
then uses the low-rank Cayley curve from the supplied SRIP derivation.  A failed
numerical retraction is rejected; this implementation never repairs it through
singular-value clipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from low_spectral_variance import (
    _direct_feasible_curve,
    _logical_orientation,
    _polar_factor,
)


_FINITE_PRECISION_BAND_TOLERANCE = 1.0e-4


@dataclass(frozen=True)
class SRIPBandDiagnostics:
    """Numerically checkable feasibility facts for one matrix update."""

    accepted_step: float
    upper_boundary_violation: float
    lower_boundary_violation: float
    condition_number: float
    rejected_degenerate_update: bool


def srip_condition_limit(rho: float) -> float:
    """Return the normalized band condition cap induced by ``rho``."""
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between zero and one")
    return math.sqrt((1.0 + rho) / (1.0 - rho))


def _normalized_lower_bound(rho: float) -> float:
    return 1.0 / srip_condition_limit(rho)


@torch.no_grad()
def srip_band_parameter_names(
    model: nn.Module,
    *,
    rho: float,
    candidates: set[str] | None = None,
) -> set[str]:
    """Select existing Muon-eligible matrices already inside the SRIP band."""
    limit = srip_condition_limit(rho)
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
            and singular_values[0] / singular_values[-1] <= limit
        ):
            selected.add(name)
    return selected


def _psd_part(matrix: Tensor) -> Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2)
    return (vectors * values.clamp_min(0).unsqueeze(0)) @ vectors.T


def _boundary_subspaces(
    left: Tensor, singular_values: Tensor, right_transpose: Tensor, *, lower: float,
    tolerance: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    upper_mask = singular_values >= 1.0 - tolerance
    lower_mask = singular_values <= lower * (1.0 + tolerance)
    return (
        left[:, upper_mask],
        right_transpose[upper_mask].T,
        left[:, lower_mask],
        right_transpose[lower_mask].T,
    )


def _boundary_form(weight: Tensor, phi: Tensor, left: Tensor, right: Tensor) -> Tensor:
    if right.numel() == 0:
        return weight.new_empty((0, 0))
    return (right.T @ (weight.T @ phi + phi.T @ weight) @ right +
            right.T @ (weight.T @ phi + phi.T @ weight).T @ right) / 2


def srip_band_direction(
    weight: Tensor,
    gradient: Tensor,
    *,
    rho: float,
    dual_steps: int = 8,
    boundary_tolerance: float = 2.0e-4,
) -> tuple[Tensor, float, float]:
    """Approximate the SRIP dual direction and satisfy its tangent cone.

    ``weight`` is tall, normalized, and has top singular value one.  The small
    positive-semidefinite multiplier descent is the smooth numerical version of
    the supplied nuclear-norm dual; the final correction enforces the primal
    upper and lower boundary inequalities before direction normalization.
    """
    if weight.ndim != 2 or gradient.shape != weight.shape or dual_steps <= 0:
        raise ValueError("SRIP direction requires equal matrices and positive dual steps")
    left, singular_values, right_transpose = torch.linalg.svd(weight.float(), full_matrices=False)
    lower = _normalized_lower_bound(rho)
    upper_left, upper_right, lower_left, lower_right = _boundary_subspaces(
        left, singular_values, right_transpose, lower=lower, tolerance=boundary_tolerance
    )
    lambda_upper = weight.new_zeros((upper_right.shape[1], upper_right.shape[1]), dtype=torch.float32)
    lambda_lower = weight.new_zeros((lower_right.shape[1], lower_right.shape[1]), dtype=torch.float32)
    phi = _polar_factor(gradient.float())
    for iteration in range(dual_steps):
        shift = gradient.float()
        if upper_right.numel():
            shift = shift + 2.0 * weight.float() @ upper_right @ lambda_upper @ upper_right.T
        if lower_right.numel():
            shift = shift - 2.0 * weight.float() @ lower_right @ lambda_lower @ lower_right.T
        phi = _polar_factor(shift)
        rate = 0.5 / math.sqrt(iteration + 1)
        upper_form = _boundary_form(weight.float(), phi, upper_left, upper_right)
        lower_form = _boundary_form(weight.float(), phi, lower_left, lower_right)
        if upper_form.numel():
            lambda_upper = _psd_part(lambda_upper - rate * upper_form)
        if lower_form.numel():
            lambda_lower = _psd_part(lambda_lower + rate * lower_form)
    upper_form = _boundary_form(weight.float(), phi, upper_left, upper_right)
    lower_form = _boundary_form(weight.float(), phi, lower_left, lower_right)
    if upper_form.numel():
        phi = phi + weight.float() @ upper_right @ _psd_part(-upper_form / 2) @ upper_right.T
    if lower_form.numel():
        phi = phi - weight.float() @ lower_right @ _psd_part(lower_form / 2) @ lower_right.T
    spectral_norm = torch.linalg.svdvals(phi)[0].clamp_min(1.0e-8)
    phi = phi / spectral_norm
    upper_form = _boundary_form(weight.float(), phi, upper_left, upper_right)
    lower_form = _boundary_form(weight.float(), phi, lower_left, lower_right)
    upper_violation = float((-torch.linalg.eigvalsh(upper_form).min()).clamp_min(0)) if upper_form.numel() else 0.0
    lower_violation = float(torch.linalg.eigvalsh(lower_form).max().clamp_min(0)) if lower_form.numel() else 0.0
    return phi, upper_violation, lower_violation


def srip_band_update(
    weight: Tensor,
    gradient: Tensor,
    *,
    rho: float,
    step_size: float,
    dual_steps: int = 8,
) -> tuple[Tensor, SRIPBandDiagnostics]:
    """Apply an SRIP tangent direction through a strictly feasible Cayley path."""
    if gradient.shape != weight.shape or step_size <= 0:
        raise ValueError("SRIP update requires equal matrices and positive step size")
    logical_weight, transposed = _logical_orientation(weight)
    logical_gradient = gradient.T if transposed else gradient
    values = torch.linalg.svdvals(logical_weight.float())
    lower = _normalized_lower_bound(rho)
    if (
        values[0] > 1.0 + _FINITE_PRECISION_BAND_TOLERANCE
        or values[-1] < lower * (1.0 - _FINITE_PRECISION_BAND_TOLERANCE)
    ):
        raise ValueError(
            "SRIP weight lies outside the selected spectral band: "
            f"top={float(values[0]):.8f}, minimum={float(values[-1]):.8f}, lower={lower:.8f}"
        )
    phi, upper_violation, lower_violation = srip_band_direction(
        logical_weight, logical_gradient, rho=rho, dual_steps=dual_steps
    )
    left, singular_values, right_transpose = torch.linalg.svd(logical_weight.float(), full_matrices=False)
    candidate = logical_weight.float()
    accepted_step = 0.0
    rejected_degenerate = False
    for _ in range(12):
        requested_step = step_size if accepted_step == 0 else accepted_step / 2
        try:
            if singular_values[0] >= 1.0 - 2e-4:
                proposal, used_step = _direct_feasible_curve(
                    logical_weight.float(), -phi, left, singular_values, right_transpose,
                    condition_limit=srip_condition_limit(rho), step_size=requested_step,
                )
            else:
                # In the strict interior no boundary rotation is required:
                # the source curve is W+hE with L=R=0.  Backtracking below
                # certifies the finite point stays in the spectral band.
                proposal, used_step = logical_weight.float() - requested_step * phi, requested_step
        except ValueError:
            rejected_degenerate = True
            break
        proposed_values = torch.linalg.svdvals(proposal)
        if proposed_values[0] <= 1.0 + 2e-5 and proposed_values[-1] >= lower * (1.0 - 2e-5):
            candidate, accepted_step = proposal, used_step
            break
        accepted_step = used_step
    diagnostics = SRIPBandDiagnostics(
        accepted_step=accepted_step,
        upper_boundary_violation=upper_violation,
        lower_boundary_violation=lower_violation,
        condition_number=float(values[0] / values[-1]),
        rejected_degenerate_update=rejected_degenerate,
    )
    candidate = candidate.to(weight.dtype)
    return (candidate.T if transposed else candidate), diagnostics


class SRIPBand(torch.optim.Optimizer):
    """Momentum SRIP-band optimizer with fixed physical parameter scales."""

    def __init__(self, params, *, lr: float, rho: float, momentum: float = 0.95, nesterov: bool = True, dual_steps: int = 8) -> None:
        if lr <= 0 or not 0.0 < rho < 1.0 or not 0.0 <= momentum < 1.0 or dual_steps <= 0:
            raise ValueError("SRIP-band hyperparameters are invalid")
        super().__init__(params, dict(lr=lr, rho=rho, momentum=momentum, nesterov=nesterov, dual_steps=dual_steps))
        with torch.no_grad():
            limit = srip_condition_limit(rho)
            for group in self.param_groups:
                for parameter in group["params"]:
                    logical, _ = _logical_orientation(parameter)
                    values = torch.linalg.svdvals(logical.float())
                    if not torch.isfinite(values).all() or values[-1] <= 0 or values[0] / values[-1] > limit:
                        raise ValueError("parameter is not initially feasible for the SRIP band")
                    self.state[parameter]["spectral_scale"] = float(values[0])

    @staticmethod
    def scaled_lr(lr: float, rows: int, columns: int) -> float:
        return lr * 0.2 * math.sqrt(max(rows, columns))

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
                direction = parameter.grad.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
                scale = float(state["spectral_scale"])
                updated, diagnostics = srip_band_update(
                    parameter / scale, direction / scale, rho=group["rho"],
                    step_size=self.scaled_lr(group["lr"], *parameter.shape), dual_steps=group["dual_steps"],
                )
                parameter.copy_(scale * updated)
                state["diagnostics"] = diagnostics
        return loss
