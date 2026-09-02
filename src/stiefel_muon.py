"""Muon updates constrained to square orthogonal and Stiefel manifolds.

The square route implements the analytic direction from Scientific Spaces
article 11215.  The tall route implements the closed-form skew generator from
article 11864.  Wide weights are treated as the transpose of a tall Stiefel
matrix, so every matrix has orthonormal columns in its logical orientation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


def _matrix_sign(matrix: Tensor, *, ns_steps: int, epsilon: float = 1.0e-7) -> Tensor:
    """Approximate the polar matrix sign with the Muon Newton--Schulz iteration."""
    if matrix.ndim != 2:
        raise ValueError("matrix sign requires a two-dimensional matrix")
    if ns_steps <= 0:
        raise ValueError("ns_steps must be positive")
    source_dtype = matrix.dtype
    value = matrix.float()
    value = value / value.norm().clamp_min(epsilon)
    for _ in range(ns_steps):
        gram = value.T @ value
        value = 3.4445 * value + value @ (-4.775 * gram + 2.0315 * (gram @ gram))
    return value.to(source_dtype)


def initialize_stiefel_matrix(matrix: Tensor) -> Tensor:
    """Project a tall matrix to column-orthonormal columns through QR."""
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("Stiefel initialization requires rows >= columns")
    orthogonal, triangular = torch.linalg.qr(matrix.float(), mode="reduced")
    signs = torch.where(
        torch.diagonal(triangular) < 0,
        -torch.ones(matrix.shape[1], dtype=orthogonal.dtype, device=matrix.device),
        torch.ones(matrix.shape[1], dtype=orthogonal.dtype, device=matrix.device),
    )
    return (orthogonal * signs).to(matrix.dtype)


def _column_retraction(candidate: Tensor, *, ns_steps: int) -> Tensor:
    """Apply the polar retraction and fall back to QR only for numeric drift."""
    retracted = _matrix_sign(candidate, ns_steps=ns_steps)
    identity = torch.eye(
        candidate.shape[1], dtype=retracted.dtype, device=retracted.device
    )
    error = (retracted.T @ retracted - identity).norm()
    if not torch.isfinite(error) or error > 1.0e-3:
        return initialize_stiefel_matrix(candidate)
    return retracted


def square_stiefel_update(
    weight: Tensor, momentum: Tensor, *, step_size: float, ns_steps: int
) -> tuple[Tensor, Tensor]:
    """Article-11215 square orthogonal update and polar retraction."""
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("square update requires a square matrix")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    skew = 0.5 * (weight.T @ momentum - momentum.T @ weight)
    rotation = _matrix_sign(skew, ns_steps=ns_steps)
    rotation = 0.5 * (rotation - rotation.T)
    direction = weight @ rotation
    candidate = weight - step_size * direction
    return _column_retraction(candidate, ns_steps=ns_steps), direction


def rectangular_stiefel_update(
    weight: Tensor, momentum: Tensor, *, step_size: float, ns_steps: int
) -> tuple[Tensor, Tensor]:
    """Article-11864 closed-form Stiefel direction and polar retraction."""
    if weight.ndim != 2 or weight.shape[0] < weight.shape[1]:
        raise ValueError("rectangular update requires a tall matrix")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    rows, columns = weight.shape
    if rows >= 2 * columns:
        basis, triangular = torch.linalg.qr(
            torch.cat((momentum, weight), dim=1).float(), mode="reduced"
        )
        identity = torch.eye(columns, device=weight.device, dtype=basis.dtype)
        zero = torch.zeros_like(identity)
        skew_block = torch.cat(
            (torch.cat((zero, identity), dim=1), torch.cat((-identity, zero), dim=1)),
            dim=0,
        )
        small_skew = triangular @ skew_block @ triangular.T
        generator_small = _matrix_sign(small_skew, ns_steps=ns_steps)
        generator_small = 0.5 * (generator_small - generator_small.T)
        direction = basis @ (generator_small @ (basis.T @ weight.float()))
        direction = direction.to(weight.dtype)
    else:
        generator = momentum @ weight.T - weight @ momentum.T
        generator = _matrix_sign(generator, ns_steps=ns_steps)
        generator = 0.5 * (generator - generator.T)
        direction = generator @ weight
    candidate = weight - step_size * direction
    return _column_retraction(candidate, ns_steps=ns_steps), direction


def stiefel_update(
    parameter: Tensor, gradient: Tensor, *, step_size: float, ns_steps: int
) -> tuple[Tensor, Tensor, str]:
    """Route a stored matrix to square, column-, or row-Stiefel geometry."""
    if parameter.ndim != 2 or gradient.shape != parameter.shape:
        raise ValueError("Stiefel update requires equal two-dimensional tensors")
    if parameter.shape[0] == parameter.shape[1]:
        updated, direction = square_stiefel_update(
            parameter, gradient, step_size=step_size, ns_steps=ns_steps
        )
        return updated, direction, "square"
    if parameter.shape[0] > parameter.shape[1]:
        updated, direction = rectangular_stiefel_update(
            parameter, gradient, step_size=step_size, ns_steps=ns_steps
        )
        return updated, direction, "column_stiefel"
    updated, direction = rectangular_stiefel_update(
        parameter.T, gradient.T, step_size=step_size, ns_steps=ns_steps
    )
    return updated.T, direction.T, "row_stiefel"


@torch.no_grad()
def initialize_stiefel_parameters(parameters: Iterable[Tensor]) -> None:
    """Initialize each stored parameter on its corresponding Stiefel manifold."""
    for parameter in parameters:
        if parameter.ndim != 2:
            raise ValueError("Stiefel-Muon accepts only matrix parameters")
        if parameter.shape[0] >= parameter.shape[1]:
            parameter.copy_(initialize_stiefel_matrix(parameter))
        else:
            parameter.copy_(initialize_stiefel_matrix(parameter.T).T)


class StiefelMuon(torch.optim.Optimizer):
    """Momentum Muon with a square/Stiefel closed-form route per parameter."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps),
        )
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.ndim != 2:
                        raise ValueError("Stiefel-Muon accepts only matrix parameters")
                    logical = parameter if parameter.shape[0] >= parameter.shape[1] else parameter.T
                    scale = logical.norm() / math.sqrt(logical.shape[1])
                    if not torch.isfinite(scale) or scale <= 0:
                        raise ValueError("Stiefel-Muon requires a finite non-zero parameter scale")
                    logical.copy_(initialize_stiefel_matrix(logical) * scale)
                    self.state[parameter]["stiefel_scale"] = float(scale)

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
                if parameter.ndim != 2:
                    raise ValueError("Stiefel-Muon received a non-matrix parameter")
                gradient = parameter.grad
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                buffer.mul_(group["momentum"]).add_(gradient)
                momentum = (
                    gradient.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                rows, columns = parameter.shape
                scale = float(state["stiefel_scale"])
                updated, _, route = stiefel_update(
                    parameter / scale,
                    momentum,
                    step_size=self.scaled_lr(group["lr"], rows, columns),
                    ns_steps=group["ns_steps"],
                )
                parameter.copy_(scale * updated)
                state["route"] = route
        return loss
