"""Muon constrained to the fixed spectral-norm sphere from article 11241."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor

def _dot(left: Tensor, right: Tensor) -> Tensor:
    """Return the Frobenius inner product used throughout article 11241."""
    return torch.sum(left * right)


def _exact_matrix_sign(matrix: Tensor) -> Tensor:
    """The exact singular-value-decomposition ``msign`` used in article 11241."""
    left, _, right_transpose = torch.linalg.svd(matrix.float(), full_matrices=False)
    return (left @ right_transpose).to(matrix.dtype)


def spectral_tangent(
    matrix: Tensor,
    *,
    right_seed: Tensor | None = None,
    power_iterations: int = 10,
    epsilon: float = 1.0e-7,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``Theta=u_1v_1^T``, the top singular value, and a warm-start vector."""
    if matrix.ndim != 2:
        raise ValueError("spectral-sphere Muon requires two-dimensional matrices")
    if power_iterations <= 0:
        raise ValueError("power_iterations must be positive")
    value = matrix.float()
    columns = value.shape[1]
    if right_seed is None:
        right = torch.full(
            (columns,), 1.0 / math.sqrt(columns), device=value.device, dtype=value.dtype
        )
    else:
        if right_seed.shape != (columns,):
            raise ValueError("right_seed has incompatible shape")
        right = right_seed.to(device=value.device, dtype=value.dtype)
        right = right / right.norm().clamp_min(epsilon)
    for _ in range(power_iterations):
        left = value @ right
        left = left / left.norm().clamp_min(epsilon)
        right = value.T @ left
        right = right / right.norm().clamp_min(epsilon)
    singular_value = torch.dot(left, value @ right).abs().clamp_min(epsilon)
    return torch.outer(left, right).to(matrix.dtype), singular_value, right.to(matrix.dtype)


def spectral_sphere_direction(
    gradient: Tensor,
    theta: Tensor,
    *,
    lambda_steps: int,
    ns_steps: int,
) -> tuple[Tensor, Tensor]:
    """Article-11241 fixed-point solve for the tangent Muon direction."""
    if gradient.ndim != 2 or theta.shape != gradient.shape:
        raise ValueError("gradient and theta must be equal two-dimensional matrices")
    if lambda_steps <= 0:
        raise ValueError("lambda_steps must be positive")
    theta_norm_squared = _dot(theta, theta).clamp_min(1.0e-7)
    multiplier = -_dot(theta, gradient) / theta_norm_squared
    for _ in range(lambda_steps):
        shifted_gradient = gradient + multiplier * theta
        direction = _exact_matrix_sign(shifted_gradient)
        square_root = shifted_gradient.T @ direction
        theta_direction = theta.T @ direction
        multiplier = (
            _dot(theta_direction, square_root)
            - torch.trace(theta_direction) * torch.trace(square_root) / gradient.shape[1]
            - _dot(theta, gradient)
        ) / theta_norm_squared
    return _exact_matrix_sign(gradient + multiplier * theta), multiplier


class SpectralSphereMuon(torch.optim.Optimizer):
    """Momentum Muon with the article-11241 spectral-sphere constraint."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        lambda_steps: int = 10,
        power_iterations: int = 10,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if ns_steps <= 0 or lambda_steps <= 0 or power_iterations <= 0:
            raise ValueError("iteration counts must be positive")
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                lambda_steps=lambda_steps,
                power_iterations=power_iterations,
            ),
        )
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.ndim != 2:
                        raise ValueError("SpectralSphereMuon accepts only matrix parameters")
                    _, spectral_scale, right = spectral_tangent(
                        parameter, power_iterations=group["power_iterations"]
                    )
                    self.state[parameter]["spectral_scale"] = float(spectral_scale)
                    self.state[parameter]["right_vector"] = right

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
                gradient = parameter.grad
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                buffer.mul_(group["momentum"]).add_(gradient)
                momentum = (
                    gradient.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                scale = float(state["spectral_scale"])
                normalized_parameter = parameter / scale
                theta, _, right = spectral_tangent(
                    normalized_parameter,
                    right_seed=state.get("right_vector"),
                    power_iterations=group["power_iterations"],
                )
                direction, multiplier = spectral_sphere_direction(
                    momentum,
                    theta,
                    lambda_steps=group["lambda_steps"],
                    ns_steps=group["ns_steps"],
                )
                step_size = self.scaled_lr(group["lr"], *parameter.shape)
                candidate = normalized_parameter - step_size * direction
                _, candidate_norm, candidate_right = spectral_tangent(
                    candidate,
                    right_seed=right,
                    power_iterations=group["power_iterations"],
                )
                parameter.copy_(scale * candidate / candidate_norm)
                state["right_vector"] = candidate_right
                state["lambda"] = float(multiplier)
        return loss
