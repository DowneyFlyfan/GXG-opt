from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .config import GNConfig
from .krylov import ReducedSolveResult


@dataclass(frozen=True)
class TrustResult:
    direction: torch.Tensor
    alpha: float
    curvature_norm: float
    relative_update: float
    finite: bool


def apply_trust_limits(
    solve: ReducedSolveResult,
    reduced_matrix: torch.Tensor,
    parameter: torch.Tensor,
    config: GNConfig,
    eps: float = 1.0e-12,
) -> TrustResult:
    curvature_squared = torch.dot(solve.coordinates, reduced_matrix @ solve.coordinates)
    curvature_norm = math.sqrt(max(float(curvature_squared.item()), 0.0))
    direction_norm = float(solve.direction.norm().item())
    parameter_norm = float(parameter.detach().float().norm().item())
    alpha = min(
        config.alpha_max,
        config.trust_radius / math.sqrt(curvature_norm**2 + eps),
        config.max_relative_block_update * parameter_norm / (direction_norm + eps),
    )
    relative_update = alpha * direction_norm / (parameter_norm + eps)
    finite = math.isfinite(alpha) and math.isfinite(curvature_norm) and torch.isfinite(solve.direction).all().item()
    if not finite or alpha < 0:
        return TrustResult(torch.zeros_like(solve.direction), 0.0, curvature_norm, math.inf, False)
    return TrustResult(alpha * solve.direction, alpha, curvature_norm, relative_update, True)


def increase_damping(value: float, config: GNConfig) -> float:
    return min(value * config.damping_increase, config.max_damping)


def decrease_damping(value: float, config: GNConfig) -> float:
    return max(value * config.damping_decrease, config.min_damping)
