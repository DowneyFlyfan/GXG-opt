from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from .config import BridgeConfig
from .layer_partition import LayerGroup
from .probes import dot, norm


@dataclass(frozen=True)
class BridgeResult:
    candidate: dict[str, torch.Tensor]
    rho: float
    used_gn: bool
    accepted: bool
    update_norm: float
    reason: str | None = None


def bridge_weight(config: BridgeConfig, step: int) -> float:
    denominator = max(config.length_adam_steps - 1, 1)
    progress = 1.0 if config.length_adam_steps == 1 else min(max(step / denominator, 0.0), 1.0)
    if config.schedule == "cosine":
        progress = 0.5 - 0.5 * math.cos(math.pi * progress)
    return config.rho_start + (config.rho_end - config.rho_start) * progress


def secant_scale(
    parameter_step: Mapping[str, torch.Tensor],
    gradient_change: Mapping[str, torch.Tensor],
    curvature_floor: float,
) -> float | None:
    curvature = float(dot(parameter_step, gradient_change).item())
    if curvature <= curvature_floor:
        return None
    denominator = float(dot(gradient_change, gradient_change).item())
    if denominator <= 0:
        return None
    return min(max(curvature / denominator, 0.1), 10.0)


def blend_candidates(
    gn: Mapping[str, torch.Tensor],
    adam: Mapping[str, torch.Tensor],
    groups: tuple[LayerGroup, ...],
    reference_gradient: Mapping[str, torch.Tensor],
    config: BridgeConfig,
    step: int,
    recent_norms: Mapping[str, float | None],
    *,
    scale: float | None = None,
) -> BridgeResult:
    if set(adam) != set(reference_gradient):
        raise ValueError("Adam candidate and reference gradient must cover the same parameters")
    rho = bridge_weight(config, step)
    if rho <= 0 or set(gn) != set(adam):
        accepted = float(dot(adam, reference_gradient).item()) < 0
        return BridgeResult(dict(adam) if accepted else {}, rho, False, accepted, norm(adam) if accepted else 0.0, "adam_only")
    adjusted = {name: value.float().clone() * (scale or 1.0) for name, value in gn.items()}
    if config.per_layer_norm_match:
        for group in groups:
            group_gn = {name: adjusted[name] for name in group.parameter_names}
            group_adam = {name: adam[name] for name in group.parameter_names}
            gn_norm, adam_norm = norm(group_gn), norm(group_adam)
            multiplier = adam_norm / max(gn_norm, 1.0e-12) if adam_norm > 0 else 0.0
            for name in group.parameter_names:
                adjusted[name].mul_(multiplier)
    blended = {name: rho * adjusted[name] + (1 - rho) * adam[name].float() for name in adam}
    for group in groups:
        baseline = recent_norms.get(group.name)
        if baseline is None or baseline <= 0:
            continue
        group_value = {name: blended[name] for name in group.parameter_names}
        current = norm(group_value)
        cap = config.max_update_ratio * baseline
        if current > cap:
            factor = cap / max(current, 1.0e-12)
            for name in group.parameter_names:
                blended[name].mul_(factor)
    if float(dot(blended, reference_gradient).item()) >= 0:
        if float(dot(adam, reference_gradient).item()) < 0:
            return BridgeResult(dict(adam), rho, False, True, norm(adam), "gn_descent_guard")
        return BridgeResult({}, rho, False, False, 0.0, "non_descent")
    return BridgeResult(blended, rho, True, True, norm(blended))
