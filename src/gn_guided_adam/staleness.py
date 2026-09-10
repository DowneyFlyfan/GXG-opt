from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StalenessResult:
    age: int
    drift: float
    weight: float
    valid: bool


def staleness_weight(
    parameter: torch.Tensor,
    snapshot: torch.Tensor,
    *,
    step: int,
    refresh_step: int,
    max_age: int,
    max_drift: float,
    eps: float = 1.0e-12,
) -> StalenessResult:
    age = max(step - refresh_step, 0)
    current = parameter.detach().float().reshape(-1)
    reference = snapshot.to(device=current.device, dtype=current.dtype).reshape(-1)
    drift = float((current - reference).norm().item() / (reference.norm().item() + eps))
    valid = age <= max_age and drift <= max_drift
    weight = max(0.0, 1.0 - age / max(max_age, 1)) if valid else 0.0
    return StalenessResult(age, drift, weight, valid and weight > 0)
