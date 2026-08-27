from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class FunctionalBatch:
    args: tuple[Any, ...]
    loss_fn: Callable[[Any], torch.Tensor]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    batch_id: str | None = None


@dataclass(frozen=True)
class GuidedStepContext:
    curvature_batch: FunctionalBatch | None = None
    acceptance_batch: FunctionalBatch | None = None
    curvature_reuses_training_data: bool = False
    acceptance_reuses_other_data: bool = False
    gradient_accumulation: int = 1
    tokens: int = 0
    epoch: int = 0


@dataclass(frozen=True)
class GuidedStepResult:
    step: int
    update_applied: bool
    refreshed: bool
    guidance_used: bool
    guidance_accepted: bool
    fallback_reason: str | None
    adam_update_norm: float
    hybrid_update_norm: float
    wall_time_seconds: float
    curvature_time_seconds: float
    acceptance_time_seconds: float
    measurements: Mapping[str, float] = field(default_factory=dict)
