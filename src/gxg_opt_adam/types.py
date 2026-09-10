from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .state import GXGPhase


TensorMap = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class FunctionalBatch:
    """Caller-selected data for functional GN operations.

    `loss_fn` maps model output to a scalar true loss. The optimizer does not own or
    advance the loader that produced `args` and `kwargs`.
    """

    args: tuple[Any, ...]
    loss_fn: Callable[[Any], torch.Tensor]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    batch_id: str | None = None


@dataclass
class StepContext:
    epoch: int = 0
    reference_loss_closure: Callable[[], torch.Tensor] | None = None
    gn_batch: FunctionalBatch | None = None
    reference_batch_id: str | None = None
    reference_gradient: TensorMap | None = None
    shadow_microbatch_gradients: Sequence[TensorMap] = ()
    adam_equivalent_batches: float = 1.0
    nominal_budget_exhausted: bool = False
    elapsed_seconds: float | None = None
    data_exhausted: bool = False


@dataclass(frozen=True)
class EvalContext:
    split: str = "validation"
    metric: float | None = None
    loss: float | None = None
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class StepResult:
    phase: GXGPhase
    update_accepted: bool
    adam_update_norm: float = 0.0
    gn_update_norm: float = 0.0
    bridge_update_norm: float = 0.0
    switch_reason: str | None = None
    wall_time_seconds: float = 0.0
    measurements: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityResult:
    metric: float
    loss: float | None
    phase: GXGPhase
    improved: bool
    target_met: bool
    checkpoint_id: str | None
