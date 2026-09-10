from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class GXGPhase(str, Enum):
    GN_BOOTSTRAP = "GN_BOOTSTRAP"
    BRIDGE_TO_ADAM = "BRIDGE_TO_ADAM"
    ADAM = "ADAM"
    GN_TRIAL = "GN_TRIAL"
    GN_CORRECTION = "GN_CORRECTION"
    FINAL_QUALITY_CHECK = "FINAL_QUALITY_CHECK"
    FINAL_GN_RECOVERY = "FINAL_GN_RECOVERY"
    DONE = "DONE"


@dataclass
class LayerGNState:
    damping: float
    accepted_steps: int = 0
    rejected_steps: int = 0
    warm_start: dict[str, torch.Tensor] = field(default_factory=dict)
    proposed_direction: dict[str, torch.Tensor] = field(default_factory=dict)
    accepted_direction: dict[str, torch.Tensor] = field(default_factory=dict)
    inner_state: dict[str, dict[str, torch.Tensor | int]] = field(default_factory=dict)
    diagnostics: dict[str, float] = field(default_factory=dict)
    curvature_batch_id: str | None = None
    reference_batch_id: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "damping": self.damping,
            "accepted_steps": self.accepted_steps,
            "rejected_steps": self.rejected_steps,
            "warm_start": self.warm_start,
            "proposed_direction": self.proposed_direction,
            "accepted_direction": self.accepted_direction,
            "inner_state": self.inner_state,
            "diagnostics": dict(self.diagnostics),
            "curvature_batch_id": self.curvature_batch_id,
            "reference_batch_id": self.reference_batch_id,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "LayerGNState":
        return cls(**state)


@dataclass
class ControllerState:
    phase: GXGPhase
    phase_step: int = 0
    current_epoch: int = -1
    phase_started_epoch: int = 0
    gn_suppressed_epoch: int | None = None
    switch_count: int = 0
    accepted_gn_in_phase: int = 0
    consecutive_gn_rejections: int = 0
    phase_started_at: float = 0.0
    reason_history: list[str] = field(default_factory=list)
    best_validation_metric: float | None = None
    best_validation_loss: float | None = None
    best_checkpoint_id: str | None = None
    quality_target_met: bool = False
    final_recovery_attempts: int = 0
    final_recovery_accepted_steps: int = 0
    final_patience: int = 0
    transition_events: list[dict[str, Any]] = field(default_factory=list)

    def state_dict(self) -> dict[str, Any]:
        value = dict(vars(self))
        value["phase"] = self.phase.value
        return value

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ControllerState":
        value = dict(state)
        value["phase"] = GXGPhase(value["phase"])
        return cls(**value)
