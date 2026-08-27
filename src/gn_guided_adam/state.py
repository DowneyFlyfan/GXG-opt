from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BlockGuidanceState:
    damping: float
    basis: torch.Tensor | None = None
    reduced_matrix: torch.Tensor | None = None
    parameter_snapshot: torch.Tensor | None = None
    refresh_step: int = -1
    cooldown_until: int = -1
    consecutive_failures: int = 0
    accepted_events: int = 0
    rejected_events: int = 0
    subspace_overlap: float = 0.0
    last_metrics: dict[str, float] = field(default_factory=dict)
    curvature_batch_id: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "damping": self.damping,
            "basis": self.basis,
            "reduced_matrix": self.reduced_matrix,
            "parameter_snapshot": self.parameter_snapshot,
            "refresh_step": self.refresh_step,
            "cooldown_until": self.cooldown_until,
            "consecutive_failures": self.consecutive_failures,
            "accepted_events": self.accepted_events,
            "rejected_events": self.rejected_events,
            "subspace_overlap": self.subspace_overlap,
            "last_metrics": dict(self.last_metrics),
            "curvature_batch_id": self.curvature_batch_id,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], device: torch.device) -> "BlockGuidanceState":
        value = dict(state)
        for name in ("basis", "reduced_matrix", "parameter_snapshot"):
            if value.get(name) is not None:
                value[name] = value[name].to(device=device)
        return cls(**value)
