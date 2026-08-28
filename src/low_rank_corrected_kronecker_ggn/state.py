from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor


@dataclass
class CorrectionState:
    basis: Tensor | None = None
    eigenvalues: Tensor | None = None
    residuals: Tensor | None = None
    age: int = 0
    refresh_count: int = 0
    requested_rank: int = 0
    accepted_rank: int = 0
    matvec_count: int = 0
    build_time_seconds: float = 0.0
    memory_bytes: int = 0
    valid: bool = False
    failure_reason: str | None = "not_initialized"
    cross_batch_reliability: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def invalidate(self, reason: str) -> None:
        self.basis = None
        self.eigenvalues = None
        self.residuals = None
        self.accepted_rank = 0
        self.memory_bytes = 0
        self.valid = False
        self.failure_reason = reason

    def state_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "eigenvalues": self.eigenvalues,
            "residuals": self.residuals,
            "age": self.age,
            "refresh_count": self.refresh_count,
            "requested_rank": self.requested_rank,
            "accepted_rank": self.accepted_rank,
            "matvec_count": self.matvec_count,
            "build_time_seconds": self.build_time_seconds,
            "memory_bytes": self.memory_bytes,
            "valid": self.valid,
            "failure_reason": self.failure_reason,
            "cross_batch_reliability": self.cross_batch_reliability,
            "diagnostics": dict(self.diagnostics),
        }
