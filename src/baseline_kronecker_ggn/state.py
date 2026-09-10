from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator


@dataclass
class BaselineLayerState:
    activation: Tensor | None = None
    output: Tensor | None = None
    spectral: KroneckerSpectralOperator | None = None
    damping: float = 1.0e-3
    factor_age: int = 0
    inverse_age: int = 0
    update_count: int = 0
    fallback_status: str | None = "factors_uninitialized"
    diagnostics: dict[str, float] = field(default_factory=dict)

    def state_dict(self) -> dict[str, Any]:
        return {
            "activation": self.activation,
            "output": self.output,
            "spectral": None if self.spectral is None else self.spectral.state_dict(),
            "damping": self.damping,
            "factor_age": self.factor_age,
            "inverse_age": self.inverse_age,
            "update_count": self.update_count,
            "fallback_status": self.fallback_status,
            "diagnostics": dict(self.diagnostics),
        }
