"""Budgeted, fixed-duty-cycle layer-wise GGN-guided AdamW."""

from .config import FixedEpochDutyCycleConfig, GuidedAdamConfig
from .optimizer import GNGuidedAdamW, gn_guided_adamw
from .types import FunctionalBatch, GuidedStepContext, GuidedStepResult

__all__ = [
    "FunctionalBatch",
    "FixedEpochDutyCycleConfig",
    "GNGuidedAdamW",
    "GuidedAdamConfig",
    "GuidedStepContext",
    "GuidedStepResult",
    "gn_guided_adamw",
]
