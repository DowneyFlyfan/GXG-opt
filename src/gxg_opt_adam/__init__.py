"""Configurable layer-wise Gauss--Newton/AdamW optimizer controller."""

from .config import GXGConfig
from .optimizer import GXGOptimizer, gxg_optimizer
from .state import GXGPhase
from .types import EvalContext, FunctionalBatch, QualityResult, StepContext, StepResult

__all__ = [
    "EvalContext",
    "FunctionalBatch",
    "GXGConfig",
    "GXGOptimizer",
    "GXGPhase",
    "QualityResult",
    "StepContext",
    "StepResult",
    "gxg_optimizer",
]
