"""Shared primitives for the Kronecker-GGN optimizer family."""

from .base import LayerwiseCurvatureOptimizer
from .config import KroneckerGGNConfig, LowRankCorrectedKroneckerGGNConfig
from .curvature_operator import FunctionalCurvatureBatch, GGNLinearOperator
from .hooks import LinearCapture
from .kronecker_factors import KroneckerFactorEstimator, update_factor_ema
from .kronecker_spectral import KroneckerSpectralOperator
from .layer_registry import LayerRegistry
from .types import CurvatureFactors, CurvatureStats, CurvatureUpdate, DirectionStats

__all__ = [
    "CurvatureFactors",
    "CurvatureStats",
    "CurvatureUpdate",
    "DirectionStats",
    "FunctionalCurvatureBatch",
    "GGNLinearOperator",
    "KroneckerFactorEstimator",
    "KroneckerGGNConfig",
    "KroneckerSpectralOperator",
    "LayerRegistry",
    "LayerwiseCurvatureOptimizer",
    "LinearCapture",
    "LowRankCorrectedKroneckerGGNConfig",
    "update_factor_ema",
]
