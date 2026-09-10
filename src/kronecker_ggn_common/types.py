from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from torch import Tensor, nn


class MatrixLinearOperator(Protocol):
    def matvec(self, layer_id: str, vector: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class LayerInfo:
    layer_id: str
    module_path: str
    module: nn.Module
    weight: nn.Parameter
    bias: nn.Parameter | None
    matrix_shape: tuple[int, int]
    block_type: str
    factor_update_frequency: int
    correction_eligible: bool
    supported: bool
    fallback_reason: str | None = None
    distributed_owner: int | None = None


@dataclass(frozen=True)
class CurvatureFactors:
    activation: Tensor
    output: Tensor
    sample_count: int = 1


@dataclass(frozen=True)
class CurvatureUpdate:
    curvature_mode: str
    factors: Mapping[str, CurvatureFactors | tuple[Tensor, Tensor]]
    ggn_operators: Mapping[str, Any] = field(default_factory=dict)
    measurements: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureStats:
    curvature_mode: str
    updated_layers: tuple[str, ...]
    skipped_layers: Mapping[str, str]
    wall_time_seconds: float
    factor_update_seconds: float
    spectral_update_seconds: float
    correction_update_seconds: float = 0.0
    measurements: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerDirectionStats:
    layer_id: str
    used_curvature: bool
    used_correction: bool
    fallback_reason: str | None
    gradient_norm: float
    update_norm: float
    predicted_quadratic_decrease: float
    cosine_to_baseline: float = 1.0
    relative_update_difference: float = 0.0


@dataclass(frozen=True)
class DirectionStats:
    directions: Mapping[str, Tensor]
    layers: Mapping[str, LayerDirectionStats]
    gradient_norm: float
    update_norm: float
    fallback_parameter_count: int
    measurements: Mapping[str, float] = field(default_factory=dict)


CurvatureClosure = Callable[[nn.Module, Any, Any], CurvatureUpdate | Mapping[str, Any]]
