from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

_CURVATURE_MODES = {"exact_ggn", "mc_ggn", "empirical_fisher"}
_DTYPES = {"float32", "float64"}


@dataclass(frozen=True)
class KroneckerGGNConfig:
    learning_rate: float = 1.0
    curvature_mode: str = "mc_ggn"
    damping: float = 1.0e-3
    damping_mode: str = "exact_joint_eigenvalue"
    factor_decay: float = 0.95
    factor_update_interval: int = 1
    spectral_update_interval: int = 20
    factor_eigenvalue_floor: float = 1.0e-8
    joint_eigenvalue_floor: float = 1.0e-12
    supported_modules: tuple[str, ...] = ("linear",)
    unsupported_parameter_fallback: str = "adamw"
    fallback_learning_rate: float | None = None
    fallback_betas: tuple[float, float] = (0.9, 0.999)
    fallback_epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    linear_algebra_dtype: str = "float32"
    gradient_clip_norm: float | None = None
    trust_clip: float | None = None

    def __post_init__(self) -> None:
        if self.curvature_mode not in _CURVATURE_MODES:
            raise ValueError(
                f"curvature_mode must be one of {sorted(_CURVATURE_MODES)}"
            )
        if self.damping_mode != "exact_joint_eigenvalue":
            raise ValueError("Only exact_joint_eigenvalue damping is implemented")
        if self.linear_algebra_dtype not in _DTYPES:
            raise ValueError(f"linear_algebra_dtype must be one of {sorted(_DTYPES)}")
        if tuple(self.supported_modules) != ("linear",):
            raise ValueError("The MVP supports nn.Linear only")
        if self.unsupported_parameter_fallback not in {"adamw", "sgd"}:
            raise ValueError("unsupported_parameter_fallback must be adamw or sgd")
        if self.learning_rate <= 0 or self.damping <= 0:
            raise ValueError("learning_rate and damping must be positive")
        if not 0 <= self.factor_decay < 1:
            raise ValueError("factor_decay must be in [0, 1)")
        if self.factor_update_interval <= 0 or self.spectral_update_interval <= 0:
            raise ValueError("factor and spectral update intervals must be positive")
        if self.factor_eigenvalue_floor <= 0 or self.joint_eigenvalue_floor <= 0:
            raise ValueError("eigenvalue floors must be positive")
        if len(self.fallback_betas) != 2 or any(
            not 0 <= beta < 1 for beta in self.fallback_betas
        ):
            raise ValueError("fallback_betas must contain two values in [0, 1)")
        if self.fallback_epsilon <= 0 or self.weight_decay < 0:
            raise ValueError(
                "fallback_epsilon must be positive and weight_decay nonnegative"
            )
        if self.fallback_learning_rate is not None and self.fallback_learning_rate <= 0:
            raise ValueError("fallback_learning_rate must be positive when set")
        for name in ("gradient_clip_norm", "trust_clip"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KroneckerGGNConfig:
        value = dict(value)
        value.pop("name", None)
        valid = {field.name for field in fields(cls)}
        unknown = set(value) - valid
        if unknown:
            raise ValueError(f"Unknown optimizer configuration keys: {sorted(unknown)}")
        for name in ("supported_modules", "fallback_betas"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> KroneckerGGNConfig:
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw.get("optimizer", raw))


@dataclass(frozen=True)
class LowRankCorrectedKroneckerGGNConfig(KroneckerGGNConfig):
    correction_rank: int = 4
    correction_oversampling: int = 4
    lanczos_steps: int = 12
    lanczos_tolerance: float = 1.0e-4
    reorthogonalization_passes: int = 2
    correction_warmup_steps: int = 100
    correction_refresh_interval: int = 100
    correction_max_age: int = 200
    correction_eigenvalue_margin: float = 0.1
    correction_abs_eigenvalue_cap: float | None = 100.0
    correction_storage: str = "dense_reference"
    correction_dtype: str = "float32"
    correction_memory_budget_mb: float = 2048.0
    per_layer_correction_memory_budget_mb: float | None = None
    active_layer_policy: str = "largest_relative_residual"
    active_layer_count: int = 2
    cross_batch_validation: bool = False
    cross_batch_reliability_threshold: float = 0.0
    predicted_realized_ratio_threshold: float | None = None
    fallback_on_invalid_correction: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.correction_rank < 0 or self.correction_oversampling < 0:
            raise ValueError("correction rank and oversampling cannot be negative")
        if self.lanczos_steps <= 0 or self.reorthogonalization_passes <= 0:
            raise ValueError(
                "Lanczos steps and reorthogonalization passes must be positive"
            )
        if self.lanczos_tolerance <= 0:
            raise ValueError("lanczos_tolerance must be positive")
        if self.correction_warmup_steps < 0:
            raise ValueError("correction_warmup_steps cannot be negative")
        if self.correction_refresh_interval <= 0 or self.correction_max_age < 0:
            raise ValueError(
                "correction refresh interval must be positive and max age nonnegative"
            )
        if not 0 < self.correction_eigenvalue_margin < 1:
            raise ValueError("correction_eigenvalue_margin must be in (0, 1)")
        if (
            self.correction_abs_eigenvalue_cap is not None
            and self.correction_abs_eigenvalue_cap <= 0
        ):
            raise ValueError("correction_abs_eigenvalue_cap must be positive when set")
        if self.correction_storage not in {"dense_reference", "selected_layers"}:
            raise ValueError(
                "correction_storage must be dense_reference or selected_layers"
            )
        if self.correction_dtype not in _DTYPES:
            raise ValueError(f"correction_dtype must be one of {sorted(_DTYPES)}")
        if self.correction_memory_budget_mb <= 0:
            raise ValueError("correction_memory_budget_mb must be positive")
        if (
            self.per_layer_correction_memory_budget_mb is not None
            and self.per_layer_correction_memory_budget_mb <= 0
        ):
            raise ValueError("per-layer memory budget must be positive when set")
        if self.active_layer_policy not in {
            "largest_relative_residual",
            "largest_parameter_count",
            "fixed",
            "rotating",
        }:
            raise ValueError("unsupported active_layer_policy")
        if self.active_layer_count <= 0:
            raise ValueError("active_layer_count must be positive")
        if not 0 <= self.cross_batch_reliability_threshold <= 1:
            raise ValueError("cross_batch_reliability_threshold must be in [0, 1]")
        if (
            self.predicted_realized_ratio_threshold is not None
            and self.predicted_realized_ratio_threshold < 0
        ):
            raise ValueError("predicted_realized_ratio_threshold cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LowRankCorrectedKroneckerGGNConfig:
        value = dict(value)
        value.pop("name", None)
        valid = {field.name for field in fields(cls)}
        unknown = set(value) - valid
        if unknown:
            raise ValueError(f"Unknown optimizer configuration keys: {sorted(unknown)}")
        for name in ("supported_modules", "fallback_betas"):
            if name in value:
                value[name] = tuple(value[name])
        return cls(**value)
