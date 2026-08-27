from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.0


@dataclass(frozen=True)
class GNConfig:
    enabled: bool = True
    warmup_steps: int = 100
    rank: int = 8
    refresh_interval: int = 200
    curvature_dtype: str = "float32"
    curvature_batches: int = 1
    acceptance_batches: int = 1
    initial_damping: float = 1.0e-2
    min_damping: float = 1.0e-6
    max_damping: float = 1.0e4
    damping_increase: float = 10.0
    damping_decrease: float = 0.5
    trust_radius: float = 1.0
    max_relative_block_update: float = 1.0e-3
    alpha_max: float = 1.0
    max_basis_age: int = 200
    max_parameter_drift: float = 1.0e-2
    rho_min: float = 0.25
    acceptance_margin: float = 0.0
    momentum_subspace_decay: float = 0.0
    fallback_to_adamw: bool = True
    min_block_numel: int = 256
    include_output_projection: bool = False
    guided_block_patterns: tuple[str, ...] = ()
    reorthogonalization_passes: int = 2
    negative_eigenvalue_tolerance: float = 1.0e-6
    failures_before_cooldown: int = 3
    failure_cooldown_steps: int = 400


@dataclass(frozen=True)
class OracleConfig:
    enabled: bool = False
    damping: float = 1.0e-2
    max_iterations: int = 100
    relative_tolerance: float = 1.0e-8


@dataclass(frozen=True)
class AdaptiveConfig:
    enabled: bool = False
    min_refresh_interval: int = 50
    max_refresh_interval: int = 500
    safety_margin: float = 0.05


@dataclass(frozen=True)
class FixedEpochDutyCycleConfig:
    """Deterministic epoch switch used before any adaptive scheduling."""

    enabled: bool = False
    start_epoch: int = 0
    on_epochs: int = 1
    off_epochs: int = 1
    refresh_on_activation: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    atomic: bool = True


T = TypeVar("T")


def _nested(cls: type[T], value: Any, path: str) -> T:
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    valid = {item.name for item in fields(cls)}
    unknown = set(value) - valid
    if unknown:
        raise ValueError(f"Unknown {path} keys: {sorted(unknown)}")
    converted = dict(value)
    for name in ("betas", "guided_block_patterns"):
        if name in converted:
            converted[name] = tuple(converted[name])
    return cls(**converted)


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    return value


@dataclass(frozen=True)
class GuidedAdamConfig:
    name: str = "gn_guided_adamw"
    seed: int = 42
    adamw: AdamWConfig = AdamWConfig()
    gn: GNConfig = GNConfig()
    oracle: OracleConfig = OracleConfig()
    fixed_epoch_duty_cycle: FixedEpochDutyCycleConfig = FixedEpochDutyCycleConfig()
    adaptive: AdaptiveConfig = AdaptiveConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    def __post_init__(self) -> None:
        expected = {
            "adamw": AdamWConfig,
            "gn": GNConfig,
            "oracle": OracleConfig,
            "fixed_epoch_duty_cycle": FixedEpochDutyCycleConfig,
            "adaptive": AdaptiveConfig,
            "checkpoint": CheckpointConfig,
        }
        for name, cls in expected.items():
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be a {cls.__name__}")
        if self.name != "gn_guided_adamw":
            raise ValueError("optimizer name must be exactly 'gn_guided_adamw'")
        if _integer(self.seed, "seed") < 0:
            raise ValueError("seed cannot be negative")
        for path, value in {
            "gn.enabled": self.gn.enabled,
            "gn.fallback_to_adamw": self.gn.fallback_to_adamw,
            "gn.include_output_projection": self.gn.include_output_projection,
            "oracle.enabled": self.oracle.enabled,
            "fixed_epoch_duty_cycle.enabled": self.fixed_epoch_duty_cycle.enabled,
            "fixed_epoch_duty_cycle.refresh_on_activation": self.fixed_epoch_duty_cycle.refresh_on_activation,
            "adaptive.enabled": self.adaptive.enabled,
            "checkpoint.atomic": self.checkpoint.atomic,
        }.items():
            if not isinstance(value, bool):
                raise ValueError(f"{path} must be boolean")
        if self.adaptive.enabled:
            raise ValueError("Adaptive refresh is deferred until the fixed-frequency MVP passes Gate C")
        if self.gn.curvature_dtype != "float32":
            raise ValueError("The fixed prototype requires FP32 curvature")
        if self.gn.curvature_batches != 1 or self.gn.acceptance_batches != 1:
            raise ValueError("The fixed prototype currently requires exactly one curvature and acceptance batch")
        if not isinstance(self.adamw.betas, tuple) or len(self.adamw.betas) != 2:
            raise ValueError("adamw.betas must be a two-item tuple")
        for beta in self.adamw.betas:
            if not 0 <= _number(beta, "adamw.betas") < 1:
                raise ValueError("adamw.betas values must be in [0, 1)")
        for path, value in {
            "gn.warmup_steps": self.gn.warmup_steps,
            "gn.rank": self.gn.rank,
            "gn.refresh_interval": self.gn.refresh_interval,
            "gn.curvature_batches": self.gn.curvature_batches,
            "gn.acceptance_batches": self.gn.acceptance_batches,
            "gn.max_basis_age": self.gn.max_basis_age,
            "gn.min_block_numel": self.gn.min_block_numel,
            "gn.reorthogonalization_passes": self.gn.reorthogonalization_passes,
            "gn.failures_before_cooldown": self.gn.failures_before_cooldown,
            "gn.failure_cooldown_steps": self.gn.failure_cooldown_steps,
            "oracle.max_iterations": self.oracle.max_iterations,
            "fixed_epoch_duty_cycle.start_epoch": self.fixed_epoch_duty_cycle.start_epoch,
            "fixed_epoch_duty_cycle.on_epochs": self.fixed_epoch_duty_cycle.on_epochs,
            "fixed_epoch_duty_cycle.off_epochs": self.fixed_epoch_duty_cycle.off_epochs,
            "adaptive.min_refresh_interval": self.adaptive.min_refresh_interval,
            "adaptive.max_refresh_interval": self.adaptive.max_refresh_interval,
        }.items():
            integer = _integer(value, path)
            if path in {"gn.warmup_steps", "fixed_epoch_duty_cycle.start_epoch"}:
                if integer < 0:
                    raise ValueError(f"{path} cannot be negative")
            elif integer <= 0:
                raise ValueError(f"{path} must be positive")
        for path, value in {
            "adamw.lr": self.adamw.lr,
            "adamw.eps": self.adamw.eps,
            "gn.initial_damping": self.gn.initial_damping,
            "gn.min_damping": self.gn.min_damping,
            "gn.max_damping": self.gn.max_damping,
            "gn.damping_increase": self.gn.damping_increase,
            "gn.damping_decrease": self.gn.damping_decrease,
            "gn.trust_radius": self.gn.trust_radius,
            "gn.max_relative_block_update": self.gn.max_relative_block_update,
            "gn.alpha_max": self.gn.alpha_max,
            "oracle.damping": self.oracle.damping,
            "oracle.relative_tolerance": self.oracle.relative_tolerance,
        }.items():
            if _number(value, path) <= 0:
                raise ValueError(f"{path} must be positive")
        for path, value in {
            "adamw.weight_decay": self.adamw.weight_decay,
            "gn.max_parameter_drift": self.gn.max_parameter_drift,
            "gn.rho_min": self.gn.rho_min,
            "gn.acceptance_margin": self.gn.acceptance_margin,
            "gn.negative_eigenvalue_tolerance": self.gn.negative_eigenvalue_tolerance,
            "adaptive.safety_margin": self.adaptive.safety_margin,
        }.items():
            if _number(value, path) < 0:
                raise ValueError(f"{path} cannot be negative")
        if not 0 <= _number(self.gn.momentum_subspace_decay, "gn.momentum_subspace_decay") <= 1:
            raise ValueError("gn.momentum_subspace_decay must be in [0, 1]")
        if self.gn.min_damping > self.gn.initial_damping or self.gn.initial_damping > self.gn.max_damping:
            raise ValueError("GN damping must satisfy min <= initial <= max")
        if self.gn.damping_increase <= 1 or not 0 < self.gn.damping_decrease <= 1:
            raise ValueError("GN damping increase/decrease factors are invalid")
        if self.adaptive.min_refresh_interval > self.adaptive.max_refresh_interval:
            raise ValueError("adaptive interval bounds are invalid")
        if not isinstance(self.gn.guided_block_patterns, tuple) or any(
            not isinstance(pattern, str) for pattern in self.gn.guided_block_patterns
        ):
            raise ValueError("gn.guided_block_patterns must be a tuple of strings")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GuidedAdamConfig":
        if "optimizer" in raw:
            if set(raw) != {"optimizer"}:
                raise ValueError("Top-level config may only contain 'optimizer'")
            raw = raw["optimizer"]
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be a mapping")
        valid = {item.name for item in fields(cls)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"Unknown optimizer keys: {sorted(unknown)}")
        value = dict(raw)
        nested = {
            "adamw": AdamWConfig,
            "gn": GNConfig,
            "oracle": OracleConfig,
            "fixed_epoch_duty_cycle": FixedEpochDutyCycleConfig,
            "adaptive": AdaptiveConfig,
            "checkpoint": CheckpointConfig,
        }
        for name, nested_cls in nested.items():
            if name in value:
                value[name] = _nested(nested_cls, value[name], name)
        return cls(**value)

    @classmethod
    def from_json(cls, path: str | Path) -> "GuidedAdamConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GuidedAdamConfig":
        try:
            import yaml
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load GN-guided AdamW YAML") from error
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_dict(self, *, wrapped: bool = False) -> dict[str, Any]:
        value = asdict(self)
        return {"optimizer": value} if wrapped else value
