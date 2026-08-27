from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class AdamConfig:
    type: str = "adamw"
    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    shadow_update_during_gn: bool = True
    variance_floor_ratio: float = 1.0


@dataclass(frozen=True)
class GNConfig:
    method: str = "iclr2026_layerwise_gn"
    enabled: bool = True
    layer_partition: str = "auto_transformer"
    inner_optimizer_matrix: str = "muon"
    inner_optimizer_vector: str = "adamw"
    inner_steps: int = 20
    inner_lr: float = 1.0e-3
    inner_weight_decay: float = 0.0
    gradient_batch_multiplier: int = 8
    initial_damping: float = 1.0e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.5
    min_reduction_ratio: float = 0.25
    strong_reduction_ratio: float = 0.75
    line_search_alphas: tuple[float, ...] = (1.0, 0.70710678, 0.5, 0.35355339, 0.25, 0.0)
    mixed_precision_curvature_dtype: str = "float32"
    rejection_persistence: int = 2
    relative_residual_tolerance: float = 1.0e-4
    max_update_norm: float = 10.0


@dataclass(frozen=True)
class BridgeConfig:
    length_adam_steps: int = 50
    rho_start: float = 0.3
    rho_end: float = 0.0
    schedule: str = "cosine"
    per_layer_norm_match: bool = True
    max_update_ratio: float = 1.5
    use_secant_scale: bool = True
    secant_curvature_floor: float = 1.0e-12


@dataclass(frozen=True)
class DutyCycleConfig:
    """Deterministic phase routing for the first rough experiment."""

    gn_epochs: int = 1
    adam_epochs: int = 3
    start_phase: str = "gn"
    bridge_on_gn_to_adam: bool = True


@dataclass(frozen=True)
class FinalMileConfig:
    enabled: bool = False
    metric_name: str = "accuracy"
    metric_mode: str = "max"
    metric_threshold: float | None = None
    eval_split: str = "validation"
    min_metric_delta: float = 1.0e-4
    max_recovery_attempts: int = 2
    max_accepted_gn_steps: int = 10
    patience_evaluations: int = 3
    max_wall_time_seconds: float | None = None
    damping_multiplier: float = 10.0
    step_scale_multiplier: float = 0.5
    update_norm_cap_multiplier: float = 0.5
    restore_best_on_failure: bool = True


@dataclass(frozen=True)
class CheckpointConfig:
    save_on_phase_change: bool = True
    save_best_validation: bool = True
    atomic: bool = True


T = TypeVar("T")


def _number(value: Any, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} must be numeric")
    return float(value)


def _integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be boolean")
    return value


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
    for name in ("betas", "line_search_alphas"):
        if name in converted:
            converted[name] = tuple(converted[name])
    return cls(**converted)


@dataclass(frozen=True)
class GXGConfig:
    name: str = "gxg_optimizer"
    seed: int = 42
    adam: AdamConfig = AdamConfig()
    gn: GNConfig = GNConfig()
    bridge: BridgeConfig = BridgeConfig()
    duty_cycle: DutyCycleConfig = DutyCycleConfig()
    final_mile: FinalMileConfig = FinalMileConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()

    def __post_init__(self) -> None:
        nested_types = {
            "adam": AdamConfig,
            "gn": GNConfig,
            "bridge": BridgeConfig,
            "duty_cycle": DutyCycleConfig,
            "final_mile": FinalMileConfig,
            "checkpoint": CheckpointConfig,
        }
        for name, expected in nested_types.items():
            if not isinstance(getattr(self, name), expected):
                raise ValueError(f"{name} must be a {expected.__name__}")
        if not isinstance(self.name, str):
            raise ValueError("optimizer name must be a string")
        for path, value in {
            "adam.type": self.adam.type,
            "gn.method": self.gn.method,
            "gn.layer_partition": self.gn.layer_partition,
            "gn.inner_optimizer_matrix": self.gn.inner_optimizer_matrix,
            "gn.inner_optimizer_vector": self.gn.inner_optimizer_vector,
            "gn.mixed_precision_curvature_dtype": self.gn.mixed_precision_curvature_dtype,
            "bridge.schedule": self.bridge.schedule,
            "duty_cycle.start_phase": self.duty_cycle.start_phase,
            "final_mile.metric_name": self.final_mile.metric_name,
            "final_mile.metric_mode": self.final_mile.metric_mode,
            "final_mile.eval_split": self.final_mile.eval_split,
        }.items():
            if not isinstance(value, str):
                raise ValueError(f"{path} must be a string")
        _integer(self.seed, "seed")
        for path, value in {
            "adam.shadow_update_during_gn": self.adam.shadow_update_during_gn,
            "gn.enabled": self.gn.enabled,
            "bridge.per_layer_norm_match": self.bridge.per_layer_norm_match,
            "bridge.use_secant_scale": self.bridge.use_secant_scale,
            "duty_cycle.bridge_on_gn_to_adam": self.duty_cycle.bridge_on_gn_to_adam,
            "final_mile.enabled": self.final_mile.enabled,
            "final_mile.restore_best_on_failure": self.final_mile.restore_best_on_failure,
            "checkpoint.save_on_phase_change": self.checkpoint.save_on_phase_change,
            "checkpoint.save_best_validation": self.checkpoint.save_best_validation,
            "checkpoint.atomic": self.checkpoint.atomic,
        }.items():
            _boolean(value, path)
        for path, value in {
            "gn.inner_steps": self.gn.inner_steps,
            "gn.gradient_batch_multiplier": self.gn.gradient_batch_multiplier,
            "gn.rejection_persistence": self.gn.rejection_persistence,
            "bridge.length_adam_steps": self.bridge.length_adam_steps,
            "duty_cycle.gn_epochs": self.duty_cycle.gn_epochs,
            "duty_cycle.adam_epochs": self.duty_cycle.adam_epochs,
            "final_mile.max_recovery_attempts": self.final_mile.max_recovery_attempts,
            "final_mile.max_accepted_gn_steps": self.final_mile.max_accepted_gn_steps,
            "final_mile.patience_evaluations": self.final_mile.patience_evaluations,
        }.items():
            if _integer(value, path) <= 0:
                raise ValueError(f"{path} must be positive")
        for path, value in {
            "adam.lr": self.adam.lr,
            "adam.eps": self.adam.eps,
            "adam.weight_decay": self.adam.weight_decay,
            "adam.variance_floor_ratio": self.adam.variance_floor_ratio,
            "gn.inner_lr": self.gn.inner_lr,
            "gn.inner_weight_decay": self.gn.inner_weight_decay,
            "gn.initial_damping": self.gn.initial_damping,
            "gn.damping_increase": self.gn.damping_increase,
            "gn.damping_decrease": self.gn.damping_decrease,
            "gn.min_reduction_ratio": self.gn.min_reduction_ratio,
            "gn.strong_reduction_ratio": self.gn.strong_reduction_ratio,
            "gn.relative_residual_tolerance": self.gn.relative_residual_tolerance,
            "gn.max_update_norm": self.gn.max_update_norm,
            "bridge.rho_start": self.bridge.rho_start,
            "bridge.rho_end": self.bridge.rho_end,
            "bridge.max_update_ratio": self.bridge.max_update_ratio,
            "bridge.secant_curvature_floor": self.bridge.secant_curvature_floor,
            "final_mile.min_metric_delta": self.final_mile.min_metric_delta,
            "final_mile.damping_multiplier": self.final_mile.damping_multiplier,
            "final_mile.step_scale_multiplier": self.final_mile.step_scale_multiplier,
            "final_mile.update_norm_cap_multiplier": self.final_mile.update_norm_cap_multiplier,
        }.items():
            _number(value, path)
        for path, value in {
            "final_mile.max_wall_time_seconds": self.final_mile.max_wall_time_seconds,
            "final_mile.metric_threshold": self.final_mile.metric_threshold,
        }.items():
            if value is not None:
                _number(value, path)
        if not isinstance(self.adam.betas, tuple) or len(self.adam.betas) != 2:
            raise ValueError("Adam betas must be a two-item tuple")
        for beta in self.adam.betas:
            _number(beta, "adam.betas")
        if not isinstance(self.gn.line_search_alphas, tuple) or not self.gn.line_search_alphas:
            raise ValueError("GN line-search alphas must be a non-empty tuple")
        for alpha in self.gn.line_search_alphas:
            _number(alpha, "gn.line_search_alphas")
        if self.name != "gxg_optimizer":
            raise ValueError("optimizer name must be exactly 'gxg_optimizer'")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.adam.type != "adamw":
            raise ValueError("Only AdamW is supported by the Adam state bank")
        if self.adam.lr <= 0 or self.adam.eps <= 0:
            raise ValueError("Adam learning rate and epsilon must be positive")
        if any(not 0 <= beta < 1 for beta in self.adam.betas):
            raise ValueError("Adam betas must contain two values in [0, 1)")
        if self.adam.weight_decay < 0 or self.adam.variance_floor_ratio < 0:
            raise ValueError("Adam weight decay and variance floor ratio cannot be negative")
        if self.gn.method != "iclr2026_layerwise_gn":
            raise ValueError("GN method must be 'iclr2026_layerwise_gn'")
        if self.gn.layer_partition not in {"auto_transformer", "explicit", "regex"}:
            raise ValueError("gn.layer_partition must be auto_transformer, explicit, or regex")
        if self.gn.inner_optimizer_matrix not in {"muon", "adamw", "cg"}:
            raise ValueError("Unsupported matrix inner solver")
        if self.gn.inner_optimizer_vector not in {"adamw", "cg"}:
            raise ValueError("Unsupported vector inner solver")
        if (self.gn.inner_optimizer_matrix == "cg") != (self.gn.inner_optimizer_vector == "cg"):
            raise ValueError("Truncated CG must be selected for both matrix and vector parameters")
        if self.gn.inner_steps <= 0 or self.gn.inner_lr <= 0:
            raise ValueError("GN inner steps and learning rate must be positive")
        if self.gn.gradient_batch_multiplier <= 0 or self.gn.rejection_persistence <= 0:
            raise ValueError("GN batch multiplier and rejection persistence must be positive")
        if self.gn.inner_weight_decay < 0 or self.gn.min_reduction_ratio < 0 or self.gn.strong_reduction_ratio < 0:
            raise ValueError("GN regularization and reduction ratios cannot be negative")
        if self.gn.relative_residual_tolerance <= 0 or self.gn.max_update_norm <= 0:
            raise ValueError("GN tolerance and update norm must be positive")
        if self.gn.initial_damping <= 0 or self.gn.damping_increase <= 1 or not 0 < self.gn.damping_decrease <= 1:
            raise ValueError("GN damping values are invalid")
        if 0.0 not in self.gn.line_search_alphas:
            raise ValueError("GN line search must include the 0.0 no-op candidate")
        if any(alpha < 0 for alpha in self.gn.line_search_alphas):
            raise ValueError("GN line-search alphas cannot be negative")
        if self.gn.mixed_precision_curvature_dtype != "float32":
            raise ValueError("Curvature accumulation must use float32")
        if not 0 <= self.bridge.rho_start <= 1 or not 0 <= self.bridge.rho_end <= 1:
            raise ValueError("Bridge rho values must be in [0, 1]")
        if self.bridge.rho_end > self.bridge.rho_start:
            raise ValueError("Bridge rho_end cannot exceed rho_start")
        if self.bridge.schedule not in {"linear", "cosine"} or self.bridge.length_adam_steps <= 0:
            raise ValueError("Invalid bridge schedule or length")
        if self.bridge.max_update_ratio <= 0 or self.bridge.secant_curvature_floor < 0:
            raise ValueError("Invalid bridge norm ratio or secant floor")
        if self.duty_cycle.gn_epochs <= 0 or self.duty_cycle.adam_epochs <= 0:
            raise ValueError("Fixed duty-cycle GN and Adam epoch counts must be positive")
        if self.duty_cycle.start_phase not in {"gn", "adam"}:
            raise ValueError("duty_cycle.start_phase must be 'gn' or 'adam'")
        if self.final_mile.metric_mode not in {"max", "min"}:
            raise ValueError("final_mile.metric_mode must be 'max' or 'min'")
        if self.final_mile.min_metric_delta < 0:
            raise ValueError("final_mile.min_metric_delta cannot be negative")
        if min(
            self.final_mile.damping_multiplier,
            self.final_mile.step_scale_multiplier,
            self.final_mile.update_norm_cap_multiplier,
        ) <= 0:
            raise ValueError("Final-mile damping, step, and norm multipliers must be positive")
        if self.final_mile.max_wall_time_seconds is not None and self.final_mile.max_wall_time_seconds <= 0:
            raise ValueError("final_mile.max_wall_time_seconds must be positive when set")
        if self.final_mile.eval_split.lower() == "test":
            raise ValueError("The test split cannot control final-mile decisions")
        if self.final_mile.enabled and self.final_mile.metric_threshold is None:
            raise ValueError("Enabled final-mile recovery requires metric_threshold")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GXGConfig":
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
        values = dict(raw)
        classes = {
            "adam": AdamConfig,
            "gn": GNConfig,
            "bridge": BridgeConfig,
            "duty_cycle": DutyCycleConfig,
            "final_mile": FinalMileConfig,
            "checkpoint": CheckpointConfig,
        }
        for name, nested_cls in classes.items():
            if name in values:
                values[name] = _nested(nested_cls, values[name], name)
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "GXGConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GXGConfig":
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("PyYAML is required to load GXG YAML configuration") from error
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def to_dict(self, *, wrapped: bool = False) -> dict[str, Any]:
        value = asdict(self)
        return {"optimizer": value} if wrapped else value
