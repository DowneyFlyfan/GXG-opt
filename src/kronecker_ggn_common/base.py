from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .config import KroneckerGGNConfig
from .diagnostics import device_peak_memory, require_finite
from .layer_registry import LayerRegistry
from .types import (
    CurvatureFactors,
    CurvatureStats,
    CurvatureUpdate,
    DirectionStats,
    LayerDirectionStats,
)

LOGGER = logging.getLogger(__name__)


class LayerwiseCurvatureOptimizer(torch.optim.Optimizer, ABC):
    """Shared scheduling, fallback, serialization, and atomic-step behavior."""

    def __init__(self, model: nn.Module, config: KroneckerGGNConfig) -> None:
        self.model = model
        self.config = config
        parameters = list(model.parameters())
        super().__init__(parameters, {"lr": config.learning_rate})
        self.registry = LayerRegistry(model, config.factor_update_interval)
        self._step_count = 0
        self._last_curvature_step = -1
        self._ggn_operators: Mapping[str, Any] = {}
        self._pending_direction: DirectionStats | None = None
        self._pending_fallback_state: dict[
            nn.Parameter, tuple[Tensor, Tensor, int]
        ] = {}
        self.last_direction_stats: DirectionStats | None = None
        self.last_curvature_stats: CurvatureStats | None = None
        self.last_step_wall_time_seconds = 0.0
        self.fallback_events: list[dict[str, str]] = []
        for name, _parameter, reason in self.registry.fallback_parameters:
            event = {
                "parameter": name,
                "rule": config.unsupported_parameter_fallback,
                "reason": reason,
            }
            self.fallback_events.append(event)
            LOGGER.warning("Kronecker GGN fallback: %s", event)

    def register_model(self, model: nn.Module) -> None:
        if model is not self.model:
            raise ValueError("The optimizer is already registered to a different model")

    @property
    def step_count(self) -> int:
        return self._step_count

    def should_update_curvature(self) -> bool:
        return self._step_count % self.config.factor_update_interval == 0

    def update_curvature(self, curvature_closure, batch=None) -> CurvatureStats:
        started = time.perf_counter()
        if not self.should_update_curvature():
            stats = CurvatureStats(
                curvature_mode=self.config.curvature_mode,
                updated_layers=(),
                skipped_layers={
                    layer.layer_id: "factor_update_interval"
                    for layer in self.registry.supported
                },
                wall_time_seconds=time.perf_counter() - started,
                factor_update_seconds=0.0,
                spectral_update_seconds=0.0,
            )
            self.last_curvature_stats = stats
            return stats
        raw = curvature_closure(self.model, batch, self.registry)
        update = self._coerce_curvature_update(raw)
        if update.curvature_mode != self.config.curvature_mode:
            raise ValueError(
                f"Curvature closure returned {update.curvature_mode!r}, configured mode is {self.config.curvature_mode!r}"
            )
        factor_seconds = spectral_seconds = 0.0
        updated: list[str] = []
        skipped: dict[str, str] = {}
        refresh_spectral = self._step_count % self.config.spectral_update_interval == 0
        for layer in self.registry.supported:
            factors = update.factors.get(layer.layer_id)
            if factors is None:
                skipped[layer.layer_id] = "missing_factor_estimate"
                continue
            if isinstance(factors, tuple):
                factors = CurvatureFactors(factors[0], factors[1])
            factor_time, spectral_time = self._update_layer_curvature(
                layer.layer_id, factors, refresh_spectral
            )
            factor_seconds += factor_time
            spectral_seconds += spectral_time
            updated.append(layer.layer_id)
        self._ggn_operators = dict(update.ggn_operators)
        correction_seconds, extra_measurements = self._after_curvature_update()
        self._last_curvature_step = self._step_count
        measurements = dict(update.measurements)
        measurements.update(extra_measurements)
        stats = CurvatureStats(
            curvature_mode=self.config.curvature_mode,
            updated_layers=tuple(updated),
            skipped_layers=skipped,
            wall_time_seconds=time.perf_counter() - started,
            factor_update_seconds=factor_seconds,
            spectral_update_seconds=spectral_seconds,
            correction_update_seconds=correction_seconds,
            measurements=measurements,
        )
        self.last_curvature_stats = stats
        self._pending_direction = None
        self._pending_fallback_state.clear()
        return stats

    def _coerce_curvature_update(self, value) -> CurvatureUpdate:
        if isinstance(value, CurvatureUpdate):
            return value
        if isinstance(value, Mapping):
            return CurvatureUpdate(self.config.curvature_mode, value)
        raise TypeError(
            "curvature_closure must return CurvatureUpdate or a layer-factor mapping"
        )

    @abstractmethod
    def _update_layer_curvature(
        self,
        layer_id: str,
        factors: CurvatureFactors,
        refresh_spectral: bool,
    ) -> tuple[float, float]: ...

    def _after_curvature_update(self) -> tuple[float, Mapping[str, float]]:
        return 0.0, {}

    @abstractmethod
    def _curvature_direction(
        self, layer_id: str, gradient: Tensor
    ) -> tuple[Tensor | None, LayerDirectionStats]: ...

    def _fallback_direction(self, parameter: nn.Parameter, gradient: Tensor) -> Tensor:
        config = self.config
        fallback_lr = config.fallback_learning_rate or config.learning_rate
        scale = fallback_lr / config.learning_rate
        if config.unsupported_parameter_fallback == "sgd":
            return scale * (-gradient - config.weight_decay * parameter.detach())
        beta1, beta2 = config.fallback_betas
        state = self.state[parameter]
        step = int(state.get("fallback_step", 0)) + 1
        old_first = state.get("fallback_exp_avg")
        old_second = state.get("fallback_exp_avg_sq")
        first = torch.zeros_like(parameter) if old_first is None else old_first * beta1
        second = (
            torch.zeros_like(parameter) if old_second is None else old_second * beta2
        )
        first = first.add(gradient, alpha=1 - beta1)
        second = second.addcmul(gradient, gradient, value=1 - beta2)
        self._pending_fallback_state[parameter] = (first, second, step)
        first_hat = first / (1 - beta1**step)
        second_hat = second / (1 - beta2**step)
        adam_direction = -first_hat / (second_hat.sqrt() + config.fallback_epsilon)
        return scale * (adam_direction - config.weight_decay * parameter.detach())

    @torch.no_grad()
    def compute_direction(self) -> DirectionStats:
        if self._pending_direction is not None:
            return self._pending_direction
        directions: dict[str, Tensor] = {}
        layer_stats: dict[str, LayerDirectionStats] = {}
        curvature_parameter_ids = {
            id(layer.weight): layer for layer in self.registry.supported
        }
        gradient_square = update_square = 0.0
        fallback_count = 0
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            require_finite(gradient, f"gradient {name}")
            gradient_square += float(gradient.float().square().sum().item())
            layer = curvature_parameter_ids.get(id(parameter))
            if layer is not None:
                direction, statistics = self._curvature_direction(
                    layer.layer_id, gradient
                )
                if direction is None:
                    direction = self._fallback_direction(parameter, gradient)
                    fallback_count += parameter.numel()
                layer_stats[layer.layer_id] = statistics
            else:
                direction = self._fallback_direction(parameter, gradient)
                fallback_count += parameter.numel()
            require_finite(direction, f"direction {name}")
            if self.config.trust_clip is not None:
                maximum = self.config.trust_clip * max(
                    float(parameter.detach().float().norm().item()), 1.0e-12
                )
                norm = float(direction.float().norm().item())
                if norm > maximum:
                    direction = direction * (maximum / norm)
            directions[name] = direction
            update_square += float(direction.float().square().sum().item())
        if self.config.gradient_clip_norm is not None:
            norm = math.sqrt(gradient_square)
            if norm > self.config.gradient_clip_norm:
                scale = self.config.gradient_clip_norm / norm
                directions = {
                    name: direction * scale for name, direction in directions.items()
                }
                update_square *= scale * scale
        result = DirectionStats(
            directions=directions,
            layers=layer_stats,
            gradient_norm=math.sqrt(gradient_square),
            update_norm=math.sqrt(update_square),
            fallback_parameter_count=fallback_count,
        )
        self._pending_direction = result
        return result

    @torch.no_grad()
    def step(self, closure=None):
        started = time.perf_counter()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        statistics = self.compute_direction()
        parameters = dict(self.model.named_parameters())
        for parameter, (first, second, step) in self._pending_fallback_state.items():
            state = self.state[parameter]
            state["fallback_exp_avg"] = first
            state["fallback_exp_avg_sq"] = second
            state["fallback_step"] = step
        for name, direction in statistics.directions.items():
            parameters[name].add_(direction, alpha=self.config.learning_rate)
        self._step_count += 1
        self.last_direction_stats = statistics
        self._pending_direction = None
        self._pending_fallback_state.clear()
        self._age_curvature_state()
        self.last_step_wall_time_seconds = time.perf_counter() - started
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._pending_direction = None
        self._pending_fallback_state.clear()
        super().zero_grad(set_to_none=set_to_none)

    def _age_curvature_state(self) -> None:
        pass

    def get_metrics(self) -> dict[str, float]:
        metrics = {
            "step": float(self._step_count),
            "step_wall_time_seconds": self.last_step_wall_time_seconds,
            "fallback_parameter_count": float(
                0
                if self.last_direction_stats is None
                else self.last_direction_stats.fallback_parameter_count
            ),
        }
        metrics.update(device_peak_memory())
        if self.last_direction_stats is not None:
            metrics.update(
                {
                    "gradient_norm": self.last_direction_stats.gradient_norm,
                    "update_norm": self.last_direction_stats.update_norm,
                }
            )
            for layer_id, layer in self.last_direction_stats.layers.items():
                prefix = f"layer/{layer_id}"
                metrics[f"{prefix}/gradient_norm"] = layer.gradient_norm
                metrics[f"{prefix}/update_norm"] = layer.update_norm
                metrics[f"{prefix}/predicted_quadratic_decrease"] = (
                    layer.predicted_quadratic_decrease
                )
                metrics[f"{prefix}/used_correction"] = float(layer.used_correction)
                metrics[f"{prefix}/cosine_to_baseline"] = layer.cosine_to_baseline
                metrics[f"{prefix}/relative_update_difference"] = (
                    layer.relative_update_difference
                )
        if self.last_curvature_stats is not None:
            metrics.update(
                {
                    "curvature_update_wall_time_seconds": self.last_curvature_stats.wall_time_seconds,
                    "factor_update_seconds": self.last_curvature_stats.factor_update_seconds,
                    "eigendecomposition_seconds": self.last_curvature_stats.spectral_update_seconds,
                    "correction_update_seconds": self.last_curvature_stats.correction_update_seconds,
                }
            )
            metrics.update(self.last_curvature_stats.measurements)
        return metrics

    @abstractmethod
    def curvature_state_dict(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_curvature_state_dict(self, state: dict[str, Any]) -> None: ...

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["curvature_state"] = self.curvature_state_dict()
        state["optimizer_step_count"] = self._step_count
        state["curvature_mode"] = self.config.curvature_mode
        state["resolved_config"] = self.config.to_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state_dict = dict(state_dict)
        curvature = state_dict.pop("curvature_state", None)
        step_count = int(state_dict.pop("optimizer_step_count", 0))
        mode = state_dict.pop("curvature_mode", self.config.curvature_mode)
        state_dict.pop("resolved_config", None)
        if mode != self.config.curvature_mode:
            raise ValueError(
                f"Checkpoint curvature_mode {mode!r} does not match {self.config.curvature_mode!r}"
            )
        super().load_state_dict(state_dict)
        self._step_count = step_count
        if curvature is not None:
            self.load_curvature_state_dict(curvature)
