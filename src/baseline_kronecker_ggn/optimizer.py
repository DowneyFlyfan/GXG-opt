from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn

from kronecker_ggn_common.base import LayerwiseCurvatureOptimizer
from kronecker_ggn_common.config import KroneckerGGNConfig
from kronecker_ggn_common.kronecker_factors import update_factor_ema
from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator
from kronecker_ggn_common.types import CurvatureFactors, LayerDirectionStats

from .preconditioner import baseline_direction
from .state import BaselineLayerState


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


class KroneckerGGN(LayerwiseCurvatureOptimizer):
    def __init__(
        self,
        model: nn.Module,
        config: KroneckerGGNConfig | None = None,
        **config_overrides,
    ) -> None:
        if config is not None and config_overrides:
            raise ValueError(
                "Pass either config or keyword configuration overrides, not both"
            )
        config = config or KroneckerGGNConfig(**config_overrides)
        super().__init__(model, config)
        self.layer_state = {
            layer.layer_id: BaselineLayerState(damping=config.damping)
            for layer in self.registry.supported
        }

    def _new_spectral(self, state: BaselineLayerState) -> KroneckerSpectralOperator:
        assert state.activation is not None and state.output is not None
        return KroneckerSpectralOperator(
            state.activation,
            state.output,
            state.damping,
            eigenvalue_floor=self.config.factor_eigenvalue_floor,
            joint_eigenvalue_floor=self.config.joint_eigenvalue_floor,
            compute_dtype=_dtype(self.config.linear_algebra_dtype),
        )

    def _update_layer_curvature(
        self,
        layer_id: str,
        factors: CurvatureFactors,
        refresh_spectral: bool,
    ) -> tuple[float, float]:
        state = self.layer_state[layer_id]
        factor_started = time.perf_counter()
        activation = factors.activation.detach().to(
            dtype=_dtype(self.config.linear_algebra_dtype)
        )
        output = factors.output.detach().to(
            device=activation.device, dtype=activation.dtype
        )
        layer = self.registry.by_id(layer_id)
        if activation.shape != (layer.matrix_shape[1], layer.matrix_shape[1]):
            raise ValueError(f"Activation factor for {layer_id} has the wrong shape")
        if output.shape != (layer.matrix_shape[0], layer.matrix_shape[0]):
            raise ValueError(f"Output factor for {layer_id} has the wrong shape")
        if not torch.isfinite(activation).all() or not torch.isfinite(output).all():
            raise ValueError(f"Non-finite factors for {layer_id}")
        state.activation = update_factor_ema(
            state.activation, activation, self.config.factor_decay
        )
        state.output = update_factor_ema(state.output, output, self.config.factor_decay)
        state.factor_age = 0
        state.update_count += 1
        factor_seconds = time.perf_counter() - factor_started
        spectral_seconds = 0.0
        if state.spectral is None or refresh_spectral:
            spectral_started = time.perf_counter()
            state.spectral = self._new_spectral(state)
            state.inverse_age = 0
            state.fallback_status = None
            state.diagnostics = {
                "condition_number": state.spectral.condition_number(),
                "factor_sample_count": float(factors.sample_count),
            }
            spectral_seconds = time.perf_counter() - spectral_started
        return factor_seconds, spectral_seconds

    def _curvature_direction(self, layer_id: str, gradient: Tensor):
        state = self.layer_state[layer_id]
        gradient_matrix = gradient.reshape(self.registry.by_id(layer_id).matrix_shape)
        if state.spectral is None:
            reason = state.fallback_status or "spectral_state_unavailable"
            return None, LayerDirectionStats(
                layer_id,
                used_curvature=False,
                used_correction=False,
                fallback_reason=reason,
                gradient_norm=float(gradient_matrix.float().norm().item()),
                update_norm=0.0,
                predicted_quadratic_decrease=0.0,
            )
        try:
            direction = baseline_direction(state.spectral, gradient_matrix)
            if self.config.weight_decay:
                direction = (
                    direction
                    - self.config.weight_decay
                    * self.registry.by_id(layer_id).weight.detach()
                )
            if not torch.isfinite(direction).all():
                raise FloatingPointError("baseline direction is non-finite")
            curvature_action = state.spectral.matvec(direction)
            prediction = -float(
                (gradient_matrix * direction).sum().item()
            ) - 0.5 * float((direction * curvature_action).sum().item())
            return direction.reshape_as(gradient), LayerDirectionStats(
                layer_id,
                used_curvature=True,
                used_correction=False,
                fallback_reason=None,
                gradient_norm=float(gradient_matrix.float().norm().item()),
                update_norm=float(direction.float().norm().item()),
                predicted_quadratic_decrease=prediction,
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            state.fallback_status = f"baseline_numerical_failure:{type(error).__name__}"
            return None, LayerDirectionStats(
                layer_id,
                used_curvature=False,
                used_correction=False,
                fallback_reason=state.fallback_status,
                gradient_norm=float(gradient_matrix.float().norm().item()),
                update_norm=0.0,
                predicted_quadratic_decrease=0.0,
            )

    def baseline_direction_for_layer(self, layer_id: str, gradient: Tensor) -> Tensor:
        state = self.layer_state[layer_id]
        if state.spectral is None:
            raise RuntimeError(f"Layer {layer_id} does not have initialized factors")
        return baseline_direction(
            state.spectral, gradient.reshape(self.registry.by_id(layer_id).matrix_shape)
        )

    def _predicted_reduction(self, directions, step_scale: float) -> float | None:
        named_parameters = dict(self.model.named_parameters())
        direction_by_parameter = {
            id(named_parameters[name]): direction for name, direction in directions.items()
        }
        reduction = 0.0
        used_layer = False
        for layer in self.registry.supported:
            direction = direction_by_parameter.get(id(layer.weight))
            gradient = layer.weight.grad
            spectral = self.layer_state[layer.layer_id].spectral
            if direction is None or gradient is None or spectral is None:
                continue
            direction_matrix = direction.reshape(layer.matrix_shape)
            gradient_matrix = gradient.detach().reshape(layer.matrix_shape)
            curvature_action = spectral.matvec(direction_matrix)
            reduction += -step_scale * float(
                (gradient_matrix * direction_matrix).sum().item()
            ) - 0.5 * step_scale**2 * float(
                (direction_matrix * curvature_action).sum().item()
            )
            used_layer = True
        return reduction if used_layer else None

    def _adapt_damping(self, reduction_ratio: float) -> None:
        if reduction_ratio < 0.25:
            multiplier = self.config.damping_increase
        elif reduction_ratio > 0.75:
            multiplier = self.config.damping_decrease
        else:
            return
        for state in self.layer_state.values():
            state.damping = min(
                max(state.damping * multiplier, self.config.minimum_damping),
                self.config.maximum_damping,
            )
            if state.activation is not None and state.output is not None:
                state.spectral = self._new_spectral(state)

    def _age_curvature_state(self) -> None:
        for state in self.layer_state.values():
            state.factor_age += 1
            state.inverse_age += 1

    def curvature_state_dict(self) -> dict[str, Any]:
        return {
            "curvature_mode": self.config.curvature_mode,
            "registry": self.registry.state_metadata(),
            "layers": {
                layer_id: state.state_dict()
                for layer_id, state in self.layer_state.items()
            },
            "last_curvature_step": self._last_curvature_step,
        }

    def get_metrics(self) -> dict[str, float]:
        metrics = super().get_metrics()
        for layer_id, state in self.layer_state.items():
            metrics[f"layer/{layer_id}/damping"] = state.damping
            metrics[f"layer/{layer_id}/factor_age"] = float(state.factor_age)
            metrics[f"layer/{layer_id}/inverse_age"] = float(state.inverse_age)
            for name, value in state.diagnostics.items():
                metrics[f"layer/{layer_id}/{name}"] = value
        return metrics

    def load_curvature_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("curvature_mode") != self.config.curvature_mode:
            raise ValueError(
                "Curvature checkpoint mode does not match the optimizer configuration"
            )
        saved_layers = state.get("layers", {})
        if set(saved_layers) != set(self.layer_state):
            raise ValueError(
                "Curvature checkpoint layer registry does not match the model"
            )
        for layer_id, saved in saved_layers.items():
            layer = self.registry.by_id(layer_id)
            current = self.layer_state[layer_id]
            device = layer.weight.device
            dtype = _dtype(self.config.linear_algebra_dtype)
            current.activation = (
                None
                if saved["activation"] is None
                else saved["activation"].to(device=device, dtype=dtype)
            )
            current.output = (
                None
                if saved["output"] is None
                else saved["output"].to(device=device, dtype=dtype)
            )
            current.damping = float(saved["damping"])
            current.factor_age = int(saved["factor_age"])
            current.inverse_age = int(saved["inverse_age"])
            current.update_count = int(saved["update_count"])
            current.fallback_status = saved["fallback_status"]
            current.diagnostics = dict(saved["diagnostics"])
            current.spectral = (
                None if saved["spectral"] is None else self._new_spectral(current)
            )
        self._last_curvature_step = int(state.get("last_curvature_step", -1))


def kronecker_ggn(
    model: nn.Module, config: KroneckerGGNConfig | None = None, **kwargs
) -> KroneckerGGN:
    return KroneckerGGN(model, config, **kwargs)
