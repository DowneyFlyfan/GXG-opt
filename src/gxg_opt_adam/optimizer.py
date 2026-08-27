from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .adam_backend import AdamBackend
from .bridge import blend_candidates, bridge_weight, secant_scale
from .checkpointing import (
    build_checkpoint_payload,
    load_rng_state_dict,
    restore_checkpoint_payload,
    rng_state_dict,
    save_atomic_checkpoint,
)
from .config import GXGConfig
from .controller import GXGController
from .distributed_layers import DistributedLayerAdapter
from .layer_partition import LayerGroup, LayerPartitioner
from .layerwise_gn import LayerwiseGN
from .probes import gain_per_second, norm, transfer_ratio
from .state import GXGPhase
from .types import EvalContext, QualityResult, StepContext, StepResult


MetricFunction = Callable[[EvalContext], float | tuple[float, float | None] | Mapping[str, float | None]]


def _mean_gradients(gradients: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not gradients:
        return {}
    names = set(gradients[0])
    if any(set(value) != names for value in gradients):
        raise ValueError("All microbatch gradients must cover the same parameters")
    return {name: sum(value[name].float() for value in gradients) / len(gradients) for name in names}


class GXGOptimizer:
    """Serializable controller alternating layer-wise GN and an independent AdamW bank."""

    STATE_VERSION = 1

    def __init__(
        self,
        model: nn.Module,
        config: GXGConfig | Mapping[str, Any] | str | Path,
        layer_partition: tuple[LayerGroup, ...] | Mapping[str, Sequence[str]] | None = None,
        metric_fn: MetricFunction | None = None,
        distributed_adapter: DistributedLayerAdapter | None = None,
    ) -> None:
        self.model = model
        self.config = self._load_config(config)
        partitioner = LayerPartitioner(model)
        if layer_partition is None:
            if self.config.gn.layer_partition != "auto_transformer":
                raise ValueError("An explicit mapping is required for the configured layer partition mode")
            self.layer_groups = partitioner.automatic_transformer()
        elif isinstance(layer_partition, Mapping):
            if self.config.gn.layer_partition == "regex":
                self.layer_groups = partitioner.from_regex_rules(layer_partition)
            else:
                self.layer_groups = partitioner.from_explicit(layer_partition)
        else:
            self.layer_groups = tuple(layer_partition)
            supplied = {group.name: group.parameter_names for group in self.layer_groups}
            self.layer_groups = partitioner.from_explicit(supplied)
        self.metric_fn = metric_fn
        self.adam = AdamBackend(model, self.layer_groups, self.config.adam)
        self.gn = LayerwiseGN(model, self.layer_groups, self.config.gn, distributed_adapter)
        self.controller = GXGController(self.config)
        self._latest_gn_direction: dict[str, torch.Tensor] = {}
        self._pre_gn_parameters: dict[str, torch.Tensor] = {}
        self._pre_gn_gradient: dict[str, torch.Tensor] = {}
        self._best_model_state: dict[str, torch.Tensor] | None = None
        self._metrics: dict[str, float] = {}
        self._last_switch_reason: str | None = None

    @staticmethod
    def _load_config(config: GXGConfig | Mapping[str, Any] | str | Path) -> GXGConfig:
        if isinstance(config, GXGConfig):
            return config
        if isinstance(config, Mapping):
            return GXGConfig.from_dict(dict(config))
        path = Path(config)
        return GXGConfig.from_json(path) if path.suffix.lower() == ".json" else GXGConfig.from_yaml(path)

    def step(self, step_context: StepContext) -> StepResult:
        started = time.perf_counter()
        phase = self.get_phase()
        if phase in {GXGPhase.DONE, GXGPhase.FINAL_QUALITY_CHECK}:
            return self._result(False, started)
        route_reason = self.controller.route_epoch(step_context.epoch, step_context.nominal_budget_exhausted)
        phase = self.get_phase()
        if phase == GXGPhase.FINAL_QUALITY_CHECK:
            return self._result(False, started, reason=route_reason)
        if route_reason == "fixed_duty_cycle_enter_gn":
            self.adam.capture_variance_floor()
            self._pre_gn_parameters = {
                name: parameter.detach().float().clone() for name, parameter in self.adam.parameters.items()
            }
            if step_context.reference_gradient is not None:
                self._pre_gn_gradient = {
                    name: value.detach().float().clone() for name, value in step_context.reference_gradient.items()
                }
        self.controller.begin_step()
        if phase == GXGPhase.ADAM:
            result = self._step_adam(step_context, started)
        elif phase == GXGPhase.BRIDGE_TO_ADAM:
            result = self._step_bridge(step_context, started)
        else:
            result = self._step_gn(step_context, started)
        if result.switch_reason is None and route_reason is not None:
            result = replace(result, switch_reason=route_reason)
        self._last_switch_reason = result.switch_reason
        return result

    def _step_adam(self, context: StepContext, started: float) -> StepResult:
        candidate, update_norm = self.adam.step()
        self._metrics["adam_update_norm"] = update_norm
        return self._result(True, started, adam_norm=update_norm)

    def _step_bridge(self, context: StepContext, started: float) -> StepResult:
        if context.reference_gradient is None:
            raise ValueError("Bridge steps require a larger reference-batch gradient")
        gradients = _mean_gradients(context.shadow_microbatch_gradients) or self.adam.gradients()
        self.adam.update_moments(gradients, context.shadow_microbatch_gradients, context.adam_equivalent_batches)
        adam_candidate = self.adam.candidate()
        scale = None
        if self.config.bridge.use_secant_scale and self._pre_gn_parameters and self._pre_gn_gradient:
            current = {name: parameter.detach().float() for name, parameter in self.adam.parameters.items()}
            parameter_step = {name: current[name] - self._pre_gn_parameters[name] for name in current}
            gradient_change = {name: context.reference_gradient[name].float() - self._pre_gn_gradient[name] for name in current}
            scale = secant_scale(parameter_step, gradient_change, self.config.bridge.secant_curvature_floor)
        recent = {group.name: self.adam.recent_norm(group.name) for group in self.layer_groups}
        blended = blend_candidates(
            self._latest_gn_direction,
            adam_candidate,
            self.layer_groups,
            context.reference_gradient,
            self.config.bridge,
            self.controller.state.phase_step - 1,
            recent,
            scale=scale,
        )
        update_norm = self.adam.apply(blended.candidate) if blended.accepted else 0.0
        scheduled_rho = bridge_weight(self.config.bridge, self.controller.state.phase_step - 1)
        complete = scheduled_rho <= self.config.bridge.rho_end
        fallback = blended.reason in {"gn_descent_guard", "non_descent"} or (
            blended.reason == "adam_only" and not complete
        )
        reason = None
        if fallback or complete:
            reason = self.controller.bridge_complete(fallback)
        self._metrics.update({"bridge_rho": blended.rho, "bridge_update_norm": update_norm})
        return self._result(
            blended.accepted,
            started,
            adam_norm=norm(adam_candidate),
            bridge_norm=update_norm,
            reason=reason or blended.reason,
        )

    def _step_gn(self, context: StepContext, started: float) -> StepResult:
        if context.gn_batch is None or context.reference_loss_closure is None:
            raise ValueError("GN phases require a functional GN batch and independent reference-loss closure")
        if (
            self.get_phase() == GXGPhase.FINAL_GN_RECOVERY
            and self.controller.state.final_recovery_accepted_steps >= self.config.final_mile.max_accepted_gn_steps
        ):
            raise RuntimeError("Evaluate validation quality before another final-recovery GN step")
        if self.config.adam.shadow_update_during_gn:
            gradients = _mean_gradients(context.shadow_microbatch_gradients) or self.adam.gradients()
            self.adam.update_moments(gradients, context.shadow_microbatch_gradients, context.adam_equivalent_batches)
        phase = self.get_phase()
        recovery = phase == GXGPhase.FINAL_GN_RECOVERY
        gn_result = self.gn.step(
            context.gn_batch,
            context.reference_loss_closure,
            context.reference_batch_id,
            line_search_scale=self.config.final_mile.step_scale_multiplier if recovery else 1.0,
            update_norm_multiplier=self.config.final_mile.update_norm_cap_multiplier if recovery else 1.0,
            weight_decay=self.adam.weight_decay,
        )
        seconds = context.elapsed_seconds or max(time.perf_counter() - started, 1.0e-12)
        predicted = gn_result.line_search.predicted_reduction
        actual = gn_result.line_search.actual_reduction
        transfer = transfer_ratio(actual, predicted) if predicted else (1.0 if actual > 0 else 0.0)
        gain = gain_per_second(actual, seconds)
        reason = self.controller.after_gn(
            accepted=gn_result.accepted,
            reduction_ratio=gn_result.line_search.reduction_ratio,
            transfer_ratio=transfer,
            finite=gn_result.finite,
            update_norm_safe=gn_result.proposal_update_norm <= self.config.gn.max_update_norm * (
                self.config.final_mile.update_norm_cap_multiplier if recovery else 1.0
            ),
            data_exhausted=context.data_exhausted,
        )
        accepted = gn_result.accepted
        if accepted:
            self._latest_gn_direction = {
                name: value.detach().float().clone() for name, value in gn_result.direction.items()
            }
        if self.get_phase() == GXGPhase.DONE and self.config.final_mile.restore_best_on_failure:
            self._restore_best_model()
        measurements = dict(gn_result.diagnostics)
        measurements.update({"gn_transfer_ratio": transfer, "gn_gain_per_second": gain})
        self._metrics.update(measurements)
        return self._result(accepted, started, gn_norm=gn_result.update_norm if accepted else 0.0, reason=reason, measurements=measurements)

    def evaluate_quality(self, eval_context: EvalContext) -> QualityResult:
        if eval_context.split.lower() == "test":
            raise ValueError("The test split is final-reporting-only and cannot control GXG")
        if eval_context.split.lower() != self.config.final_mile.eval_split.lower():
            raise ValueError("Quality control must use final_mile.eval_split")
        metric, loss = self._evaluate_metric(eval_context)
        state = self.controller.state
        improved = self._quality_improved(metric, loss)
        if improved:
            state.best_validation_metric = metric
            state.best_validation_loss = loss
            state.best_checkpoint_id = eval_context.checkpoint_id
            self._best_model_state = {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}
            state.final_patience = 0
        elif state.phase == GXGPhase.FINAL_GN_RECOVERY:
            state.final_patience += 1
        threshold = self.config.final_mile.metric_threshold
        target_met = threshold is not None and (
            metric >= threshold if self.config.final_mile.metric_mode == "max" else metric <= threshold
        )
        state.quality_target_met = target_met
        if state.phase == GXGPhase.FINAL_QUALITY_CHECK:
            if target_met:
                self.controller.transition(GXGPhase.DONE, "quality_target_met")
            elif self.config.final_mile.enabled and state.final_recovery_attempts < self.config.final_mile.max_recovery_attempts:
                self._restore_best_model()
                state.final_recovery_attempts += 1
                state.final_recovery_accepted_steps = 0
                state.final_patience = 0
                self.gn.reset_linearization(self.config.final_mile.damping_multiplier)
                self.controller.transition(GXGPhase.FINAL_GN_RECOVERY, "quality_below_threshold")
            else:
                self.controller.transition(GXGPhase.DONE, "quality_check_complete")
        elif state.phase == GXGPhase.FINAL_GN_RECOVERY:
            if target_met:
                self.controller.transition(GXGPhase.DONE, "quality_target_met_during_recovery")
            elif (
                state.final_patience >= self.config.final_mile.patience_evaluations
                or state.final_recovery_accepted_steps >= self.config.final_mile.max_accepted_gn_steps
            ):
                if state.final_recovery_attempts < self.config.final_mile.max_recovery_attempts:
                    self._restart_recovery("final_recovery_attempt_retry")
                else:
                    self._restore_best_model()
                    self.controller.transition(GXGPhase.DONE, "final_recovery_patience_exhausted")
        if state.phase == GXGPhase.DONE and self.config.final_mile.restore_best_on_failure:
            self._restore_best_model()
        self._metrics.update(
            {
                "validation_metric": metric,
                "best_validation_metric": state.best_validation_metric if state.best_validation_metric is not None else metric,
                "quality_threshold_gap": 0.0
                if threshold is None
                else (threshold - metric if self.config.final_mile.metric_mode == "max" else metric - threshold),
                "quality_target_met": float(target_met),
                "final_recovery_attempt": float(state.final_recovery_attempts),
                "final_patience": float(state.final_patience),
            }
        )
        return QualityResult(metric, loss, state.phase, improved, target_met, state.best_checkpoint_id)

    def _evaluate_metric(self, context: EvalContext) -> tuple[float, float | None]:
        if context.metric is not None:
            return float(context.metric), None if context.loss is None else float(context.loss)
        if self.metric_fn is None:
            raise ValueError("EvalContext.metric or metric_fn is required")
        result = self.metric_fn(context)
        if isinstance(result, Mapping):
            return float(result["metric"]), None if result.get("loss") is None else float(result["loss"])
        if isinstance(result, tuple):
            return float(result[0]), None if result[1] is None else float(result[1])
        return float(result), None

    def _quality_improved(self, metric: float, loss: float | None) -> bool:
        state, config = self.controller.state, self.config.final_mile
        if state.best_validation_metric is None:
            return True
        delta = metric - state.best_validation_metric
        if config.metric_mode == "min":
            delta = -delta
        if delta >= config.min_metric_delta:
            return True
        tied = abs(delta) < config.min_metric_delta
        return tied and loss is not None and (state.best_validation_loss is None or loss < state.best_validation_loss)

    @torch.no_grad()
    def _restore_best_model(self) -> None:
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)

    def _restart_recovery(self, reason: str) -> None:
        state = self.controller.state
        previous = state.phase
        self._restore_best_model()
        state.final_recovery_attempts += 1
        state.final_recovery_accepted_steps = 0
        state.final_patience = 0
        self.gn.reset_linearization(self.config.final_mile.damping_multiplier)
        if previous != GXGPhase.FINAL_GN_RECOVERY:
            self.controller.transition(GXGPhase.FINAL_GN_RECOVERY, reason)
        else:
            state.reason_history.append(reason)
            state.transition_events.append(
                {
                    "from": GXGPhase.FINAL_GN_RECOVERY.value,
                    "to": GXGPhase.FINAL_GN_RECOVERY.value,
                    "reason": reason,
                    "epoch": state.current_epoch,
                    "wall_time": time.perf_counter(),
                }
            )

    def should_stop(self) -> bool:
        return self.get_phase() == GXGPhase.DONE

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    def get_phase(self) -> GXGPhase:
        return self.controller.state.phase

    def get_metrics(self) -> dict[str, float]:
        state = self.controller.state
        values = dict(self._metrics)
        values.update(
            {
                "gxg/phase": float(list(GXGPhase).index(state.phase)),
                "gxg/phase_step": float(state.phase_step),
                "gxg/switch_count": float(state.switch_count),
                "gxg/epoch": float(state.current_epoch),
                "gxg/adam_lr": self.adam.lr,
            }
        )
        for group in self.layer_groups:
            recent = self.adam.recent_norm(group.name)
            if recent is not None:
                values[f"adam_layer/{group.name}/recent_update_norm"] = recent
        return values

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "config": self.config.to_dict(),
            "adam": self.adam.state_dict(),
            "gn": self.gn.state_dict(),
            "controller": self.controller.state_dict(),
            "latest_gn_direction": self._latest_gn_direction,
            "pre_gn_parameters": self._pre_gn_parameters,
            "pre_gn_gradient": self._pre_gn_gradient,
            "best_model_state": self._best_model_state,
            "metrics": self._metrics,
            "rng": rng_state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("Unsupported GXG optimizer checkpoint version")
        if GXGConfig.from_dict(dict(state["config"])) != self.config:
            raise ValueError("GXG checkpoint configuration does not match")
        self.adam.load_state_dict(state["adam"])
        self.gn.load_state_dict(state["gn"])
        self.controller.load_state_dict(state["controller"])
        self._latest_gn_direction = dict(state["latest_gn_direction"])
        self._pre_gn_parameters = dict(state["pre_gn_parameters"])
        self._pre_gn_gradient = dict(state["pre_gn_gradient"])
        self._best_model_state = state["best_model_state"]
        self._metrics = dict(state["metrics"])
        load_rng_state_dict(state["rng"])

    def checkpoint_payload(self, *, scheduler=None, scaler=None, sampler=None, extra=None) -> dict[str, Any]:
        return build_checkpoint_payload(
            self.model, self, scheduler=scheduler, scaler=scaler, sampler=sampler, extra=extra
        )

    def save_checkpoint(self, path: str | Path, *, scheduler=None, scaler=None, sampler=None, extra=None) -> Path:
        payload = self.checkpoint_payload(scheduler=scheduler, scaler=scaler, sampler=sampler, extra=extra)
        return save_atomic_checkpoint(path, payload)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        scheduler=None,
        scaler=None,
        sampler=None,
        map_location=None,
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location)
        return restore_checkpoint_payload(
            payload,
            self.model,
            self,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
        )

    def _result(
        self,
        accepted: bool,
        started: float,
        *,
        adam_norm: float = 0.0,
        gn_norm: float = 0.0,
        bridge_norm: float = 0.0,
        reason: str | None = None,
        measurements: Mapping[str, float] | None = None,
    ) -> StepResult:
        return StepResult(
            phase=self.get_phase(),
            update_accepted=accepted,
            adam_update_norm=adam_norm,
            gn_update_norm=gn_norm,
            bridge_update_norm=bridge_norm,
            switch_reason=reason,
            wall_time_seconds=time.perf_counter() - started,
            measurements=dict(measurements or {}),
        )


def gxg_optimizer(
    model: nn.Module,
    config: GXGConfig | Mapping[str, Any] | str | Path,
    layer_partition: tuple[LayerGroup, ...] | Mapping[str, Sequence[str]] | None = None,
    metric_fn: MetricFunction | None = None,
    distributed_adapter: DistributedLayerAdapter | None = None,
) -> GXGOptimizer:
    """Public factory with the exact configured optimizer name required by the plan."""

    return GXGOptimizer(model, config, layer_partition, metric_fn, distributed_adapter)
