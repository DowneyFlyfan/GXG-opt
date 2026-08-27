from __future__ import annotations

import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .acceptance import compare_candidates_statelessly, predicted_hybrid_reduction
from .adam_state import AdamStateBank
from .blocks import BlockRegistry, BlockSpec
from .checkpointing import (
    checkpoint_payload,
    load_rng_state_dict,
    restore_payload,
    rng_state_dict,
    save_atomic,
)
from .config import GuidedAdamConfig
from .ggn_operator import GGNBlockOperator
from .krylov import KrylovError, build_krylov_basis, reduced_gn_solve
from .metrics import OptimizerEvent, OptimizerMetrics
from .staleness import staleness_weight
from .state import BlockGuidanceState
from .tensor_ops import map_norm, subspace_overlap
from .trust_region import apply_trust_limits, decrease_damping, increase_damping
from .types import GuidedStepContext, GuidedStepResult


class GNGuidedAdamW(torch.optim.Optimizer):
    """Fixed-frequency GGN-guided AdamW with safe Adam fallback."""

    STATE_VERSION = 2

    def __init__(
        self,
        model: nn.Module,
        config: GuidedAdamConfig | Mapping[str, Any] | str | Path,
    ) -> None:
        self.model = model
        self.config = self._load_config(config)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("GN-guided AdamW requires trainable parameters")
        super().__init__(
            trainable,
            defaults={
                "lr": self.config.adamw.lr,
                "betas": self.config.adamw.betas,
                "eps": self.config.adamw.eps,
                "weight_decay": self.config.adamw.weight_decay,
            },
        )
        self.registry = BlockRegistry(model, self.config.gn)
        self.blocks = self.registry.by_name()
        self.guidance = {
            spec.name: BlockGuidanceState(self.config.gn.initial_damping)
            for spec in self.registry.enabled
        }
        self.adam = AdamStateBank(model, self.config.adamw)
        self.metrics = OptimizerMetrics()
        self.step_count = 0
        self._last_epoch: int | None = None
        self._last_duty_active: bool | None = None
        self._last_measurements: dict[str, float] = {}

    @staticmethod
    def _load_config(config: GuidedAdamConfig | Mapping[str, Any] | str | Path) -> GuidedAdamConfig:
        if isinstance(config, GuidedAdamConfig):
            return config
        if isinstance(config, Mapping):
            return GuidedAdamConfig.from_dict(dict(config))
        path = Path(config)
        return GuidedAdamConfig.from_json(path) if path.suffix.lower() == ".json" else GuidedAdamConfig.from_yaml(path)

    @property
    def lr(self) -> float:
        if len(self.param_groups) != 1:
            raise RuntimeError("The fixed prototype supports one AdamW parameter group")
        return float(self.param_groups[0]["lr"])

    def _duty_cycle_active(self, epoch: int) -> bool:
        schedule = self.config.fixed_epoch_duty_cycle
        if not self.config.gn.enabled:
            return False
        if not schedule.enabled:
            return True
        if epoch < schedule.start_epoch:
            return False
        period = schedule.on_epochs + schedule.off_epochs
        return (epoch - schedule.start_epoch) % period < schedule.on_epochs

    def _refresh_due(self, duty_active: bool) -> bool:
        gn = self.config.gn
        if not duty_active or self.step_count < gn.warmup_steps:
            return False
        activation_refresh = (
            self.config.fixed_epoch_duty_cycle.enabled
            and self.config.fixed_epoch_duty_cycle.refresh_on_activation
            and self._last_duty_active is not True
        )
        interval_refresh = (self.step_count - gn.warmup_steps) % gn.refresh_interval == 0
        return activation_refresh or interval_refresh

    def step(self, context: GuidedStepContext | None = None, closure=None) -> GuidedStepResult:
        del closure
        started = time.perf_counter()
        context = context or GuidedStepContext()
        if (
            not isinstance(context.epoch, int)
            or isinstance(context.epoch, bool)
            or context.epoch < 0
        ):
            raise ValueError("Epoch must be a non-negative integer")
        if self._last_epoch is not None and context.epoch < self._last_epoch:
            raise ValueError("Epoch cannot move backwards")
        if context.gradient_accumulation <= 0 or context.tokens < 0:
            raise ValueError("Gradient accumulation must be positive and tokens cannot be negative")
        duty_active = self._duty_cycle_active(context.epoch)
        gradients = self.adam.update()
        adam_candidate = self.adam.ordinary_candidate(self.lr)
        adam_candidate_ready = time.perf_counter()
        refreshed = self._refresh_due(duty_active)
        curvature_seconds = 0.0
        acceptance_seconds = 0.0
        refresh_reason = None
        refresh_metrics: dict[str, float] = {}
        if refreshed:
            curvature_seconds, refresh_reason, refresh_metrics = self._refresh(context, gradients)

        if duty_active:
            hybrid_started = time.perf_counter()
            hybrid_candidate, reduced_terms, active_blocks, hybrid_metrics, hybrid_error = self._hybrid_candidate(
                gradients, adam_candidate
            )
            if self.config.gn.enabled:
                curvature_seconds += time.perf_counter() - hybrid_started
        else:
            hybrid_candidate = dict(adam_candidate)
            reduced_terms = {}
            active_blocks = []
            hybrid_metrics = {}
            hybrid_error = None
        measurements = {**refresh_metrics, **hybrid_metrics}
        guidance_used = bool(active_blocks) and hybrid_error is None
        accepted = guidance_used
        decision_reason = "stale_basis_reuse" if guidance_used and not refreshed else "adamw_only"
        if self.config.gn.enabled and not duty_active:
            decision_reason = "fixed_epoch_duty_cycle_off"
        cheap_prediction = (
            predicted_hybrid_reduction(gradients, hybrid_candidate, reduced_terms)
            if guidance_used
            else 0.0
        )
        if guidance_used:
            measurements["optimizer/predicted_hybrid_reduction"] = cheap_prediction
        if hybrid_error is not None:
            accepted = False
            decision_reason = hybrid_error
        elif guidance_used and not refreshed and (
            not math.isfinite(cheap_prediction) or cheap_prediction <= 0
        ):
            accepted = False
            decision_reason = "non_descent_stale_guidance"
        elif refreshed:
            if refresh_reason is not None:
                accepted = False
                decision_reason = refresh_reason
            elif not guidance_used:
                accepted = False
                decision_reason = "no_valid_guided_blocks"
            elif context.acceptance_batch is None:
                accepted = False
                decision_reason = "missing_acceptance_batch"
            else:
                try:
                    decision = compare_candidates_statelessly(
                        self.model,
                        context.acceptance_batch,
                        adam_candidate,
                        hybrid_candidate,
                        predicted_hybrid_reduction=cheap_prediction,
                        rho_min=self.config.gn.rho_min,
                        acceptance_margin=self.config.gn.acceptance_margin,
                        lr=self.lr,
                        weight_decay=self.adam.weight_decay,
                        weight_decay_names=set(gradients),
                    )
                    acceptance_seconds = decision.wall_time_seconds
                    accepted = decision.accepted
                    decision_reason = decision.reason
                    measurements.update(
                        {
                            "acceptance/initial_loss": decision.initial_loss,
                            "acceptance/hybrid_loss": decision.hybrid_loss,
                            "acceptance/adam_loss": decision.adam_loss,
                            "acceptance/actual_hybrid_reduction": decision.actual_hybrid_reduction,
                            "acceptance/actual_adam_reduction": decision.actual_adam_reduction,
                            "acceptance/predicted_hybrid_reduction": decision.predicted_hybrid_reduction,
                            "acceptance/rho": decision.rho,
                        }
                    )
                except Exception as error:  # candidate evaluation must fail closed
                    accepted = False
                    decision_reason = f"acceptance_failure:{type(error).__name__}"

        if not accepted and not self.config.gn.fallback_to_adamw and self.config.gn.enabled:
            raise RuntimeError(f"GN guidance rejected without AdamW fallback: {decision_reason}")
        chosen = hybrid_candidate if accepted else adam_candidate
        apply_started = time.perf_counter()
        chosen_norm = self.adam.apply(chosen, self.lr)
        applied_at = time.perf_counter()
        adam_baseline_seconds = (adam_candidate_ready - started) + (applied_at - apply_started)
        self.metrics.observe_adam_time(max(adam_baseline_seconds, 1.0e-12))
        if accepted:
            for name in active_blocks:
                state = self.guidance[name]
                state.accepted_events += int(refreshed)
                state.consecutive_failures = 0
                if refreshed:
                    state.damping = decrease_damping(state.damping, self.config.gn)
                self.adam.remove_first_moment_subspace(
                    name,
                    state.basis,
                    self.config.gn.momentum_subspace_decay,
                )
        elif refreshed:
            self._reject_refreshed_blocks(active_blocks)

        wall_time = time.perf_counter() - started
        baseline_time = max(self.metrics.adam_step_ema_seconds, 1.0e-12)
        event_cost = (curvature_seconds + acceptance_seconds) / baseline_time
        measurements.update(
            {
                "optimizer/adam_update_norm": map_norm(adam_candidate),
                "optimizer/hybrid_update_norm": map_norm(hybrid_candidate),
                "optimizer/chosen_update_norm": chosen_norm,
                "optimizer/gn_cost_adam_steps": event_cost,
                "data/curvature_reuses_training": float(context.curvature_reuses_training_data),
                "data/acceptance_reuses_other": float(context.acceptance_reuses_other_data),
                "data/gradient_accumulation": float(context.gradient_accumulation),
                "data/tokens": float(context.tokens),
                "schedule/epoch": float(context.epoch),
                "schedule/fixed_duty_cycle_active": float(duty_active),
            }
        )
        if torch.cuda.is_available():
            measurements["memory/peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
            measurements["memory/peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved())
        event = OptimizerEvent(
            step=self.step_count,
            refreshed=refreshed,
            accepted=accepted,
            reason=decision_reason,
            wall_time_seconds=wall_time,
            curvature_time_seconds=curvature_seconds,
            acceptance_time_seconds=acceptance_seconds,
            adam_step_ema_seconds=self.metrics.adam_step_ema_seconds,
            gn_cost_adam_steps=event_cost,
            metrics=measurements,
        )
        self.metrics.add(event)
        self._last_measurements = measurements
        result = GuidedStepResult(
            step=self.step_count,
            update_applied=True,
            refreshed=refreshed,
            guidance_used=guidance_used,
            guidance_accepted=accepted,
            fallback_reason=None if accepted else decision_reason,
            adam_update_norm=measurements["optimizer/adam_update_norm"],
            hybrid_update_norm=measurements["optimizer/hybrid_update_norm"],
            wall_time_seconds=wall_time,
            curvature_time_seconds=curvature_seconds,
            acceptance_time_seconds=acceptance_seconds,
            measurements=measurements,
        )
        self._last_epoch = context.epoch
        self._last_duty_active = duty_active
        self.step_count += 1
        return result

    def _refresh(
        self,
        context: GuidedStepContext,
        gradients: dict[str, torch.Tensor],
    ) -> tuple[float, str | None, dict[str, float]]:
        started = time.perf_counter()
        metrics: dict[str, float] = {}
        if context.curvature_batch is None:
            return 0.0, "missing_curvature_batch", metrics
        built = 0
        for spec in self.registry.enabled:
            if spec.name not in gradients:
                metrics[f"block/{spec.name}/missing_gradient"] = 1.0
                continue
            state = self.guidance[spec.name]
            if self.step_count < state.cooldown_until:
                metrics[f"block/{spec.name}/cooldown"] = 1.0
                continue
            try:
                operator = GGNBlockOperator(self.model, spec, context.curvature_batch)
                krylov = build_krylov_basis(
                    operator.matvec,
                    gradients[spec.name].reshape(-1),
                    min(self.config.gn.rank, spec.numel),
                    reorthogonalization_passes=self.config.gn.reorthogonalization_passes,
                    negative_eigenvalue_tolerance=self.config.gn.negative_eigenvalue_tolerance,
                )
                overlap = subspace_overlap(krylov.basis, state.basis)
                state.basis = krylov.basis.detach().clone()
                state.reduced_matrix = krylov.reduced_matrix.detach().clone()
                state.parameter_snapshot = spec.parameters[0].detach().float().reshape(-1).clone()
                state.refresh_step = self.step_count
                state.subspace_overlap = overlap
                state.curvature_batch_id = context.curvature_batch.batch_id
                state.last_metrics = {
                    "rank": float(krylov.rank),
                    "matvecs": float(krylov.matvecs),
                    "orthogonality_error": krylov.orthogonality_error,
                    "minimum_eigenvalue": krylov.minimum_eigenvalue,
                    "maximum_eigenvalue": krylov.maximum_eigenvalue,
                    "build_time_seconds": krylov.build_time_seconds,
                    "matvec_time_seconds": operator.matvec_time_seconds,
                    "subspace_overlap": overlap,
                }
                for key, value in state.last_metrics.items():
                    metrics[f"block/{spec.name}/{key}"] = value
                built += 1
            except Exception as error:
                self._invalidate_block(spec, f"refresh_failure:{type(error).__name__}")
                metrics[f"block/{spec.name}/refresh_failure"] = 1.0
        elapsed = time.perf_counter() - started
        return elapsed, None if built else "all_curvature_builds_failed", metrics

    def _hybrid_candidate(
        self,
        gradients: dict[str, torch.Tensor],
        adam_candidate: dict[str, torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        list[str],
        dict[str, float],
        str | None,
    ]:
        hybrid = dict(adam_candidate)
        reduced_terms: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        active: list[str] = []
        metrics: dict[str, float] = {}
        try:
            for spec in self.registry.enabled:
                if spec.name not in gradients:
                    continue
                state = self.guidance[spec.name]
                if state.basis is None or state.reduced_matrix is None or state.parameter_snapshot is None:
                    continue
                if self.step_count < state.cooldown_until:
                    continue
                stale = staleness_weight(
                    spec.parameters[0],
                    state.parameter_snapshot,
                    step=self.step_count,
                    refresh_step=state.refresh_step,
                    max_age=self.config.gn.max_basis_age,
                    max_drift=self.config.gn.max_parameter_drift,
                )
                metrics[f"block/{spec.name}/basis_age"] = float(stale.age)
                metrics[f"block/{spec.name}/parameter_drift"] = stale.drift
                metrics[f"block/{spec.name}/staleness_weight"] = stale.weight
                if not stale.valid:
                    continue
                solve = reduced_gn_solve(
                    state.basis,
                    state.reduced_matrix,
                    gradients[spec.name],
                    state.damping,
                )
                trusted = apply_trust_limits(
                    solve,
                    state.reduced_matrix,
                    spec.parameters[0],
                    self.config.gn,
                )
                if not trusted.finite:
                    raise KrylovError(f"Nonfinite trust scaling for {spec.name}")
                adam_complement = self.adam.candidate_for(spec.name, self.lr, state.basis)
                guided = stale.weight * trusted.direction.reshape_as(spec.parameters[0])
                hybrid[spec.name] = adam_complement + guided
                reduced_terms[spec.name] = (state.basis, state.reduced_matrix)
                active.append(spec.name)
                metrics.update(
                    {
                        f"block/{spec.name}/damping": state.damping,
                        f"block/{spec.name}/solve_residual": solve.residual_norm,
                        f"block/{spec.name}/predicted_gn_reduction": solve.predicted_reduction,
                        f"block/{spec.name}/trust_alpha": trusted.alpha,
                        f"block/{spec.name}/curvature_norm": trusted.curvature_norm,
                        f"block/{spec.name}/relative_update": trusted.relative_update,
                        f"block/{spec.name}/adam_complement_norm": float(adam_complement.norm().item()),
                        f"block/{spec.name}/guidance_norm": float(guided.norm().item()),
                    }
                )
                ordinary_subspace = state.basis @ (
                    state.basis.T @ adam_candidate[spec.name].float().reshape(-1).to(state.basis)
                )
                ordinary_coordinates = state.basis.T @ ordinary_subspace
                metrics[f"block/{spec.name}/adam_predicted_subspace_reduction"] = -float(
                    torch.dot(gradients[spec.name].reshape(-1).to(ordinary_subspace), ordinary_subspace).item()
                ) - 0.5 * float(
                    torch.dot(ordinary_coordinates, state.reduced_matrix @ ordinary_coordinates).item()
                )
            if not all(torch.isfinite(value).all().item() for value in hybrid.values()):
                return dict(adam_candidate), {}, [], metrics, "nonfinite_hybrid_candidate"
            return hybrid, reduced_terms, active, metrics, None
        except Exception as error:
            return dict(adam_candidate), {}, [], metrics, f"hybrid_failure:{type(error).__name__}"

    def _invalidate_block(self, spec: BlockSpec, reason: str) -> None:
        state = self.guidance[spec.name]
        state.basis = None
        state.reduced_matrix = None
        state.parameter_snapshot = None
        state.refresh_step = -1
        state.consecutive_failures += 1
        state.rejected_events += 1
        state.damping = increase_damping(state.damping, self.config.gn)
        state.last_metrics = {"failure": 1.0}
        if state.consecutive_failures >= self.config.gn.failures_before_cooldown:
            state.cooldown_until = self.step_count + self.config.gn.failure_cooldown_steps
            state.consecutive_failures = 0
        state.curvature_batch_id = reason

    def _reject_refreshed_blocks(self, active_blocks: list[str]) -> None:
        refreshed_names = {
            name for name, state in self.guidance.items() if state.refresh_step == self.step_count
        }
        for name in refreshed_names | set(active_blocks):
            state = self.guidance[name]
            state.rejected_events += 1
            state.consecutive_failures += 1
            state.damping = increase_damping(state.damping, self.config.gn)
            state.basis = None
            state.reduced_matrix = None
            state.parameter_snapshot = None
            state.refresh_step = -1
            if state.consecutive_failures >= self.config.gn.failures_before_cooldown:
                state.cooldown_until = self.step_count + self.config.gn.failure_cooldown_steps
                state.consecutive_failures = 0

    def get_metrics(self) -> dict[str, float]:
        fallback_count = sum(not event.accepted for event in self.metrics.events if event.refreshed)
        cooldown_count = sum(state.cooldown_until > self.step_count for state in self.guidance.values())
        return {
            **self.metrics.summary(),
            **self._last_measurements,
            "step": float(self.step_count),
            "fallback_count": float(fallback_count),
            "cooldown_block_count": float(cooldown_count),
        }

    def state_dict(self) -> dict[str, Any]:
        parent = super().state_dict()
        return {
            "version": self.STATE_VERSION,
            "config": self.config.to_dict(),
            "param_groups": parent["param_groups"],
            "adam": self.adam.state_dict(),
            "guidance": {name: state.state_dict() for name, state in self.guidance.items()},
            "step_count": self.step_count,
            "last_epoch": self._last_epoch,
            "last_duty_active": self._last_duty_active,
            "metrics": self.metrics.state_dict(),
            "last_measurements": dict(self._last_measurements),
            "rng": rng_state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        state = dict(state_dict)
        if state.get("version") != self.STATE_VERSION:
            raise ValueError("Unsupported GN-guided AdamW checkpoint version")
        if GuidedAdamConfig.from_dict(dict(state["config"])) != self.config:
            raise ValueError("Checkpoint configuration does not match the optimizer")
        if set(state["guidance"]) != set(self.guidance):
            raise ValueError("Checkpoint guidance blocks do not match the current model")
        super().load_state_dict({"state": {}, "param_groups": state["param_groups"]})
        self.adam.load_state_dict(state["adam"])
        self.guidance = {
            name: BlockGuidanceState.from_state_dict(value, self.blocks[name].parameters[0].device)
            for name, value in state["guidance"].items()
        }
        self.step_count = int(state["step_count"])
        self._last_epoch = state["last_epoch"]
        self._last_duty_active = state["last_duty_active"]
        self.metrics.load_state_dict(state["metrics"])
        self._last_measurements = dict(state["last_measurements"])
        load_rng_state_dict(state["rng"])

    def save_checkpoint(self, path: str | Path, *, scheduler=None, scaler=None, sampler=None, extra=None) -> Path:
        payload = checkpoint_payload(
            self.model,
            self,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            extra=extra,
        )
        return save_atomic(path, payload)

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        scheduler=None,
        scaler=None,
        sampler=None,
        map_location=None,
    ) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        return restore_payload(
            payload,
            self.model,
            self,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
        )


def gn_guided_adamw(
    model: nn.Module,
    config: GuidedAdamConfig | Mapping[str, Any] | str | Path,
) -> GNGuidedAdamW:
    return GNGuidedAdamW(model, config)
