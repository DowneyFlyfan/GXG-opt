from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn

from baseline_kronecker_ggn.optimizer import KroneckerGGN
from kronecker_ggn_common.config import LowRankCorrectedKroneckerGGNConfig
from kronecker_ggn_common.diagnostics import cosine_similarity, relative_difference
from kronecker_ggn_common.types import LayerDirectionStats

from .correction import corrected_direction, validate_correction_basis
from .eigensolver import signed_lanczos
from .refresh_policy import CorrectionRefreshPolicy
from .residual_operator import RelativeResidualOperator
from .state import CorrectionState
from .storage import dtype_bytes, plan_dense_correction


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _subspace_overlap(left: Tensor, right: Tensor) -> float:
    if left.shape[0] == 0 or right.shape[0] == 0:
        return 0.0
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    rank = min(left_flat.shape[0], right_flat.shape[0])
    return float((left_flat @ right_flat.T).square().sum().item()) / max(rank, 1)


class LowRankCorrectedKroneckerGGN(KroneckerGGN):
    def __init__(
        self,
        model: nn.Module,
        config: LowRankCorrectedKroneckerGGNConfig | None = None,
        **config_overrides,
    ) -> None:
        if config is not None and config_overrides:
            raise ValueError(
                "Pass either config or keyword configuration overrides, not both"
            )
        resolved = config or LowRankCorrectedKroneckerGGNConfig(**config_overrides)
        super().__init__(model, resolved)
        self.config = resolved
        self.correction_state = {
            layer.layer_id: CorrectionState() for layer in self.registry.supported
        }
        self.refresh_policy = CorrectionRefreshPolicy(resolved)

    def _generator(self, layer_id: str, device: torch.device) -> torch.Generator:
        generator = torch.Generator(device=device)
        stable_offset = sum(
            (index + 1) * ord(character) for index, character in enumerate(layer_id)
        )
        generator.manual_seed(
            self.config.seed + self.step_count * 1_000_003 + stable_offset
        )
        return generator

    def _run_eigensolver(self, layer_id: str, ggn_operator, rank: int, steps: int):
        layer = self.registry.by_id(layer_id)
        spectral = self.layer_state[layer_id].spectral
        if spectral is None:
            raise RuntimeError("Kronecker spectral state is unavailable")
        residual = RelativeResidualOperator(layer_id, spectral, ggn_operator)
        result = signed_lanczos(
            residual.matvec,
            layer.matrix_shape,
            rank,
            steps=steps,
            oversampling=self.config.correction_oversampling,
            generator=self._generator(layer_id, layer.weight.device),
            dtype=_dtype(self.config.correction_dtype),
            device=layer.weight.device,
            reorthogonalization_passes=self.config.reorthogonalization_passes,
            tolerance=self.config.lanczos_tolerance,
        )
        return result, residual.matvec_count

    def _after_curvature_update(self):
        started = time.perf_counter()
        measurements: dict[str, float] = {}
        if self.config.correction_rank == 0:
            for state in self.correction_state.values():
                state.invalidate("rank_zero")
            measurements["correction/rank_requested"] = 0.0
            return time.perf_counter() - started, measurements
        if not self.refresh_policy.should_refresh(self.step_count):
            return 0.0, measurements
        active = self.refresh_policy.active_layers(
            self.registry, self.correction_state, self.step_count
        )
        if self.config.correction_storage == "selected_layers":
            for layer_id, state in self.correction_state.items():
                if layer_id not in active:
                    state.invalidate("not_selected")
        total_budget = int(self.config.correction_memory_budget_mb * 1024**2)
        retained = sum(
            state.memory_bytes
            for layer_id, state in self.correction_state.items()
            if state.valid and layer_id not in active
        )
        remaining = max(total_budget - retained, 0)
        accepted_total = matvec_total = memory_total = 0
        for layer_id in active:
            state = self.correction_state[layer_id]
            state.requested_rank = self.config.correction_rank
            operator_entry = self._ggn_operators.get(layer_id)
            if operator_entry is None:
                state.invalidate("missing_ggn_operator")
                measurements[f"correction/{layer_id}/fallback"] = 1.0
                continue
            primary = (
                operator_entry[0]
                if isinstance(operator_entry, tuple)
                else operator_entry
            )
            validation = (
                operator_entry[1]
                if isinstance(operator_entry, tuple) and len(operator_entry) > 1
                else None
            )
            layer = self.registry.by_id(layer_id)
            per_layer = (
                None
                if self.config.per_layer_correction_memory_budget_mb is None
                else int(self.config.per_layer_correction_memory_budget_mb * 1024**2)
            )
            allocation = plan_dense_correction(
                layer.matrix_shape,
                self.config.correction_rank,
                self.config.lanczos_steps,
                _dtype(self.config.correction_dtype),
                remaining,
                per_layer,
            )
            if allocation.allocated_rank == 0:
                state.invalidate(allocation.reason or "memory_budget")
                measurements[f"correction/{layer_id}/memory_rejected"] = 1.0
                continue
            bytes_per_vector = layer.weight.numel() * dtype_bytes(
                _dtype(self.config.correction_dtype)
            )
            actual_steps = allocation.workspace_bytes // max(2 * bytes_per_vector, 1)
            actual_steps = min(
                self.config.lanczos_steps, max(allocation.allocated_rank, actual_steps)
            )
            try:
                previous_basis = state.basis if state.valid else None
                result, matvecs = self._run_eigensolver(
                    layer_id,
                    primary,
                    allocation.allocated_rank,
                    actual_steps,
                )
                if result.accepted_rank == 0:
                    state.invalidate("no_converged_eigenpairs")
                    state.matvec_count = matvecs
                    continue
                validate_correction_basis(result.basis, result.eigenvalues)
                reliability = None
                if self.config.cross_batch_validation:
                    if validation is None:
                        state.invalidate("missing_cross_batch_operator")
                        continue
                    validation_result, validation_matvecs = self._run_eigensolver(
                        layer_id,
                        validation,
                        min(result.accepted_rank, allocation.allocated_rank),
                        actual_steps,
                    )
                    matvecs += validation_matvecs
                    reliability = _subspace_overlap(
                        result.basis, validation_result.basis
                    )
                    if reliability < self.config.cross_batch_reliability_threshold:
                        state.invalidate("cross_batch_reliability")
                        state.cross_batch_reliability = reliability
                        state.matvec_count = matvecs
                        continue
                state.basis = result.basis.detach()
                state.eigenvalues = result.eigenvalues.detach()
                state.residuals = result.residuals.detach()
                state.age = 0
                state.refresh_count += 1
                state.accepted_rank = result.accepted_rank
                state.matvec_count = matvecs
                state.build_time_seconds = result.build_time_seconds
                state.memory_bytes = (
                    result.basis.numel() * result.basis.element_size()
                    + result.eigenvalues.numel() * result.eigenvalues.element_size()
                    + result.residuals.numel() * result.residuals.element_size()
                )
                state.valid = True
                state.failure_reason = allocation.reason
                state.cross_batch_reliability = reliability
                state.diagnostics = {
                    "largest_abs_eigenvalue": float(
                        result.eigenvalues.abs().max().item()
                    ),
                    "orthogonality_error": result.orthogonality_error,
                    "maximum_eigenpair_residual": float(result.residuals.max().item()),
                    "subspace_overlap_previous": 0.0
                    if previous_basis is None
                    else _subspace_overlap(result.basis, previous_basis),
                }
                remaining = max(remaining - state.memory_bytes, 0)
                accepted_total += result.accepted_rank
                matvec_total += matvecs
                memory_total += state.memory_bytes
            except (RuntimeError, ValueError, FloatingPointError) as error:
                if not self.config.fallback_on_invalid_correction:
                    raise
                state.invalidate(f"correction_build_failure:{type(error).__name__}")
        self.refresh_policy.force_refresh = False
        measurements.update(
            {
                "correction/rank_requested": float(
                    self.config.correction_rank * len(active)
                ),
                "correction/rank_accepted": float(accepted_total),
                "correction/ggn_matvecs": float(matvec_total),
                "correction/memory_bytes": float(memory_total),
            }
        )
        return time.perf_counter() - started, measurements

    def _curvature_direction(self, layer_id: str, gradient: Tensor):
        baseline, baseline_statistics = super()._curvature_direction(layer_id, gradient)
        if baseline is None:
            return None, baseline_statistics
        correction = self.correction_state[layer_id]
        if self.config.correction_rank == 0:
            reason = "rank_zero"
        elif (
            not correction.valid
            or correction.basis is None
            or correction.eigenvalues is None
        ):
            reason = correction.failure_reason or "correction_unavailable"
        elif correction.age > self.config.correction_max_age:
            correction.invalidate("correction_stale")
            reason = "correction_stale"
        else:
            reason = None
        if reason is not None:
            return baseline, LayerDirectionStats(
                layer_id,
                used_curvature=True,
                used_correction=False,
                fallback_reason=reason,
                gradient_norm=baseline_statistics.gradient_norm,
                update_norm=baseline_statistics.update_norm,
                predicted_quadratic_decrease=baseline_statistics.predicted_quadratic_decrease,
            )
        try:
            layer = self.registry.by_id(layer_id)
            spectral = self.layer_state[layer_id].spectral
            assert (
                spectral is not None
                and correction.basis is not None
                and correction.eigenvalues is not None
            )
            application = corrected_direction(
                spectral,
                gradient.reshape(layer.matrix_shape),
                correction.basis,
                correction.eigenvalues,
                eigenvalue_margin=self.config.correction_eigenvalue_margin,
                absolute_eigenvalue_cap=self.config.correction_abs_eigenvalue_cap,
            )
            direction = application.direction
            if self.config.weight_decay:
                direction = direction - self.config.weight_decay * layer.weight.detach()
            gradient_matrix = gradient.reshape(layer.matrix_shape)
            baseline_matrix = baseline.reshape(layer.matrix_shape)
            prediction = -float(
                (gradient_matrix * direction).sum().item()
            ) - 0.5 * float((direction * spectral.matvec(direction)).sum().item())
            correction.diagnostics["clipped_eigenvalue_count"] = float(
                application.clipped_count
            )
            correction.diagnostics["cosine_to_baseline"] = cosine_similarity(
                direction, baseline_matrix
            )
            correction.diagnostics["relative_update_difference"] = relative_difference(
                direction, baseline_matrix
            )
            return direction.reshape_as(gradient), LayerDirectionStats(
                layer_id,
                used_curvature=True,
                used_correction=True,
                fallback_reason=None,
                gradient_norm=float(gradient_matrix.float().norm().item()),
                update_norm=float(direction.float().norm().item()),
                predicted_quadratic_decrease=prediction,
                cosine_to_baseline=correction.diagnostics["cosine_to_baseline"],
                relative_update_difference=correction.diagnostics[
                    "relative_update_difference"
                ],
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            if not self.config.fallback_on_invalid_correction:
                raise
            correction.invalidate(f"correction_apply_failure:{type(error).__name__}")
            return baseline, LayerDirectionStats(
                layer_id,
                used_curvature=True,
                used_correction=False,
                fallback_reason=correction.failure_reason,
                gradient_norm=baseline_statistics.gradient_norm,
                update_norm=baseline_statistics.update_norm,
                predicted_quadratic_decrease=baseline_statistics.predicted_quadratic_decrease,
            )

    def set_correction(
        self,
        layer_id: str,
        basis: Tensor,
        eigenvalues: Tensor,
        residuals: Tensor | None = None,
    ) -> None:
        """Install a reference/test correction after validating shape, memory, and orthogonality."""
        layer = self.registry.by_id(layer_id)
        dtype = _dtype(self.config.correction_dtype)
        basis = basis.detach().to(device=layer.weight.device, dtype=dtype)
        eigenvalues = eigenvalues.detach().to(device=layer.weight.device, dtype=dtype)
        validate_correction_basis(basis, eigenvalues)
        if tuple(basis.shape[1:]) != layer.matrix_shape:
            raise ValueError("Correction basis has the wrong layer shape")
        memory = (
            basis.numel() * basis.element_size()
            + 2 * eigenvalues.numel() * eigenvalues.element_size()
        )
        other_memory = sum(
            state.memory_bytes
            for name, state in self.correction_state.items()
            if name != layer_id and state.valid
        )
        if memory + other_memory > self.config.correction_memory_budget_mb * 1024**2:
            raise MemoryError("Correction exceeds the configured memory budget")
        state = self.correction_state[layer_id]
        state.basis = basis
        state.eigenvalues = eigenvalues
        state.residuals = (
            torch.zeros_like(eigenvalues)
            if residuals is None
            else residuals.detach().to(eigenvalues)
        )
        state.age = 0
        state.requested_rank = basis.shape[0]
        state.accepted_rank = basis.shape[0]
        state.memory_bytes = memory
        state.valid = True
        state.failure_reason = None

    def report_realized_decrease(self, predicted: float, realized: float) -> None:
        self.refresh_policy.report_decrease(predicted, realized)

    def _age_curvature_state(self) -> None:
        super()._age_curvature_state()
        for state in self.correction_state.values():
            state.age += 1
            if state.valid and state.age > self.config.correction_max_age:
                state.invalidate("correction_stale")

    def curvature_state_dict(self) -> dict[str, Any]:
        state = super().curvature_state_dict()
        state["corrections"] = {
            layer_id: correction.state_dict()
            for layer_id, correction in self.correction_state.items()
        }
        state["correction_storage"] = self.config.correction_storage
        return state

    def get_metrics(self) -> dict[str, float]:
        metrics = super().get_metrics()
        for layer_id, state in self.correction_state.items():
            prefix = f"correction/{layer_id}"
            metrics[f"{prefix}/rank_requested"] = float(state.requested_rank)
            metrics[f"{prefix}/rank_accepted"] = float(state.accepted_rank)
            metrics[f"{prefix}/age"] = float(state.age)
            metrics[f"{prefix}/ggn_matvecs"] = float(state.matvec_count)
            metrics[f"{prefix}/memory_bytes"] = float(state.memory_bytes)
            metrics[f"{prefix}/valid"] = float(state.valid)
            if state.cross_batch_reliability is not None:
                metrics[f"{prefix}/cross_batch_reliability"] = (
                    state.cross_batch_reliability
                )
            for name, value in state.diagnostics.items():
                metrics[f"{prefix}/{name}"] = value
            if state.eigenvalues is not None:
                for index, value in enumerate(state.eigenvalues):
                    metrics[f"{prefix}/eigenvalue_{index}"] = float(value.item())
            if state.residuals is not None:
                for index, value in enumerate(state.residuals):
                    metrics[f"{prefix}/eigenpair_residual_{index}"] = float(
                        value.item()
                    )
        return metrics

    def load_curvature_state_dict(self, state: dict[str, Any]) -> None:
        super().load_curvature_state_dict(state)
        saved = state.get("corrections", {})
        if set(saved) != set(self.correction_state):
            raise ValueError(
                "Correction checkpoint layer registry does not match the model"
            )
        for layer_id, value in saved.items():
            layer = self.registry.by_id(layer_id)
            current = self.correction_state[layer_id]
            dtype = _dtype(self.config.correction_dtype)
            current.basis = (
                None
                if value["basis"] is None
                else value["basis"].to(device=layer.weight.device, dtype=dtype)
            )
            current.eigenvalues = (
                None
                if value["eigenvalues"] is None
                else value["eigenvalues"].to(device=layer.weight.device, dtype=dtype)
            )
            current.residuals = (
                None
                if value["residuals"] is None
                else value["residuals"].to(device=layer.weight.device, dtype=dtype)
            )
            current.age = int(value["age"])
            current.refresh_count = int(value["refresh_count"])
            current.requested_rank = int(value["requested_rank"])
            current.accepted_rank = int(value["accepted_rank"])
            current.matvec_count = int(value["matvec_count"])
            current.build_time_seconds = float(value["build_time_seconds"])
            current.memory_bytes = int(value["memory_bytes"])
            current.valid = bool(value["valid"])
            current.failure_reason = value["failure_reason"]
            current.cross_batch_reliability = value["cross_batch_reliability"]
            current.diagnostics = dict(value["diagnostics"])
            if (
                current.valid
                and current.basis is not None
                and current.eigenvalues is not None
            ):
                validate_correction_basis(current.basis, current.eigenvalues)


def low_rank_corrected_kronecker_ggn(
    model: nn.Module,
    config: LowRankCorrectedKroneckerGGNConfig | None = None,
    **kwargs,
) -> LowRankCorrectedKroneckerGGN:
    return LowRankCorrectedKroneckerGGN(model, config, **kwargs)
