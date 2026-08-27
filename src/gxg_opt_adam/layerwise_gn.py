from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, grad, jvp, vjp

from .config import GNConfig
from .distributed_layers import DistributedLayerAdapter, SingleProcessLayerAdapter
from .gn_inner_solvers import InnerSolveResult, solve_quadratic
from .layer_partition import LayerGroup, muon_eligible_parameter_names
from .line_search import LineSearchResult, global_line_search
from .probes import norm
from .state import LayerGNState
from .types import FunctionalBatch


Direction = dict[str, torch.Tensor]


@dataclass(frozen=True)
class GNProposal:
    direction: Direction
    predicted_reduction: float
    finite: bool
    descent: bool
    update_norm: float
    layer_results: Mapping[str, InnerSolveResult]


@dataclass(frozen=True)
class GNStepResult:
    accepted: bool
    direction: Direction
    update_norm: float
    proposal_update_norm: float
    finite: bool
    descent: bool
    line_search: LineSearchResult
    diagnostics: Mapping[str, float]


def _curvature_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def layer_gradient_and_gvp(
    model: nn.Module,
    batch: FunctionalBatch,
    parameter_names: tuple[str, ...],
    vector: Mapping[str, torch.Tensor] | None = None,
) -> tuple[Direction, Direction | None]:
    """Return layer gradient and an exact generalized Gauss--Newton product.

    This is J^T H_loss J v, obtained with JVP, output-loss HVP, and VJP. It
    never constructs a parameter-space Hessian or GN matrix.
    """

    all_parameters = dict(model.named_parameters())
    unknown = set(parameter_names) - set(all_parameters)
    if unknown:
        raise ValueError(f"Unknown layer parameters: {sorted(unknown)}")
    base = tuple(all_parameters[name] for name in parameter_names)

    def layer_function(*values: torch.Tensor) -> Any:
        parameters = dict(all_parameters)
        parameters.update(zip(parameter_names, values, strict=True))
        return functional_call(model, parameters, batch.args, dict(batch.kwargs), strict=False)

    output, pullback = vjp(layer_function, *base)
    loss_gradient_fn = grad(batch.loss_fn)
    output_gradient = loss_gradient_fn(output)
    layer_gradient = pullback(output_gradient)
    gradient_map = {
        name: value.to(dtype=_curvature_dtype(value))
        for name, value in zip(parameter_names, layer_gradient, strict=True)
    }
    if vector is None:
        return gradient_map, None
    tangents = tuple(vector[name].to(device=value.device, dtype=value.dtype) for name, value in zip(parameter_names, base, strict=True))
    _, output_tangent = jvp(layer_function, base, tangents)
    _, loss_hvp = jvp(loss_gradient_fn, (output,), (output_tangent,))
    products = pullback(loss_hvp)
    product_map = {
        name: value.to(dtype=_curvature_dtype(value))
        for name, value in zip(parameter_names, products, strict=True)
    }
    return gradient_map, product_map


class LayerwiseGN:
    def __init__(
        self,
        model: nn.Module,
        groups: tuple[LayerGroup, ...],
        config: GNConfig,
        adapter: DistributedLayerAdapter | None = None,
    ) -> None:
        self.model = model
        self.groups = groups
        self.config = config
        self.adapter = adapter or SingleProcessLayerAdapter()
        self.muon_names = muon_eligible_parameter_names(model)
        self.states = {group.name: LayerGNState(config.initial_damping) for group in groups}

    def reset_linearization(self, damping_multiplier: float = 1.0) -> None:
        for state in self.states.values():
            state.warm_start.clear()
            state.proposed_direction.clear()
            state.accepted_direction.clear()
            state.inner_state.clear()
            state.damping = self.config.initial_damping * damping_multiplier

    def propose(self, batch: FunctionalBatch, *, update_norm_multiplier: float = 1.0) -> GNProposal:
        merged: Direction = {}
        layer_results: dict[str, InnerSolveResult] = {}
        predicted = 0.0
        all_finite = True
        all_descent = True
        for group in self.groups:
            if not self.adapter.owns(group.name):
                continue
            state = self.states[group.name]
            gradient, _ = layer_gradient_and_gvp(self.model, batch, group.parameter_names)

            def operator(value: Mapping[str, torch.Tensor], *, names=group.parameter_names, damping=state.damping):
                _, product = layer_gradient_and_gvp(self.model, batch, names, value)
                assert product is not None
                regularization = damping + self.config.inner_weight_decay
                return {name: product[name] + regularization * value[name].float() for name in names}

            result = solve_quadratic(
                gradient,
                operator,
                state.warm_start,
                state.inner_state,
                self.muon_names & set(group.parameter_names),
                self.config,
            )
            layer_results[group.name] = result
            state.inner_state = result.state
            state.warm_start = {name: value.detach().clone() for name, value in result.direction.items()}
            state.proposed_direction = {name: value.detach().clone() for name, value in result.direction.items()}
            layer_norm = norm(result.direction)
            state.curvature_batch_id = batch.batch_id
            state.diagnostics = {
                "inner_iterations": float(result.iterations),
                "residual_norm": result.residual_norm,
                "predicted_reduction": result.predicted_reduction,
                "update_norm": layer_norm,
            }
            safe = layer_norm <= self.config.max_update_norm * update_norm_multiplier
            all_finite = all_finite and result.finite and safe
            all_descent = all_descent and result.descent
            predicted += result.predicted_reduction
            merged.update(result.direction)
        merged = self.adapter.merge_directions(merged)
        update_norm = norm(merged)
        if not math.isfinite(update_norm):
            all_finite = False
        return GNProposal(merged, predicted, all_finite, all_descent, update_norm, layer_results)

    def step(
        self,
        batch: FunctionalBatch,
        reference_loss_closure,
        reference_batch_id: str | None,
        *,
        line_search_scale: float = 1.0,
        update_norm_multiplier: float = 1.0,
        weight_decay: float = 0.0,
    ) -> GNStepResult:
        proposal = self.propose(batch, update_norm_multiplier=update_norm_multiplier)
        alphas = tuple(alpha * line_search_scale for alpha in self.config.line_search_alphas)
        if 0.0 not in alphas:
            alphas += (0.0,)
        if not proposal.finite or not proposal.descent:
            alphas = (0.0,)
        search = global_line_search(
            self.model,
            proposal.direction,
            reference_loss_closure,
            alphas,
            proposal.predicted_reduction,
            weight_decay=weight_decay,
            decay_step_size=self.config.inner_lr,
        )
        selected = {name: search.alpha * value for name, value in proposal.direction.items()} if search.accepted else {}
        for group in self.groups:
            state = self.states[group.name]
            state.reference_batch_id = reference_batch_id
            if search.accepted:
                state.accepted_steps += 1
                state.accepted_direction = {
                    name: selected[name].detach().clone() for name in group.parameter_names if name in selected
                }
            else:
                state.rejected_steps += 1
            if not search.accepted or search.reduction_ratio < self.config.min_reduction_ratio:
                state.damping *= self.config.damping_increase
            elif search.reduction_ratio >= self.config.strong_reduction_ratio:
                state.damping = max(state.damping * self.config.damping_decrease, 1.0e-12)
        diagnostics = {
            "gn_predicted_reduction": search.predicted_reduction,
            "gn_actual_reduction": search.actual_reduction,
            "gn_reduction_ratio": search.reduction_ratio,
            "gn_line_search_alpha": search.alpha,
            "gn_update_norm": norm(selected),
        }
        for group in self.groups:
            state = self.states[group.name]
            prefix = f"gn_layer/{group.name}"
            diagnostics[f"{prefix}/damping"] = state.damping
            diagnostics[f"{prefix}/update_norm"] = state.diagnostics.get("update_norm", 0.0)
            diagnostics[f"{prefix}/inner_iterations"] = state.diagnostics.get("inner_iterations", 0.0)
            diagnostics[f"{prefix}/residual_norm"] = state.diagnostics.get("residual_norm", 0.0)
        return GNStepResult(
            search.accepted,
            selected,
            diagnostics["gn_update_norm"],
            proposal.update_norm,
            proposal.finite and search.finite,
            proposal.descent,
            search,
            diagnostics,
        )

    def state_dict(self) -> dict[str, Any]:
        return {name: state.state_dict() for name, state in self.states.items()}

    def rollback_last_accept(self) -> None:
        """Correct diagnostics/state when a competitive GN trial is rolled back."""

        for state in self.states.values():
            if state.accepted_steps:
                state.accepted_steps -= 1
            state.rejected_steps += 1
            state.accepted_direction.clear()
            state.damping *= self.config.damping_increase

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != set(self.states):
            raise ValueError("GN checkpoint layer groups do not match the current model partition")
        parameters = dict(self.model.named_parameters())
        restored = {}
        for group in self.groups:
            layer_state = LayerGNState.from_state_dict(dict(state[group.name]))
            for bank_name in ("warm_start", "proposed_direction", "accepted_direction"):
                bank = getattr(layer_state, bank_name)
                setattr(
                    layer_state,
                    bank_name,
                    {
                        name: tensor.to(device=parameters[name].device)
                        for name, tensor in bank.items()
                    },
                )
            layer_state.inner_state = {
                name: {
                    key: value.to(device=parameters[name].device) if isinstance(value, torch.Tensor) else value
                    for key, value in inner.items()
                }
                for name, inner in layer_state.inner_state.items()
            }
            restored[group.name] = layer_state
        self.states = restored
