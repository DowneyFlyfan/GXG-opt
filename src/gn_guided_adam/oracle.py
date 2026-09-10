from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .execution import candidate_loss, current_loss
from .ggn_operator import GGNBlockOperator
from .types import FunctionalBatch


@dataclass(frozen=True)
class OracleResult:
    direction: torch.Tensor
    iterations: int
    residual_norm: float
    predicted_reduction: float
    converged: bool
    wall_time_seconds: float
    matvecs: int


@dataclass(frozen=True)
class OracleProbeResult:
    accepted: bool
    selected_alpha: float
    initial_loss: float
    oracle_loss: float
    adam_loss: float
    predicted_reduction: float
    actual_reduction: float
    directions: Mapping[str, torch.Tensor]
    block_results: Mapping[str, OracleResult]
    wall_time_seconds: float


class GNOracle:
    """High-accuracy damped matrix-free CG probe for selected checkpoints."""

    def __init__(self, operator: GGNBlockOperator, damping: float, max_iterations: int, relative_tolerance: float) -> None:
        self.operator = operator
        self.damping = damping
        self.max_iterations = max_iterations
        self.relative_tolerance = relative_tolerance

    def solve(self, gradient: torch.Tensor | None = None) -> OracleResult:
        started = time.perf_counter()
        gradient = self.operator.gradient() if gradient is None else gradient.detach().reshape(-1)
        dtype = torch.float64 if gradient.dtype == torch.float64 else torch.float32
        gradient = gradient.to(dtype=dtype)
        solution = torch.zeros_like(gradient)
        residual = -gradient.clone()
        search = residual.clone()
        residual_squared = torch.dot(residual, residual)
        initial = math.sqrt(max(float(residual_squared.item()), 0.0))
        converged = False
        iterations = 0

        def damped(value: torch.Tensor) -> torch.Tensor:
            return self.operator.matvec(value).to(value) + self.damping * value

        for iterations in range(1, self.max_iterations + 1):
            product = damped(search)
            curvature = torch.dot(search, product)
            if not torch.isfinite(curvature) or float(curvature.item()) <= 0:
                break
            alpha = residual_squared / curvature
            solution = solution + alpha * search
            residual = residual - alpha * product
            next_squared = torch.dot(residual, residual)
            if math.sqrt(max(float(next_squared.item()), 0.0)) <= self.relative_tolerance * max(initial, 1.0e-12):
                residual_squared = next_squared
                converged = True
                break
            beta = next_squared / residual_squared.clamp_min(1.0e-30)
            search = residual + beta * search
            residual_squared = next_squared
        ggn_product = self.operator.matvec(solution).to(solution)
        predicted = -float(torch.dot(gradient, solution).item()) - 0.5 * float(torch.dot(solution, ggn_product).item())
        return OracleResult(
            direction=solution,
            iterations=iterations,
            residual_norm=math.sqrt(max(float(residual_squared.item()), 0.0)),
            predicted_reduction=predicted,
            converged=converged,
            wall_time_seconds=time.perf_counter() - started,
            matvecs=self.operator.matvec_count,
        )


def run_oracle_probe(
    model: nn.Module,
    operators: Mapping[str, GGNBlockOperator],
    gradients: Mapping[str, torch.Tensor],
    adam_candidate: Mapping[str, torch.Tensor],
    acceptance_batch: FunctionalBatch,
    *,
    damping: float,
    max_iterations: int,
    relative_tolerance: float,
    lr: float,
    weight_decay: float,
    line_search_alphas: Sequence[float] = (1.0, 0.5, 0.25, 0.0),
) -> OracleProbeResult:
    """Solve selected blocks accurately, merge once, then test globally."""

    started = time.perf_counter()
    if 0.0 not in line_search_alphas:
        raise ValueError("Oracle global line search must include 0.0")
    block_results = {
        name: GNOracle(operator, damping, max_iterations, relative_tolerance).solve(gradients[name])
        for name, operator in operators.items()
    }
    full_direction = dict(adam_candidate)
    predicted = 0.0
    for name, result in block_results.items():
        full_direction[name] = result.direction.reshape_as(full_direction[name])
        predicted += result.predicted_reduction
    initial = current_loss(model, acceptance_batch)
    adam_loss = candidate_loss(
        model,
        acceptance_batch,
        dict(adam_candidate),
        lr=lr,
        weight_decay=weight_decay,
    )
    best_alpha = 0.0
    best_loss = initial
    best_direction = dict(adam_candidate)
    for alpha in line_search_alphas:
        candidate = dict(adam_candidate)
        for name, result in block_results.items():
            candidate[name] = alpha * result.direction.reshape_as(candidate[name])
        loss = candidate_loss(model, acceptance_batch, candidate, lr=lr, weight_decay=weight_decay)
        if math.isfinite(loss) and loss < best_loss:
            best_alpha = float(alpha)
            best_loss = loss
            best_direction = candidate
    accepted = best_alpha > 0 and best_loss < adam_loss
    return OracleProbeResult(
        accepted=accepted,
        selected_alpha=best_alpha,
        initial_loss=initial,
        oracle_loss=best_loss,
        adam_loss=adam_loss,
        predicted_reduction=best_alpha * predicted,
        actual_reduction=initial - best_loss,
        directions=best_direction,
        block_results=block_results,
        wall_time_seconds=time.perf_counter() - started,
    )
