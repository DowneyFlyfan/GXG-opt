from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn

from .execution import candidate_loss, current_loss
from .types import FunctionalBatch


@dataclass(frozen=True)
class AcceptanceDecision:
    accepted: bool
    reason: str
    initial_loss: float
    hybrid_loss: float
    adam_loss: float
    actual_hybrid_reduction: float
    actual_adam_reduction: float
    predicted_hybrid_reduction: float
    rho: float
    wall_time_seconds: float


def compare_candidates_statelessly(
    model: nn.Module,
    batch: FunctionalBatch,
    adam_candidate: dict[str, torch.Tensor],
    hybrid_candidate: dict[str, torch.Tensor],
    *,
    predicted_hybrid_reduction: float,
    rho_min: float,
    acceptance_margin: float,
    lr: float,
    weight_decay: float,
    weight_decay_names: set[str] | None = None,
) -> AcceptanceDecision:
    started = time.perf_counter()
    initial = current_loss(model, batch)
    hybrid_loss = candidate_loss(
        model,
        batch,
        hybrid_candidate,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_names=weight_decay_names,
    )
    adam_loss = candidate_loss(
        model,
        batch,
        adam_candidate,
        lr=lr,
        weight_decay=weight_decay,
        weight_decay_names=weight_decay_names,
    )
    actual_hybrid = initial - hybrid_loss
    actual_adam = initial - adam_loss
    rho = actual_hybrid / (predicted_hybrid_reduction + 1.0e-12)
    finite = all(
        math.isfinite(value)
        for value in (initial, hybrid_loss, adam_loss, actual_hybrid, actual_adam, predicted_hybrid_reduction, rho)
    )
    if not finite:
        accepted, reason = False, "nonfinite_acceptance"
    elif predicted_hybrid_reduction <= 0:
        accepted, reason = False, "nonpositive_prediction"
    elif rho < rho_min:
        accepted, reason = False, "poor_reduction_ratio"
    elif actual_hybrid <= actual_adam + acceptance_margin:
        accepted, reason = False, "adam_candidate_better"
    else:
        accepted, reason = True, "hybrid_accepted"
    return AcceptanceDecision(
        accepted,
        reason,
        initial,
        hybrid_loss,
        adam_loss,
        actual_hybrid,
        actual_adam,
        predicted_hybrid_reduction,
        rho,
        time.perf_counter() - started,
    )


def predicted_hybrid_reduction(
    gradients: dict[str, torch.Tensor],
    direction: dict[str, torch.Tensor],
    reduced_terms: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> float:
    linear = sum(
        float(torch.dot(gradient.reshape(-1), direction[name].float().reshape(-1)).item())
        for name, gradient in gradients.items()
    )
    quadratic = 0.0
    for name, (basis, reduced_matrix) in reduced_terms.items():
        coordinates = basis.T @ direction[name].float().reshape(-1).to(basis)
        quadratic += float(torch.dot(coordinates, reduced_matrix @ coordinates).item())
    return -linear - 0.5 * quadratic
