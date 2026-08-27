from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LineSearchResult:
    accepted: bool
    finite: bool
    alpha: float
    initial_loss: float
    selected_loss: float
    actual_reduction: float
    predicted_reduction: float
    reduction_ratio: float


@torch.no_grad()
def global_line_search(
    model: nn.Module,
    direction: Mapping[str, torch.Tensor],
    closure: Callable[[], torch.Tensor],
    alphas: Sequence[float],
    predicted_reduction: float,
    *,
    weight_decay: float = 0.0,
    decay_step_size: float = 0.0,
) -> LineSearchResult:
    parameters = dict(model.named_parameters())
    missing = set(direction) - set(parameters)
    if missing:
        raise ValueError(f"Direction contains unknown parameters: {sorted(missing)}")
    snapshots = {name: parameter.detach().clone() for name, parameter in parameters.items() if name in direction}
    initial = float(closure().detach().float().item())
    if not math.isfinite(initial):
        return LineSearchResult(False, False, 0.0, initial, initial, 0.0, 0.0, 0.0)
    best_alpha = 0.0
    best_loss = initial
    try:
        for alpha in alphas:
            for name, base in snapshots.items():
                candidate = base if alpha == 0 else base * (1 - alpha * decay_step_size * weight_decay) + alpha * direction[name]
                parameters[name].copy_(candidate)
            candidate_loss = float(closure().detach().float().item())
            if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_loss = candidate_loss
                best_alpha = float(alpha)
    finally:
        for name, base in snapshots.items():
            parameters[name].copy_(base)
    for name, base in snapshots.items():
        selected = base if best_alpha == 0 else base * (1 - best_alpha * decay_step_size * weight_decay) + best_alpha * direction[name]
        parameters[name].copy_(selected)
    actual = initial - best_loss
    scaled_prediction = max(best_alpha * predicted_reduction, 0.0)
    ratio = actual / (scaled_prediction + 1.0e-12) if best_alpha else 0.0
    return LineSearchResult(True, True, best_alpha, initial, best_loss, actual, scaled_prediction, ratio) if best_alpha > 0 and actual > 0 else LineSearchResult(False, True, 0.0, initial, initial, 0.0, 0.0, 0.0)
