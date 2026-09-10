"""Finite feature-budget steepest descent on the column Stiefel manifold."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch


@dataclass(frozen=True)
class FeaturePolarDiagnostics:
    feature_budget: float
    orthogonality_error: float


def column_polar(matrix: torch.Tensor) -> torch.Tensor:
    """Return the reduced column-polar factor of a tall or square matrix."""
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("column polar requires a tall or square matrix")
    # GPT feature covariances can be ill-conditioned.  CUDA FP32 SVD then
    # returns factors whose orthogonality error is about 5e-3 at width 512;
    # factor in FP64 and cast only the final model weight back to FP32.
    left, _, right_transpose = torch.linalg.svd(matrix.double(), full_matrices=False)
    return (left @ right_transpose).to(matrix.dtype)


def feature_polar_update(
    weight: torch.Tensor,
    gradient: torch.Tensor,
    covariance: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, FeaturePolarDiagnostics]:
    """Apply ``polar(W C - alpha G)`` and measure its real feature movement."""
    if weight.ndim != 2 or weight.shape[0] < weight.shape[1]:
        raise ValueError("feature polar update requires a tall or square weight")
    if gradient.shape != weight.shape:
        raise ValueError("gradient shape must equal weight shape")
    columns = weight.shape[1]
    if covariance.shape != (columns, columns):
        raise ValueError("covariance shape must equal the input-feature dimension")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    covariance = covariance.double()
    if not torch.allclose(covariance, covariance.T, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("covariance must be symmetric")
    candidate = weight.double() @ covariance - alpha * gradient.double()
    updated = column_polar(candidate)
    difference = updated - weight.double()
    budget_squared = torch.sum((difference @ covariance) * difference).clamp_min(0.0)
    identity = torch.eye(columns, device=updated.device, dtype=updated.dtype)
    diagnostics = FeaturePolarDiagnostics(
        feature_budget=float(torch.sqrt(budget_squared)),
        orthogonality_error=float(torch.linalg.vector_norm(updated.T @ updated - identity)),
    )
    return updated.to(weight.dtype), diagnostics


def stiefel_feature_parameter_names(model: torch.nn.Module) -> set[str]:
    """Select only the source-valid native tall/square GPT block linears."""
    selected: set[str] = set()
    eligible_suffixes = {"qkv", "projection", "feedforward_in"}
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module_name.rsplit(".", 1)[-1] not in eligible_suffixes:
            continue
        if module.weight.shape[0] < module.weight.shape[1]:
            continue
        selected.add(f"{module_name}.weight")
    return selected


class InputCovarianceCollector:
    """Accumulate real linear-layer input covariances without retaining graphs."""

    def __init__(self, modules: Mapping[str, torch.nn.Linear]) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._handles = [
            module.register_forward_pre_hook(self._hook(name))
            for name, module in modules.items()
        ]

    def _hook(self, name: str):
        def collect(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                raise ValueError("linear pre-hook received no input")
            value = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            # The enclosing model forward pass uses BF16 autocast.  The feature
            # metric is defined by the real input Gram matrix, so form it in
            # FP64 rather than allowing autocast or an ill-conditioned FP32
            # Gram matrix to perturb its low-eigenvalue directions.
            with torch.autocast(device_type=value.device.type, enabled=False):
                value = value.double()
                gram = value.T @ value
            self._sums[name] = self._sums.get(name, torch.zeros_like(gram)).add_(gram)
            self._counts[name] = self._counts.get(name, 0) + value.shape[0]
        return collect

    def consume(self, name: str) -> torch.Tensor | None:
        count = self._counts.pop(name, 0)
        total = self._sums.pop(name, None)
        return None if total is None or count == 0 else total / count

    def clear(self) -> None:
        self._sums.clear()
        self._counts.clear()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()
