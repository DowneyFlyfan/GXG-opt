from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CorrectionAllocation:
    requested_rank: int
    allocated_rank: int
    state_bytes: int
    workspace_bytes: int
    reason: str | None = None


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def plan_dense_correction(
    matrix_shape: tuple[int, int],
    requested_rank: int,
    lanczos_steps: int,
    dtype: torch.dtype,
    remaining_budget_bytes: int,
    per_layer_budget_bytes: int | None = None,
) -> CorrectionAllocation:
    if requested_rank < 0 or remaining_budget_bytes < 0:
        raise ValueError("rank and memory budget cannot be negative")
    elements = matrix_shape[0] * matrix_shape[1]
    bytes_per_vector = elements * dtype_bytes(dtype)
    layer_budget = remaining_budget_bytes
    if per_layer_budget_bytes is not None:
        layer_budget = min(layer_budget, per_layer_budget_bytes)
    if requested_rank == 0:
        return CorrectionAllocation(0, 0, 0, 0, "rank_zero")
    # Stored U plus eigenvalues/residuals. Workspace accounts for Lanczos Q and AQ.
    state_bytes_per_rank = bytes_per_vector + 2 * dtype_bytes(dtype)
    # At minimum Lanczos needs ``rank`` vectors in both Q and A@Q in addition
    # to the retained state. Reduce rank up front instead of risking an OOM.
    minimum_bytes_per_rank = state_bytes_per_rank + 2 * bytes_per_vector
    feasible_rank = min(requested_rank, layer_budget // max(minimum_bytes_per_rank, 1))
    if feasible_rank <= 0:
        return CorrectionAllocation(requested_rank, 0, 0, 0, "memory_budget")
    state_bytes = feasible_rank * state_bytes_per_rank
    usable_workspace = max(layer_budget - state_bytes, 0)
    feasible_steps = min(
        lanczos_steps, usable_workspace // max(2 * bytes_per_vector, 1)
    )
    minimum_steps = feasible_rank
    if feasible_steps < minimum_steps:
        return CorrectionAllocation(requested_rank, 0, 0, 0, "lanczos_workspace_budget")
    workspace_bytes = int(feasible_steps * 2 * bytes_per_vector)
    reason = "rank_reduced_by_memory_budget" if feasible_rank < requested_rank else None
    return CorrectionAllocation(
        requested_rank, int(feasible_rank), int(state_bytes), workspace_bytes, reason
    )
