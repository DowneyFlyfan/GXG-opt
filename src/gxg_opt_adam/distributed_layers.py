from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch


class DistributedLayerAdapter(Protocol):
    """Ownership and reductions kept deliberately outside GN mathematics."""

    def owns(self, group_name: str) -> bool: ...

    def merge_directions(self, direction: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...

    def reduce_scalar(self, value: float, reduction: str = "mean") -> float: ...


class SingleProcessLayerAdapter:
    def owns(self, group_name: str) -> bool:
        return True

    def merge_directions(self, direction: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return dict(direction)

    def reduce_scalar(self, value: float, reduction: str = "mean") -> float:
        if reduction not in {"mean", "sum", "min", "max"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        return value
