from __future__ import annotations

from collections.abc import Mapping

import torch


def map_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> torch.Tensor:
    names = sorted(set(left) & set(right))
    if not names:
        return torch.tensor(0.0)
    return sum((left[name].float() * right[name].float()).sum() for name in names)


def map_norm(value: Mapping[str, torch.Tensor]) -> float:
    return float(torch.sqrt(torch.clamp(map_dot(value, value), min=0)).item())


def project(basis: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return torch.zeros_like(vector)
    flat = vector.reshape(-1).to(dtype=basis.dtype, device=basis.device)
    return (basis @ (basis.T @ flat)).reshape_as(vector)


def project_complement(basis: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return vector - project(basis, vector)


def subspace_overlap(new_basis: torch.Tensor, old_basis: torch.Tensor | None) -> float:
    if old_basis is None or old_basis.numel() == 0 or new_basis.numel() == 0:
        return 0.0
    rank = min(new_basis.shape[1], old_basis.shape[1])
    overlap = (new_basis.T @ old_basis).square().sum() / max(rank, 1)
    return float(overlap.item())
