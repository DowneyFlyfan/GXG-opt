"""Sampled-column Nyström approximation for a matrix-free positive curvature."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class NystromState:
    """Low-rank Nyström factor whose damped inverse uses Woodbury."""

    basis: Tensor
    damping: float
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.basis.ndim != 2 or self.basis.shape[1] == 0:
            raise ValueError("Nyström basis must be a non-empty matrix")
        if self.damping <= 0:
            raise ValueError("Nyström damping must be positive")

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    def inverse_action(self, vector: Tensor) -> Tensor:
        """Return ``(Z Z^T + damping I)^-1 vector`` without dense curvature."""
        if vector.ndim != 1 or vector.numel() != self.basis.shape[0]:
            raise ValueError("Nyström inverse vector has the wrong shape")
        rank = self.basis.shape[1]
        small = torch.zeros(rank, rank, device=vector.device, dtype=vector.dtype)
        right_hand_side = torch.zeros(rank, device=vector.device, dtype=vector.dtype)
        chunk_size = 1_048_576
        for start in range(0, vector.numel(), chunk_size):
            stop = min(start + chunk_size, vector.numel())
            basis_chunk = self.basis[start:stop].to(device=vector.device, dtype=vector.dtype)
            small.add_(basis_chunk.T @ basis_chunk)
            right_hand_side.add_(basis_chunk.T @ vector[start:stop])
        small.diagonal().add_(self.damping)
        coefficients = torch.linalg.solve(small, right_hand_side)
        result = vector / self.damping
        for start in range(0, vector.numel(), chunk_size):
            stop = min(start + chunk_size, vector.numel())
            basis_chunk = self.basis[start:stop].to(device=vector.device, dtype=vector.dtype)
            result[start:stop].sub_(basis_chunk @ coefficients, alpha=1.0 / self.damping)
        if not torch.isfinite(result).all():
            raise RuntimeError("Nyström inverse action is non-finite")
        return result


def build_nystrom_state(
    matvec: Callable[[Tensor], Tensor],
    *,
    dimension: int,
    indices: Sequence[int],
    rank: int,
    damping: float,
    dtype: torch.dtype = torch.float32,
    storage_dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> NystromState:
    """Build the rank-limited column Nyström factor from implicit curvature."""
    if dimension <= 0 or rank <= 0 or damping <= 0:
        raise ValueError("dimension, rank, and damping must be positive")
    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Nyström indices must be non-empty and unique")
    if any(index < 0 or index >= dimension for index in selected):
        raise ValueError("Nyström index lies outside the curvature dimension")
    if rank > len(selected):
        raise ValueError("Nyström rank cannot exceed sampled column count")
    device = torch.device("cpu") if device is None else device
    columns: list[Tensor] = []
    for index in selected:
        coordinate = torch.zeros(dimension, device=device, dtype=dtype)
        coordinate[index] = 1
        column = matvec(coordinate)
        if column.shape != coordinate.shape:
            raise ValueError("curvature matvec returned the wrong vector shape")
        columns.append(column.detach().to(device=device, dtype=dtype))
    sampled_columns = torch.stack(columns, dim=1)
    intersection = sampled_columns[torch.as_tensor(selected, device=device)]
    intersection = 0.5 * (intersection + intersection.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(intersection)
    threshold = torch.finfo(dtype).eps * eigenvalues.abs().max()
    valid = torch.nonzero(eigenvalues > threshold, as_tuple=False).flatten()
    if valid.numel() == 0:
        raise ValueError("Nyström sampled intersection has no positive eigenvalue")
    valid = valid[-min(rank, valid.numel()) :]
    storage_dtype = dtype if storage_dtype is None else storage_dtype
    basis = torch.empty(
        dimension,
        valid.numel(),
        device=device,
        dtype=storage_dtype,
    )
    for column, eigen_index in enumerate(valid.tolist()):
        vector = sampled_columns @ eigenvectors[:, eigen_index]
        basis[:, column].copy_(vector.mul(eigenvalues[eigen_index].rsqrt()))
    if not torch.isfinite(basis).all():
        raise RuntimeError("Nyström basis is non-finite")
    return NystromState(basis=basis, damping=damping, indices=selected)
