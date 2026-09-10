from __future__ import annotations

import math

import torch


Partition = list[tuple[int, int]]


def svd_polar(matrix: torch.Tensor) -> torch.Tensor:
    """Return the exact thin polar factor used as the small-matrix oracle."""
    if matrix.ndim != 2:
        raise ValueError("polar transforms require a matrix")
    if matrix.numel() == 0:
        return matrix.float().clone()
    u, _, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    return u @ vh


def newton_schulz_polar(
    matrix: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, float]:
    """Approximate the polar factor with stable FP32 Newton--Schulz steps."""
    if matrix.ndim != 2:
        raise ValueError("polar transforms require a matrix")
    if steps < 1:
        raise ValueError("steps must be positive")
    original_was_wide = matrix.shape[0] < matrix.shape[1]
    x = matrix.float().T if original_was_wide else matrix.float()
    scale = torch.linalg.vector_norm(x)
    if not bool(torch.isfinite(scale)) or float(scale) <= eps:
        return torch.zeros_like(matrix, dtype=torch.float32), 0.0
    x = x / (scale + eps)
    for _ in range(steps):
        gram = x.T @ x
        x = 1.5 * x - 0.5 * (x @ gram)
    identity = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    residual = float(
        torch.linalg.vector_norm(x.T @ x - identity) / math.sqrt(x.shape[1])
    )
    return (x.T if original_was_wide else x), residual


def atomic_row_intervals(rows: int, atom_rows: int) -> list[tuple[int, int]]:
    if rows < 1 or atom_rows < 1:
        raise ValueError("rows and atom_rows must be positive")
    intervals = [
        (start, min(start + atom_rows, rows)) for start in range(0, rows, atom_rows)
    ]
    if len(intervals) > 1 and intervals[-1][1] - intervals[-1][0] < atom_rows:
        intervals[-2] = (intervals[-2][0], intervals[-1][1])
        intervals.pop()
    return intervals


def row_slices(
    atoms: list[tuple[int, int]], partition: Partition
) -> list[slice]:
    slices = []
    expected = 0
    for start_atom, end_atom in partition:
        if start_atom != expected or end_atom <= start_atom or end_atom > len(atoms):
            raise ValueError("partition must contain every atom exactly once in order")
        slices.append(slice(atoms[start_atom][0], atoms[end_atom - 1][1]))
        expected = end_atom
    if expected != len(atoms):
        raise ValueError("partition does not cover all atoms")
    return slices


def block_polar(
    matrix: torch.Tensor,
    atoms: list[tuple[int, int]],
    partition: Partition,
    *,
    backend: str = "newton_schulz",
    ns_steps: int = 5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, float]:
    pieces = []
    residuals = []
    for rows in row_slices(atoms, partition):
        block = matrix[rows]
        if backend == "svd":
            pieces.append(svd_polar(block))
            residuals.append(0.0)
        elif backend == "newton_schulz":
            factor, residual = newton_schulz_polar(
                block, steps=ns_steps, eps=eps
            )
            pieces.append(factor)
            residuals.append(residual)
        else:
            raise ValueError(f"unsupported polar backend: {backend}")
    return torch.cat(pieces, dim=0), max(residuals, default=0.0)


def partition_norm_match(
    shape: tuple[int, int], atoms: list[tuple[int, int]], partition: Partition
) -> float:
    rows, columns = shape
    partition_energy = sum(
        min(row_slice.stop - row_slice.start, columns)
        for row_slice in row_slices(atoms, partition)
    )
    return math.sqrt(min(rows, columns) / max(partition_energy, 1))
