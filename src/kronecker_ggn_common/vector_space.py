from __future__ import annotations

import torch
from torch import Tensor


def frobenius_inner(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError("Frobenius operands must have identical shapes")
    return (left * right).sum()


def frobenius_norm(value: Tensor) -> Tensor:
    return torch.sqrt(torch.clamp(frobenius_inner(value, value), min=0))


def normalize_frobenius(value: Tensor, epsilon: float = 1.0e-12) -> Tensor:
    norm = frobenius_norm(value)
    if not torch.isfinite(norm) or float(norm.item()) <= epsilon:
        raise ValueError("Cannot normalize a zero or non-finite matrix")
    return value / norm


def project_frobenius(basis: Tensor, value: Tensor) -> Tensor:
    if basis.ndim != value.ndim + 1 or tuple(basis.shape[1:]) != tuple(value.shape):
        raise ValueError("Basis must have shape [rank, *value.shape]")
    if basis.shape[0] == 0:
        return torch.empty(0, device=value.device, dtype=value.dtype)
    return torch.einsum("r...,...->r", basis, value)


def reconstruct_frobenius(basis: Tensor, coefficients: Tensor) -> Tensor:
    if basis.ndim < 2 or coefficients.shape != (basis.shape[0],):
        raise ValueError("Coefficient count must equal the basis rank")
    return torch.einsum("r,r...->...", coefficients, basis)


def orthonormalize_matrices(basis: Tensor, tolerance: float = 1.0e-10) -> Tensor:
    if basis.ndim != 3:
        raise ValueError("Matrix basis must have shape [rank, output_dim, input_dim]")
    if basis.shape[0] == 0:
        return basis.clone()
    columns = basis.reshape(basis.shape[0], -1).T
    q, r = torch.linalg.qr(columns, mode="reduced")
    diagonal = r.diagonal().abs()
    keep = diagonal > tolerance
    return q[:, keep].T.reshape(-1, *basis.shape[1:])


def orthogonality_error(basis: Tensor) -> float:
    if basis.shape[0] == 0:
        return 0.0
    flat = basis.reshape(basis.shape[0], -1)
    identity = torch.eye(flat.shape[0], device=flat.device, dtype=flat.dtype)
    return float((flat @ flat.T - identity).norm().item())
