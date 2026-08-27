from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


class KrylovError(RuntimeError):
    pass


@dataclass(frozen=True)
class KrylovResult:
    basis: torch.Tensor
    reduced_matrix: torch.Tensor
    rank: int
    matvecs: int
    orthogonality_error: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    build_time_seconds: float
    breakdown: bool


@dataclass(frozen=True)
class ReducedSolveResult:
    direction: torch.Tensor
    coordinates: torch.Tensor
    projected_gradient: torch.Tensor
    residual_norm: float
    method: str
    predicted_reduction: float


def build_krylov_basis(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    seed: torch.Tensor,
    rank: int,
    *,
    reorthogonalization_passes: int = 2,
    breakdown_tolerance: float = 1.0e-10,
    negative_eigenvalue_tolerance: float = 1.0e-6,
) -> KrylovResult:
    started = time.perf_counter()
    vector = seed.detach().reshape(-1)
    dtype = torch.float64 if vector.dtype == torch.float64 else torch.float32
    vector = vector.to(dtype=dtype)
    seed_norm = vector.norm()
    if not torch.isfinite(seed_norm) or seed_norm <= breakdown_tolerance:
        raise KrylovError("Krylov seed is zero or nonfinite")
    current = vector / seed_norm
    previous = torch.zeros_like(current)
    previous_beta = torch.tensor(0.0, device=current.device, dtype=current.dtype)
    basis_vectors: list[torch.Tensor] = []
    products: list[torch.Tensor] = []
    breakdown = False
    for _ in range(rank):
        basis_vectors.append(current)
        product = matvec(current).to(device=current.device, dtype=current.dtype)
        if not torch.isfinite(product).all():
            raise KrylovError("Krylov matrix-vector product is nonfinite")
        products.append(product)
        alpha = torch.dot(current, product)
        residual = product - alpha * current - previous_beta * previous
        for _ in range(reorthogonalization_passes):
            for basis_vector in basis_vectors:
                residual = residual - torch.dot(basis_vector, residual) * basis_vector
        beta = residual.norm()
        if not torch.isfinite(beta) or beta <= breakdown_tolerance:
            breakdown = True
            break
        previous, current, previous_beta = current, residual / beta, beta
    basis = torch.stack(basis_vectors, dim=1)
    product_matrix = torch.stack(products, dim=1)
    reduced = basis.T @ product_matrix
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(reduced)
    scale = max(float(eigenvalues.abs().max().item()), 1.0)
    minimum = float(eigenvalues.min().item())
    if minimum < -negative_eigenvalue_tolerance * scale:
        raise KrylovError(f"Reduced GGN has a significant negative eigenvalue: {minimum}")
    clamped = torch.clamp(eigenvalues, min=0)
    reduced = (eigenvectors * clamped.unsqueeze(0)) @ eigenvectors.T
    identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
    orthogonality = float((basis.T @ basis - identity).norm().item())
    return KrylovResult(
        basis=basis,
        reduced_matrix=reduced,
        rank=basis.shape[1],
        matvecs=len(products),
        orthogonality_error=orthogonality,
        minimum_eigenvalue=float(clamped.min().item()),
        maximum_eigenvalue=float(clamped.max().item()),
        build_time_seconds=time.perf_counter() - started,
        breakdown=breakdown,
    )


def reduced_gn_solve(
    basis: torch.Tensor,
    reduced_matrix: torch.Tensor,
    gradient: torch.Tensor,
    damping: float,
) -> ReducedSolveResult:
    if basis.ndim != 2 or reduced_matrix.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("Basis and reduced matrix shapes are inconsistent")
    flat_gradient = gradient.reshape(-1).to(device=basis.device, dtype=basis.dtype)
    projected = basis.T @ flat_gradient
    damped = reduced_matrix + damping * torch.eye(
        reduced_matrix.shape[0], device=reduced_matrix.device, dtype=reduced_matrix.dtype
    )
    cholesky, info = torch.linalg.cholesky_ex(damped)
    if int(info.max().item()) == 0:
        coordinates = torch.cholesky_solve((-projected).unsqueeze(1), cholesky).squeeze(1)
        method = "cholesky"
    else:
        eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (damped + damped.T))
        if not torch.isfinite(eigenvalues).all() or float(eigenvalues.min().item()) <= 0:
            raise KrylovError("Damped reduced solve is not positive definite")
        coordinates = eigenvectors @ ((eigenvectors.T @ -projected) / eigenvalues)
        method = "symmetric_eigh"
    residual = damped @ coordinates + projected
    direction = basis @ coordinates
    predicted = -float(torch.dot(flat_gradient, direction).item()) - 0.5 * float(
        torch.dot(coordinates, reduced_matrix @ coordinates).item()
    )
    if not torch.isfinite(direction).all() or not math.isfinite(predicted):
        raise KrylovError("Reduced solve produced a nonfinite result")
    return ReducedSolveResult(
        direction=direction,
        coordinates=coordinates,
        projected_gradient=projected,
        residual_norm=float(residual.norm().item()),
        method=method,
        predicted_reduction=predicted,
    )


def capture_metrics(reduced: ReducedSolveResult, oracle_direction: torch.Tensor, oracle_prediction: float) -> dict[str, float]:
    oracle = oracle_direction.reshape(-1).to(reduced.direction)
    denominator = reduced.direction.norm() * oracle.norm() + 1.0e-12
    return {
        "capture": reduced.predicted_reduction / (oracle_prediction + 1.0e-12),
        "cosine": float(torch.dot(reduced.direction, oracle).item() / denominator),
        "relative_error": float((reduced.direction - oracle).norm().item() / (oracle.norm().item() + 1.0e-12)),
    }
