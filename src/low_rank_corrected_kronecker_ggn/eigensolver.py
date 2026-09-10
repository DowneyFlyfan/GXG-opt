from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from kronecker_ggn_common.vector_space import orthogonality_error


@dataclass(frozen=True)
class SignedEigenResult:
    basis: Tensor
    eigenvalues: Tensor
    residuals: Tensor
    requested_rank: int
    accepted_rank: int
    matvec_count: int
    orthogonality_error: float
    build_time_seconds: float
    rejected_count: int


def _orthogonalize(vector: Tensor, basis: list[Tensor], passes: int) -> Tensor:
    for _ in range(passes):
        for item in basis:
            vector = vector - torch.dot(item, vector) * item
    return vector


def signed_lanczos(
    matvec,
    matrix_shape: tuple[int, int],
    rank: int,
    *,
    steps: int,
    oversampling: int = 0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    reorthogonalization_passes: int = 2,
    tolerance: float = 1.0e-4,
    breakdown_tolerance: float = 1.0e-10,
) -> SignedEigenResult:
    started = time.perf_counter()
    if rank < 0 or steps <= 0 or oversampling < 0:
        raise ValueError("rank/oversampling must be nonnegative and steps positive")
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("signed Lanczos requires float32 or float64")
    count = matrix_shape[0] * matrix_shape[1]
    if rank == 0 or count == 0:
        empty_basis = torch.empty((0, *matrix_shape), device=device, dtype=dtype)
        empty = torch.empty(0, device=device, dtype=dtype)
        return SignedEigenResult(
            empty_basis, empty, empty, rank, 0, 0, 0.0, time.perf_counter() - started, 0
        )
    subspace_size = min(
        count, steps, rank + oversampling if rank + oversampling > 0 else rank
    )
    if subspace_size < rank:
        raise ValueError("Lanczos subspace must be at least the requested rank")
    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
    basis: list[Tensor] = []
    products: list[Tensor] = []
    candidate = torch.randn(count, generator=generator, device=device, dtype=dtype)
    while len(basis) < subspace_size:
        candidate = _orthogonalize(candidate, basis, reorthogonalization_passes)
        norm = candidate.norm()
        if not torch.isfinite(norm) or float(norm.item()) <= breakdown_tolerance:
            if len(basis) >= count:
                break
            candidate = torch.randn(
                count, generator=generator, device=device, dtype=dtype
            )
            candidate = _orthogonalize(candidate, basis, reorthogonalization_passes)
            norm = candidate.norm()
            if float(norm.item()) <= breakdown_tolerance:
                break
        current = candidate / norm
        product = (
            matvec(current.reshape(matrix_shape))
            .reshape(-1)
            .to(device=device, dtype=dtype)
        )
        if product.numel() != count or not torch.isfinite(product).all():
            raise FloatingPointError("Residual operator returned an invalid product")
        basis.append(current)
        products.append(product)
        candidate = _orthogonalize(product.clone(), basis, reorthogonalization_passes)
    if len(basis) < rank:
        raise RuntimeError("Lanczos breakdown before reaching the requested rank")
    q = torch.stack(basis, dim=1)
    aq = torch.stack(products, dim=1)
    reduced = 0.5 * (q.T @ aq + aq.T @ q)
    values, vectors = torch.linalg.eigh(reduced)
    order = torch.argsort(values.abs(), descending=True)[:rank]
    values = values[order]
    ritz = q @ vectors[:, order]
    residual_values = []
    accepted = []
    accepted_values = []
    matvec_count = len(products)
    for index in range(values.numel()):
        vector = ritz[:, index]
        product = matvec(vector.reshape(matrix_shape)).reshape(-1).to(vector)
        matvec_count += 1
        residual = (product - values[index] * vector).norm() / max(
            1.0, abs(float(values[index].item()))
        )
        residual_values.append(residual)
        if torch.isfinite(residual) and float(residual.item()) <= tolerance:
            accepted.append(vector)
            accepted_values.append(values[index])
    if accepted:
        accepted_matrix = torch.stack(accepted, dim=1)
        accepted_basis = accepted_matrix.T.reshape(-1, *matrix_shape)
        accepted_eigenvalues = torch.stack(accepted_values)
        accepted_residuals = torch.stack(
            [
                residual_values[index]
                for index in range(len(residual_values))
                if float(residual_values[index].item()) <= tolerance
            ]
        )
    else:
        accepted_basis = torch.empty((0, *matrix_shape), device=device, dtype=dtype)
        accepted_eigenvalues = torch.empty(0, device=device, dtype=dtype)
        accepted_residuals = torch.empty(0, device=device, dtype=dtype)
    return SignedEigenResult(
        basis=accepted_basis,
        eigenvalues=accepted_eigenvalues,
        residuals=accepted_residuals,
        requested_rank=rank,
        accepted_rank=accepted_basis.shape[0],
        matvec_count=matvec_count,
        orthogonality_error=orthogonality_error(accepted_basis),
        build_time_seconds=time.perf_counter() - started,
        rejected_count=rank - accepted_basis.shape[0],
    )
