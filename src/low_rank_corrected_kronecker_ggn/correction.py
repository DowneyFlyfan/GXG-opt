from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator
from kronecker_ggn_common.vector_space import (
    orthogonality_error,
    project_frobenius,
    reconstruct_frobenius,
)


@dataclass(frozen=True)
class CorrectionApplication:
    direction: Tensor
    clipped_eigenvalues: Tensor
    clipped_count: int


def validate_correction_basis(
    basis: Tensor, eigenvalues: Tensor, tolerance: float = 1.0e-5
) -> None:
    if basis.ndim != 3 or eigenvalues.shape != (basis.shape[0],):
        raise ValueError("Correction basis/eigenvalue shapes are inconsistent")
    if not torch.isfinite(basis).all() or not torch.isfinite(eigenvalues).all():
        raise ValueError("Correction contains non-finite values")
    if orthogonality_error(basis) > tolerance:
        raise ValueError("Correction basis is not orthonormal")


def corrected_direction(
    operator: KroneckerSpectralOperator,
    gradient: Tensor,
    basis: Tensor,
    eigenvalues: Tensor,
    *,
    eigenvalue_margin: float,
    absolute_eigenvalue_cap: float | None,
    orthogonality_tolerance: float = 1.0e-5,
) -> CorrectionApplication:
    validate_correction_basis(basis, eigenvalues, orthogonality_tolerance)
    if tuple(basis.shape[1:]) != tuple(gradient.shape):
        raise ValueError("Correction basis and gradient matrix shapes differ")
    lower = -1.0 + eigenvalue_margin
    clipped = eigenvalues.clamp_min(lower)
    if absolute_eigenvalue_cap is not None:
        clipped = clipped.clamp(
            min=max(lower, -absolute_eigenvalue_cap), max=absolute_eigenvalue_cap
        )
    clipped_count = int((clipped != eigenvalues).sum().item())
    whitened = operator.apply_inverse_sqrt(gradient)
    basis = basis.to(device=whitened.device, dtype=whitened.dtype)
    clipped = clipped.to(device=whitened.device, dtype=whitened.dtype)
    coefficients = project_frobenius(basis, whitened)
    scale = clipped / (1.0 + clipped)
    corrected = whitened - reconstruct_frobenius(basis, scale * coefficients)
    direction = -operator.apply_inverse_sqrt(corrected)
    if not torch.isfinite(direction).all():
        raise FloatingPointError("Corrected inverse action is non-finite")
    return CorrectionApplication(direction, clipped, clipped_count)
