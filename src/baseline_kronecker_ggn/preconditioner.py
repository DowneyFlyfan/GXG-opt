from __future__ import annotations

from torch import Tensor

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator


def baseline_direction(operator: KroneckerSpectralOperator, gradient: Tensor) -> Tensor:
    return -operator.apply_inverse(gradient)
