from __future__ import annotations

from torch import Tensor

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator


class RelativeResidualOperator:
    """E = M^-1/2 (G + damping I) M^-1/2 - I."""

    def __init__(
        self, layer_id: str, kron_operator: KroneckerSpectralOperator, ggn_operator
    ) -> None:
        self.layer_id = layer_id
        self.kron_operator = kron_operator
        self.ggn_operator = ggn_operator
        self.matvec_count = 0

    def matvec(self, vector: Tensor) -> Tensor:
        whitened = self.kron_operator.apply_inverse_sqrt(vector)
        if hasattr(self.ggn_operator, "matvec"):
            try:
                ggn_product = self.ggn_operator.matvec(self.layer_id, whitened)
            except TypeError:
                ggn_product = self.ggn_operator.matvec(whitened)
        else:
            ggn_product = self.ggn_operator(whitened)
        damped = ggn_product + self.kron_operator.damping * whitened
        result = self.kron_operator.apply_inverse_sqrt(damped) - vector
        self.matvec_count += 1
        return result
