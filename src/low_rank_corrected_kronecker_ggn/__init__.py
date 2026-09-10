"""Low-rank relative-residual correction of the Kronecker-GGN baseline."""

from kronecker_ggn_common.config import LowRankCorrectedKroneckerGGNConfig

from .optimizer import LowRankCorrectedKroneckerGGN, low_rank_corrected_kronecker_ggn

__all__ = [
    "LowRankCorrectedKroneckerGGN",
    "LowRankCorrectedKroneckerGGNConfig",
    "low_rank_corrected_kronecker_ggn",
]
