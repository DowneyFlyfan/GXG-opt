"""Damped layer-wise Kronecker-factored GGN optimizer."""

from kronecker_ggn_common.config import KroneckerGGNConfig

from .optimizer import KroneckerGGN, kronecker_ggn

__all__ = ["KroneckerGGN", "KroneckerGGNConfig", "kronecker_ggn"]
