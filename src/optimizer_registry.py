from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from baseline_kronecker_ggn import KroneckerGGN
from kronecker_ggn_common.config import (
    KroneckerGGNConfig,
    LowRankCorrectedKroneckerGGNConfig,
)
from low_rank_corrected_kronecker_ggn import LowRankCorrectedKroneckerGGN

OPTIMIZER_REGISTRY = {
    "kronecker_ggn": KroneckerGGN,
    "low_rank_corrected_kronecker_ggn": LowRankCorrectedKroneckerGGN,
}


def build_optimizer(name: str, model: nn.Module, config=None, **kwargs):
    """Build a curvature optimizer without changing the legacy AdamW/Muon harness."""
    try:
        optimizer_type = OPTIMIZER_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"Unsupported curvature optimizer: {name}") from error
    if isinstance(config, Mapping):
        config_type = (
            LowRankCorrectedKroneckerGGNConfig
            if name == "low_rank_corrected_kronecker_ggn"
            else KroneckerGGNConfig
        )
        config = config_type.from_dict(dict(config))
    return optimizer_type(model=model, config=config, **kwargs)
