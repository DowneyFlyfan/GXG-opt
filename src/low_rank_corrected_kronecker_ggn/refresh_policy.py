from __future__ import annotations

from collections.abc import Mapping

from kronecker_ggn_common.config import LowRankCorrectedKroneckerGGNConfig
from kronecker_ggn_common.layer_registry import LayerRegistry

from .state import CorrectionState


class CorrectionRefreshPolicy:
    def __init__(self, config: LowRankCorrectedKroneckerGGNConfig) -> None:
        self.config = config
        self.force_refresh = False

    def should_refresh(self, step: int) -> bool:
        return (
            self.config.correction_rank > 0
            and step >= self.config.correction_warmup_steps
            and (
                self.force_refresh
                or (step - self.config.correction_warmup_steps)
                % self.config.correction_refresh_interval
                == 0
            )
        )

    def active_layers(
        self, registry: LayerRegistry, states: Mapping[str, CorrectionState], step: int
    ) -> tuple[str, ...]:
        candidates = list(registry.supported)
        count = min(self.config.active_layer_count, len(candidates))
        if self.config.active_layer_policy == "fixed":
            ordered = candidates
        elif self.config.active_layer_policy == "rotating":
            offset = (step // self.config.correction_refresh_interval) % max(
                len(candidates), 1
            )
            ordered = candidates[offset:] + candidates[:offset]
        elif self.config.active_layer_policy == "largest_parameter_count":
            ordered = sorted(
                candidates, key=lambda item: item.weight.numel(), reverse=True
            )
        else:
            ordered = sorted(
                candidates,
                key=lambda item: max(
                    states[item.layer_id].diagnostics.get(
                        "largest_abs_eigenvalue", 0.0
                    ),
                    float(item.weight.numel()) * 1.0e-30,
                ),
                reverse=True,
            )
        return tuple(layer.layer_id for layer in ordered[:count])

    def report_decrease(self, predicted: float, realized: float) -> None:
        threshold = self.config.predicted_realized_ratio_threshold
        if threshold is None or predicted <= 0:
            return
        self.force_refresh = realized / predicted < threshold
