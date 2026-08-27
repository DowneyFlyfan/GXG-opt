from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

from .config import GNConfig


@dataclass
class BlockSpec:
    name: str
    parameters: list[nn.Parameter]
    numel: int
    module_path: str
    enabled: bool
    parameter_names: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    disabled_reason: str | None = None


class BlockRegistry:
    """One tensor per candidate block with tied-parameter safety."""

    _OUTPUT_FRAGMENTS = ("head", "lm_head", "classifier", "output_layer", "ctc_lo")

    def __init__(self, model: nn.Module, config: GNConfig) -> None:
        self.model = model
        self.config = config
        self.specs = self._build()
        enabled_ids = [id(spec.parameters[0]) for spec in self.specs if spec.enabled]
        if len(enabled_ids) != len(set(enabled_ids)):
            raise ValueError("A parameter belongs to more than one enabled guidance block")

    @property
    def enabled(self) -> tuple[BlockSpec, ...]:
        return tuple(spec for spec in self.specs if spec.enabled)

    def by_name(self) -> dict[str, BlockSpec]:
        return {spec.name: spec for spec in self.specs}

    def _build(self) -> tuple[BlockSpec, ...]:
        try:
            aliases = list(self.model.named_parameters(remove_duplicate=False))
        except TypeError:  # pragma: no cover
            aliases = list(self.model.named_parameters())
        names_by_id: dict[int, list[str]] = defaultdict(list)
        parameter_by_id: dict[int, nn.Parameter] = {}
        for name, parameter in aliases:
            names_by_id[id(parameter)].append(name)
            parameter_by_id[id(parameter)] = parameter
        modules = dict(self.model.named_modules())
        patterns = tuple(re.compile(pattern) for pattern in self.config.guided_block_patterns)
        specs = []
        for parameter_id, names in names_by_id.items():
            parameter = parameter_by_id[parameter_id]
            canonical = names[0]
            module_path = canonical.rsplit(".", 1)[0] if "." in canonical else ""
            module = modules.get(module_path, self.model)
            reason = self._disabled_reason(canonical, names, parameter, module, patterns)
            specs.append(
                BlockSpec(
                    name=canonical,
                    parameters=[parameter],
                    numel=parameter.numel(),
                    module_path=module_path,
                    enabled=reason is None,
                    parameter_names=(canonical,),
                    aliases=tuple(names),
                    disabled_reason=reason,
                )
            )
        return tuple(sorted(specs, key=lambda spec: spec.name))

    def _disabled_reason(
        self,
        name: str,
        aliases: list[str],
        parameter: nn.Parameter,
        module: nn.Module,
        patterns: tuple[re.Pattern[str], ...],
    ) -> str | None:
        if not parameter.requires_grad:
            return "frozen"
        if len(aliases) > 1:
            return "tied_or_shared"
        if parameter.layout != torch.strided:
            return "sparse_or_non_strided"
        if parameter.ndim < 2 or not name.endswith("weight"):
            return "vector_or_bias"
        if isinstance(module, (nn.Embedding, nn.LayerNorm)):
            return "embedding_or_normalization"
        if parameter.numel() < self.config.min_block_numel:
            return "too_small"
        if not self.config.include_output_projection and any(fragment in name for fragment in self._OUTPUT_FRAGMENTS):
            return "output_projection"
        if patterns and not any(pattern.search(name) for pattern in patterns):
            return "not_selected"
        return None
