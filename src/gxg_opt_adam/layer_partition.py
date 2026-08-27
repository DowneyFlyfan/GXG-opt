from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class LayerGroup:
    name: str
    parameter_names: tuple[str, ...]


class LayerPartitioner:
    """Build and validate complete, disjoint logical layer groups."""

    _BLOCK_PATTERN = re.compile(r"(?:^|\.)(blocks|layers|h|encoder\.layer)\.(\d+)(?:\.|$)")
    _HEAD_PARTS = {"head", "lm_head", "classifier", "output_layer", "ctc", "ctc_lo"}

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        try:
            aliases = list(model.named_parameters(remove_duplicate=False))
        except TypeError:  # pragma: no cover - for older supported torch versions
            aliases = list(model.named_parameters())
        self._aliases = aliases
        self._canonical_by_id: dict[int, str] = {}
        self._parameter_by_name = {}
        self._names_by_id: dict[int, list[str]] = defaultdict(list)
        for name, parameter in aliases:
            self._canonical_by_id.setdefault(id(parameter), name)
            self._parameter_by_name[name] = parameter
            self._names_by_id[id(parameter)].append(name)
        self._trainable = {
            canonical
            for parameter_id, canonical in self._canonical_by_id.items()
            if self._parameter_by_name[canonical].requires_grad
        }

    def automatic_transformer(self) -> tuple[LayerGroup, ...]:
        tentative = {name: self._automatic_name(name) for name, _ in self._aliases}
        block_names = {group for group in tentative.values() if group.startswith("block:")}
        if block_names:
            last = max(block_names, key=lambda name: (tuple(int(value) for value in re.findall(r"\d+", name)), name))
            for name, group in tuple(tentative.items()):
                if group == "final_norm":
                    tentative[name] = last
        else:
            for name, group in tuple(tentative.items()):
                if group == "final_norm":
                    tentative[name] = "root"
        return self._validate_alias_assignments(tentative)

    def from_explicit(self, mapping: Mapping[str, Sequence[str]]) -> tuple[LayerGroup, ...]:
        assignments: dict[str, str] = {}
        all_names = set(self._parameter_by_name)
        for group, names in mapping.items():
            if not names:
                raise ValueError(f"Layer group {group!r} is empty")
            for name in names:
                if name not in all_names:
                    raise ValueError(f"Unknown parameter in layer partition: {name}")
                if name in assignments:
                    raise ValueError(f"Parameter {name} appears in more than one layer group")
                assignments[name] = group
        return self._validate_alias_assignments(assignments)

    def from_regex_rules(self, rules: Mapping[str, Sequence[str]]) -> tuple[LayerGroup, ...]:
        compiled = {group: [re.compile(pattern) for pattern in patterns] for group, patterns in rules.items()}
        assignments: dict[str, str] = {}
        for name in self._parameter_by_name:
            matches = [group for group, patterns in compiled.items() if any(pattern.search(name) for pattern in patterns)]
            if len(matches) > 1:
                raise ValueError(f"Parameter {name} matches overlapping layer rules: {matches}")
            if matches:
                assignments[name] = matches[0]
        return self._validate_alias_assignments(assignments)

    def _automatic_name(self, parameter_name: str) -> str:
        parts = parameter_name.split(".")
        lowered = {part.lower() for part in parts}
        block = self._BLOCK_PATTERN.search(parameter_name)
        if block:
            prefix = parameter_name[: block.end()].rstrip(".")
            return f"block:{prefix}"
        if any("embedding" in part.lower() or part.lower() == "embed" for part in parts):
            return "embedding"
        if lowered & self._HEAD_PARTS:
            return "output_head"
        if any(part.lower() in {"norm", "ln_f", "final_norm"} for part in parts):
            return "final_norm"
        return f"root:{parts[0]}" if len(parts) > 1 else "root"

    def _validate_alias_assignments(self, assignments: Mapping[str, str]) -> tuple[LayerGroup, ...]:
        grouped: dict[str, list[str]] = defaultdict(list)
        missing: list[str] = []
        for parameter_id, canonical in self._canonical_by_id.items():
            parameter = self._parameter_by_name[canonical]
            if not parameter.requires_grad:
                continue
            aliases = self._names_by_id[parameter_id]
            alias_groups = {assignments[name] for name in aliases if name in assignments}
            if not alias_groups:
                missing.append(canonical)
                continue
            if len(alias_groups) != 1:
                raise ValueError(f"Tied parameter aliases assigned to different groups: {aliases}")
            grouped[alias_groups.pop()].append(canonical)
        if missing:
            raise ValueError(f"Trainable parameters missing from layer partition: {sorted(missing)}")
        covered = [name for names in grouped.values() for name in names]
        if len(covered) != len(set(covered)) or set(covered) != self._trainable:
            raise ValueError("Layer partition must cover each trainable parameter exactly once")
        return tuple(LayerGroup(name, tuple(sorted(names))) for name, names in sorted(grouped.items()))


def muon_eligible_parameter_names(model: nn.Module) -> set[str]:
    """Apply repository Muon routing rules, including grouped CNN matrix shape."""

    convolution_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    first_convolution = next((module for module in model.modules() if isinstance(module, convolution_types)), None)
    excluded_ids: set[int] = set()
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    if first_convolution is not None:
        excluded_ids.update(id(parameter) for parameter in first_convolution.parameters(recurse=False))
    for module in model.modules():
        if not isinstance(module, convolution_types):
            continue
        kernel_elements = 1
        for size in module.kernel_size:
            kernel_elements *= size
        columns = module.in_channels * kernel_elements // module.groups
        ratio = module.out_channels / columns
        if ratio > 4 or columns < 16 or module.out_channels < 16:
            excluded_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    head_fragments = ("head", "classifier", "output_layer", "ctc_lo", "lm_head")
    selected = set()
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or id(parameter) in excluded_ids or id(parameter) in embedding_ids:
            continue
        if any(fragment in name for fragment in head_fragments):
            continue
        selected.add(name)
    return selected
