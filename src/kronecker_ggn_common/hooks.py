from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from typing import Self

import torch
from torch import Tensor

from .layer_registry import LayerRegistry


class LinearCapture(AbstractContextManager["LinearCapture"]):
    """Capture detached inputs and differentiable outputs for registered linear layers."""

    def __init__(self, registry: LayerRegistry) -> None:
        self.registry = registry
        self.activations: dict[str, list[Tensor]] = defaultdict(list)
        self.outputs: dict[str, list[Tensor]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> Self:
        for layer in self.registry.supported:

            def hook(_module, arguments, output, layer_id=layer.layer_id):
                if (
                    not arguments
                    or not isinstance(arguments[0], Tensor)
                    or not isinstance(output, Tensor)
                ):
                    raise RuntimeError(
                        f"Layer {layer_id} did not receive and return tensors"
                    )
                self.activations[layer_id].append(arguments[0].detach())
                self.outputs[layer_id].append(output)

            self._handles.append(layer.module.register_forward_hook(hook))
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def flattened_activations(self, layer_id: str) -> Tensor:
        values = self.activations.get(layer_id, [])
        if not values:
            raise RuntimeError(f"No activation was captured for {layer_id}")
        return torch.cat(
            [value.reshape(-1, value.shape[-1]) for value in values], dim=0
        )
