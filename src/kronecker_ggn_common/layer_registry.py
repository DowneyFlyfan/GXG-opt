from __future__ import annotations

from collections import defaultdict

from torch import nn

from .types import LayerInfo


class LayerRegistry:
    """Stable nn.Linear registry with explicit first-order fallback ownership."""

    def __init__(self, model: nn.Module, factor_update_frequency: int = 1) -> None:
        self.model = model
        self.factor_update_frequency = factor_update_frequency
        if factor_update_frequency <= 0:
            raise ValueError("factor_update_frequency must be positive")
        self.layers = self._build_layers()
        self._by_id = {layer.layer_id: layer for layer in self.layers}
        self._parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        supported_ids = {id(layer.weight) for layer in self.layers if layer.supported}
        self._fallback = tuple(
            (name, parameter, self._fallback_reason(name, parameter))
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in supported_ids
        )

    def _build_layers(self) -> tuple[LayerInfo, ...]:
        try:
            aliases = list(self.model.named_parameters(remove_duplicate=False))
        except TypeError:  # pragma: no cover - old PyTorch
            aliases = list(self.model.named_parameters())
        aliases_by_id: dict[int, list[str]] = defaultdict(list)
        for name, parameter in aliases:
            aliases_by_id[id(parameter)].append(name)
        layers = []
        for module_path, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            layer_id = module_path or "<root>"
            names = aliases_by_id[id(module.weight)]
            reason = None
            if not module.weight.requires_grad:
                reason = "frozen"
            elif len(names) != 1:
                reason = "tied_or_shared"
            layers.append(
                LayerInfo(
                    layer_id=layer_id,
                    module_path=module_path,
                    module=module,
                    weight=module.weight,
                    bias=module.bias,
                    matrix_shape=(module.out_features, module.in_features),
                    block_type="linear",
                    factor_update_frequency=self.factor_update_frequency,
                    correction_eligible=reason is None,
                    supported=reason is None,
                    fallback_reason=reason,
                    distributed_owner=None,
                )
            )
        return tuple(sorted(layers, key=lambda layer: layer.layer_id))

    def _fallback_reason(self, name: str, parameter: nn.Parameter) -> str:
        if not parameter.requires_grad:
            return "frozen"
        if parameter.ndim < 2:
            return "bias_or_vector"
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        module = dict(self.model.named_modules()).get(module_path, self.model)
        if isinstance(module, nn.Embedding):
            return "embedding"
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return "convolution_not_implemented"
        return "unsupported_or_tied_parameter"

    def by_id(self, layer_id: str) -> LayerInfo:
        try:
            return self._by_id[layer_id]
        except KeyError as error:
            raise KeyError(f"Unknown layer_id: {layer_id}") from error

    @property
    def supported(self) -> tuple[LayerInfo, ...]:
        return tuple(layer for layer in self.layers if layer.supported)

    @property
    def fallback_parameters(self) -> tuple[tuple[str, nn.Parameter, str], ...]:
        return self._fallback

    def parameter_name(self, parameter: nn.Parameter) -> str:
        return self._parameter_names[id(parameter)]

    def state_metadata(self) -> dict[str, dict[str, object]]:
        return {
            layer.layer_id: {
                "module_path": layer.module_path,
                "matrix_shape": layer.matrix_shape,
                "block_type": layer.block_type,
                "factor_update_frequency": layer.factor_update_frequency,
                "correction_eligible": layer.correction_eligible,
                "supported": layer.supported,
                "fallback_reason": layer.fallback_reason,
                "distributed_owner": layer.distributed_owner,
            }
            for layer in self.layers
        }
