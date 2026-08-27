from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .config import AdamConfig
from .layer_partition import LayerGroup
from .probes import mean_squared_gradients, norm


class AdamBackend:
    """AdamW candidate generation with an independent, shadow-updatable bank."""

    BIG_MODEL_PARAMETERS = 100_000_000

    def __init__(self, model: nn.Module, groups: tuple[LayerGroup, ...], config: AdamConfig) -> None:
        self.model = model
        self.groups = groups
        self.config = config
        self.lr = config.lr
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.state: dict[str, dict[str, Any]] = {
            name: {
                "m": torch.zeros_like(parameter, dtype=torch.float32),
                "v": torch.zeros_like(parameter, dtype=torch.float32),
                "step": 0.0,
                "variance_floor": torch.zeros_like(parameter, dtype=torch.float32),
            }
            for name, parameter in self.parameters.items()
        }
        self.recent_layer_norms = {group.name: deque(maxlen=32) for group in groups}
        self.weight_decay = config.weight_decay if sum(p.numel() for p in model.parameters()) >= self.BIG_MODEL_PARAMETERS else 0.0

    def gradients(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.grad.detach().float().clone()
            for name, parameter in self.parameters.items()
            if parameter.grad is not None
        }

    def capture_variance_floor(self) -> None:
        for state in self.state.values():
            state["variance_floor"] = state["v"].detach().clone()

    def update_moments(
        self,
        gradients: Mapping[str, torch.Tensor] | None = None,
        microbatch_gradients: Sequence[Mapping[str, torch.Tensor]] = (),
        adam_equivalent_batches: float = 1.0,
    ) -> None:
        if adam_equivalent_batches <= 0:
            raise ValueError("adam_equivalent_batches must be positive")
        gradients = dict(gradients or self.gradients())
        if set(gradients) != set(self.parameters):
            missing = set(self.parameters) - set(gradients)
            raise ValueError(f"Missing real gradients for Adam state update: {sorted(missing)}")
        squared = mean_squared_gradients(microbatch_gradients) if microbatch_gradients else {
            name: value.float().square() for name, value in gradients.items()
        }
        if set(squared) != set(self.parameters):
            raise ValueError("Microbatch gradients must cover every trainable parameter")
        beta1, beta2 = self.config.betas
        effective_beta1 = beta1**adam_equivalent_batches
        effective_beta2 = beta2**adam_equivalent_batches
        for name, gradient in gradients.items():
            state = self.state[name]
            state["m"].mul_(effective_beta1).add_(gradient.float(), alpha=1 - effective_beta1)
            state["v"].mul_(effective_beta2).add_(squared[name].float(), alpha=1 - effective_beta2)
            state["step"] += adam_equivalent_batches

    def candidate(self) -> dict[str, torch.Tensor]:
        beta1, beta2 = self.config.betas
        candidate = {}
        for name, parameter in self.parameters.items():
            state = self.state[name]
            step = float(state["step"])
            if step <= 0:
                candidate[name] = torch.zeros_like(parameter, dtype=torch.float32)
                continue
            variance = torch.maximum(
                state["v"], self.config.variance_floor_ratio * state["variance_floor"]
            )
            first = state["m"] / (1 - beta1**step)
            second = variance / (1 - beta2**step)
            update = -self.lr * first / (second.sqrt() + self.config.eps)
            if self.weight_decay:
                update.add_(parameter.detach().float(), alpha=-self.lr * self.weight_decay)
            candidate[name] = update
        return candidate

    @torch.no_grad()
    def apply(self, candidate: Mapping[str, torch.Tensor]) -> float:
        if set(candidate) != set(self.parameters):
            raise ValueError("Adam candidate must cover every trainable parameter")
        for name, update in candidate.items():
            self.parameters[name].add_(update.to(self.parameters[name].dtype))
        for group in self.groups:
            group_update = {name: candidate[name] for name in group.parameter_names}
            self.recent_layer_norms[group.name].append(norm(group_update))
        return norm(candidate)

    def step(self) -> tuple[dict[str, torch.Tensor], float]:
        self.update_moments()
        candidate = self.candidate()
        return candidate, self.apply(candidate)

    def reduce_learning_rate(self, factor: float) -> None:
        if not 0 < factor < 1:
            raise ValueError("Adam learning-rate reduction factor must be in (0, 1)")
        self.lr *= factor

    def recent_norm(self, group_name: str) -> float | None:
        values = self.recent_layer_norms[group_name]
        return sum(values) / len(values) if values else None

    def state_dict(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "state": self.state,
            "recent_layer_norms": {name: list(values) for name, values in self.recent_layer_norms.items()},
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if set(payload["state"]) != set(self.parameters):
            raise ValueError("Adam checkpoint parameters do not match the model")
        self.lr = float(payload["lr"])
        self.state = {}
        for name, saved in payload["state"].items():
            parameter = self.parameters[name]
            self.state[name] = {
                key: value.to(device=parameter.device, dtype=torch.float32)
                if isinstance(value, torch.Tensor)
                else float(value)
                for key, value in saved.items()
            }
        for name, values in payload["recent_layer_norms"].items():
            self.recent_layer_norms[name].clear()
            self.recent_layer_norms[name].extend(values)
        self.weight_decay = float(payload.get("weight_decay", self.weight_decay))
