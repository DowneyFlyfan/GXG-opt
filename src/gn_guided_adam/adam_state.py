from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .config import AdamWConfig
from .tensor_ops import map_norm, project, project_complement


class AdamStateBank:
    BIG_MODEL_PARAMETERS = 100_000_000

    def __init__(self, model: nn.Module, config: AdamWConfig) -> None:
        self.model = model
        self.config = config
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.state = {
            name: {
                "m": torch.zeros_like(parameter, dtype=torch.float32),
                "v": torch.zeros_like(parameter, dtype=torch.float32),
                "step": 0,
            }
            for name, parameter in self.parameters.items()
        }
        self.active_names: set[str] = set()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.weight_decay = config.weight_decay if parameter_count >= self.BIG_MODEL_PARAMETERS else 0.0

    def gradients(self) -> dict[str, torch.Tensor]:
        gradients = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in self.parameters.items()
            if parameter.grad is not None
        }
        return gradients

    def update(self, gradients: Mapping[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        gradients = dict(self.gradients() if gradients is None else gradients)
        unknown = set(gradients) - set(self.parameters)
        if unknown:
            raise ValueError(f"Adam gradients contain unknown parameters: {sorted(unknown)}")
        for name, gradient in gradients.items():
            if gradient.shape != self.parameters[name].shape:
                raise ValueError(f"Adam gradient shape does not match parameter {name}")
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Nonfinite real gradient for {name}")
        beta1, beta2 = self.config.betas
        for name, gradient in gradients.items():
            state = self.state[name]
            state["m"].mul_(beta1).add_(gradient.float(), alpha=1 - beta1)
            state["v"].mul_(beta2).addcmul_(gradient.float(), gradient.float(), value=1 - beta2)
            state["step"] += 1
        self.active_names = set(gradients)
        return gradients

    def _moments(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.state[name]
        step = state["step"]
        if step <= 0:
            raise RuntimeError("Adam candidate requested before a real-gradient state update")
        beta1, beta2 = self.config.betas
        return state["m"] / (1 - beta1**step), state["v"] / (1 - beta2**step)

    def candidate_for(self, name: str, lr: float, basis: torch.Tensor | None = None) -> torch.Tensor:
        first, second = self._moments(name)
        if basis is not None and basis.numel():
            first = project_complement(basis, first.reshape(-1)).reshape_as(first)
        preconditioned = first / (second.sqrt() + self.config.eps)
        if basis is not None and basis.numel():
            preconditioned = project_complement(basis, preconditioned.reshape(-1)).reshape_as(preconditioned)
        return -lr * preconditioned

    def ordinary_candidate(self, lr: float) -> dict[str, torch.Tensor]:
        candidate = {
            name: self.candidate_for(name, lr)
            if name in self.active_names
            else torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in self.parameters.items()
        }
        if not all(torch.isfinite(value).all().item() for value in candidate.values()):
            raise FloatingPointError("AdamW candidate is nonfinite")
        return candidate

    @torch.no_grad()
    def apply(self, direction: Mapping[str, torch.Tensor], lr: float) -> float:
        if set(direction) != set(self.parameters):
            raise ValueError("Applied direction must cover every trainable parameter")
        for name, parameter in self.parameters.items():
            if name not in self.active_names:
                continue
            parameter.mul_(1 - lr * self.weight_decay)
            parameter.add_(direction[name].to(device=parameter.device, dtype=parameter.dtype))
        return map_norm(direction)

    def remove_first_moment_subspace(self, name: str, basis: torch.Tensor, fraction: float) -> None:
        if fraction <= 0 or basis.numel() == 0:
            return
        moment = self.state[name]["m"]
        moment.sub_(fraction * project(basis, moment.reshape(-1)).reshape_as(moment))

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "weight_decay": self.weight_decay,
            "active_names": sorted(self.active_names),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        if set(payload["state"]) != set(self.parameters):
            raise ValueError("Adam checkpoint parameters do not match the current model")
        restored = {}
        for name, state in payload["state"].items():
            parameter = self.parameters[name]
            restored[name] = {
                "m": state["m"].to(device=parameter.device, dtype=torch.float32),
                "v": state["v"].to(device=parameter.device, dtype=torch.float32),
                "step": int(state["step"]),
            }
        self.state = restored
        self.weight_decay = float(payload["weight_decay"])
        self.active_names = set(payload.get("active_names", ()))
