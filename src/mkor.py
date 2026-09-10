"""Momentum-enabled Kronecker preconditioning with rank-one inverse updates."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _mean_feature_vector(value: Tensor) -> Tensor:
    if value.ndim < 1:
        raise ValueError("MKOR hook value must have a feature dimension")
    return value.detach().float().reshape(-1, value.shape[-1]).mean(dim=0)


def eligible_linear_modules(model: nn.Module) -> set[str]:
    """Return non-output linear layers whose factor states fit the model scope."""
    excluded = ("head", "lm_head", "output", "classifier", "embedding")
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and module.weight.ndim == 2
        and not any(part in name for part in excluded)
    }


@dataclass
class MKORHookState:
    output_features: int
    input_features: int
    device: torch.device
    left_inverse: Tensor | None = None
    right_inverse: Tensor | None = None
    activation: Tensor | None = None
    output_gradient: Tensor | None = None

    def __post_init__(self) -> None:
        self.left_inverse = torch.eye(
            self.output_features, device=self.device, dtype=torch.float32
        )
        self.right_inverse = torch.eye(
            self.input_features, device=self.device, dtype=torch.float32
        )

    @staticmethod
    def _stabilize(inverse: Tensor, threshold: float, mix: float) -> Tensor:
        if threshold <= 0 or not 0 <= mix <= 1:
            raise ValueError("invalid MKOR stabilization controls")
        if torch.linalg.matrix_norm(inverse, ord=float("inf")) > threshold:
            return mix * inverse + (1.0 - mix) * torch.eye(
                inverse.shape[0], device=inverse.device, dtype=inverse.dtype
            )
        return inverse

    @staticmethod
    def _rank_one_inverse_update(inverse: Tensor, vector: Tensor, decay: float) -> Tensor:
        if not 0 < decay <= 1:
            raise ValueError("MKOR decay must lie in (0, 1]")
        identity = torch.eye(inverse.shape[0], device=inverse.device, dtype=inverse.dtype)
        decayed_inverse = decay * inverse + (1.0 - decay) * identity
        transformed = decayed_inverse @ vector
        denominator = 1.0 + torch.dot(vector, transformed)
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("MKOR rank-one inverse denominator is non-positive")
        result = decayed_inverse - torch.outer(transformed, transformed) / denominator
        if not torch.isfinite(result).all():
            raise RuntimeError("MKOR inverse update is non-finite")
        return result

    @torch.no_grad()
    def update(
        self,
        *,
        output_gradient: Tensor,
        activation: Tensor,
        decay: float,
        stabilizer_threshold: float,
        stabilizer_mix: float,
        statistic_clip: float = 100.0,
    ) -> None:
        if statistic_clip <= 0:
            raise ValueError("MKOR statistic clip must be positive")
        if output_gradient.shape != (self.output_features,):
            raise ValueError("output gradient shape does not match MKOR state")
        if activation.shape != (self.input_features,):
            raise ValueError("activation shape does not match MKOR state")
        assert self.left_inverse is not None and self.right_inverse is not None
        self.left_inverse = self._rank_one_inverse_update(
            self._stabilize(self.left_inverse, stabilizer_threshold, stabilizer_mix),
            output_gradient.to(self.left_inverse).clamp(-statistic_clip, statistic_clip),
            decay,
        )
        self.right_inverse = self._rank_one_inverse_update(
            self._stabilize(self.right_inverse, stabilizer_threshold, stabilizer_mix),
            activation.to(self.right_inverse).clamp(-statistic_clip, statistic_clip),
            decay,
        )


class MKOR(torch.optim.Optimizer):
    """MKOR matrix optimizer; call ``attach`` before model backward passes."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        factor_decay: float = 0.95,
        factor_update_frequency: int = 10,
        stabilizer_threshold: float = 2.0,
        stabilizer_mix: float = 0.1,
        statistic_clip: float = 100.0,
    ) -> None:
        if lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("invalid MKOR optimizer controls")
        if factor_update_frequency <= 0 or statistic_clip <= 0:
            raise ValueError("invalid MKOR factor controls")
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "factor_decay": factor_decay,
                "factor_update_frequency": factor_update_frequency,
                "stabilizer_threshold": stabilizer_threshold,
                "stabilizer_mix": stabilizer_mix,
                "statistic_clip": statistic_clip,
            },
        )
        self._states_by_parameter: dict[int, MKORHookState] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._steps = 0

    def attach(self, model: nn.Module) -> set[str]:
        """Capture linear activations and output gradients for selected layers."""
        selected = eligible_linear_modules(model)
        for name, module in model.named_modules():
            if name not in selected:
                continue
            state = MKORHookState(module.out_features, module.in_features, module.weight.device)
            self._states_by_parameter[id(module.weight)] = state

            def forward_hook(_module, inputs, _output, *, hook_state=state):
                hook_state.activation = _mean_feature_vector(inputs[0]).to(hook_state.device)

            def backward_hook(_module, _grad_input, grad_output, *, hook_state=state):
                if grad_output and grad_output[0] is not None:
                    hook_state.output_gradient = _mean_feature_vector(grad_output[0]).to(hook_state.device)

            self._handles.append(module.register_forward_hook(forward_hook))
            self._handles.append(module.register_full_backward_hook(backward_hook))
        return selected

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                state = self._states_by_parameter.get(id(parameter))
                if state is None or state.activation is None or state.output_gradient is None:
                    update = gradient.float()
                elif self._steps % group["factor_update_frequency"] == 0:
                    state.update(
                        output_gradient=state.output_gradient,
                        activation=state.activation,
                        decay=group["factor_decay"],
                        stabilizer_threshold=group["stabilizer_threshold"],
                        stabilizer_mix=group["stabilizer_mix"],
                        statistic_clip=group["statistic_clip"],
                    )
                    assert state.left_inverse is not None and state.right_inverse is not None
                    candidate = state.left_inverse @ gradient.float() @ state.right_inverse
                    update = candidate * (gradient.float().norm() / candidate.norm().clamp_min(torch.finfo(candidate.dtype).eps))
                else:
                    assert state.left_inverse is not None and state.right_inverse is not None
                    candidate = state.left_inverse @ gradient.float() @ state.right_inverse
                    update = candidate * (gradient.float().norm() / candidate.norm().clamp_min(torch.finfo(candidate.dtype).eps))
                buffer = self.state[parameter].setdefault("momentum", torch.zeros_like(update))
                buffer.mul_(group["momentum"]).add_(update)
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(buffer.to(parameter), alpha=-group["lr"])
        self._steps += 1
        return loss
