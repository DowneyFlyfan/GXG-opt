"""Row and Column Scaled SGD from structured Fisher approximation."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


class RACS(torch.optim.Optimizer):
    """Two-sided diagonal Fisher approximation for matrix parameters."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        beta: float = 0.9,
        scale: float = 1.0,
        limiter: float = 1.01,
        fixed_point_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0 or not 0 <= beta < 1 or scale <= 0 or limiter <= 0:
            raise ValueError("invalid RACS optimizer controls")
        if fixed_point_steps <= 0 or weight_decay < 0:
            raise ValueError("invalid RACS state controls")
        super().__init__(params, dict(lr=lr, beta=beta, scale=scale, limiter=limiter, fixed_point_steps=fixed_point_steps, weight_decay=weight_decay))

    @staticmethod
    def _fixed_point(gradient: Tensor, steps: int) -> tuple[Tensor, Tensor]:
        rows, columns = gradient.shape
        q = torch.ones(rows, device=gradient.device, dtype=gradient.dtype)
        s = torch.ones(columns, device=gradient.device, dtype=gradient.dtype)
        eps = torch.finfo(gradient.dtype).eps
        squared = gradient.square()
        for _ in range(steps):
            s = (squared.T @ q).clamp_min(eps) / q.square().sum().clamp_min(eps)
            q = (squared @ s).clamp_min(eps) / s.square().sum().clamp_min(eps)
        return s, q

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("RACS only supports matrix parameters")
                gradient = parameter.grad.detach().float()
                instant_s, instant_q = self._fixed_point(gradient, group["fixed_point_steps"])
                state = self.state[parameter]
                s = state.setdefault("s", instant_s.clone())
                q = state.setdefault("q", instant_q.clone())
                s.mul_(group["beta"]).add_(instant_s, alpha=1.0 - group["beta"])
                q.mul_(group["beta"]).add_(instant_q, alpha=1.0 - group["beta"])
                scaled = q.rsqrt()[:, None] * gradient * s.rsqrt()[None, :]
                update = group["scale"] * scaled
                if "limited_norm" in state:
                    ratio = group["limiter"] / update.norm().clamp_min(group["limiter"])
                    update.mul_(ratio)
                state["limited_norm"] = update.norm()
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update.to(parameter), alpha=-group["lr"])
        return loss
