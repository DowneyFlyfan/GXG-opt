from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

import torch

from .geometry import (
    SharedSpectralBasis,
    fit_projected_curvature,
    reduced_schatten_direction,
    rollout_geometry_candidates,
    schatten_direction,
)


@dataclass
class SpectralPolicyConfig:
    lr: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.01
    matrix_scale: str | float = "moonlight"
    candidates: tuple[float, ...] = (2.0, 4.0, 8.0, math.inf)
    transform_backend: str = "exact_svd"
    ns_steps: int = 8
    polynomial_degree: int = 12
    polynomial_floor: float = 1e-4
    eps: float = 1e-8
    warmup_steps: int = 500
    policy_interval: int = 100
    horizon: int = 4
    horizon_weights: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)
    switch_penalty: float = 0.01
    switch_margin: float = 0.005
    compute_penalty: float = 0.0
    history_size: int = 8
    basis_rank: int = 8
    fit_interval: int = 50
    ridge: float = 1e-4
    min_curvature: float = 0.0
    max_curvature: float = 100.0
    perpendicular_curvature: float | str = "median_secant"
    initial_p: float = math.inf
    manual_schedule: tuple[tuple[int, float], ...] = ()
    fallback_betas: tuple[float, float] = (0.9, 0.95)
    fallback_eps: float = 1e-8
    fallback_lr: float | None = None


class MultiStepSpectralOptimizer(torch.optim.Optimizer):
    """Energy-matched spectral optimizer with a side-effect-free rollout policy."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        config: SpectralPolicyConfig | None = None,
    ) -> None:
        self.config = config or SpectralPolicyConfig()
        if len(self.config.horizon_weights) != self.config.horizon:
            raise ValueError("horizon_weights must contain horizon values")
        if not self.config.candidates:
            raise ValueError("at least one spectral candidate is required")
        if self.config.transform_backend not in {"exact_svd", "shared_svd", "reduced"}:
            raise ValueError("unknown spectral transform backend")
        defaults = {
            "lr": self.config.lr,
            "weight_decay": self.config.weight_decay,
            "use_spectral": None,
        }
        super().__init__(params, defaults)
        if self.config.fallback_lr is not None:
            for group in self.param_groups:
                if group["use_spectral"] is False:
                    group["lr"] = self.config.fallback_lr
        self.last_diagnostics: list[dict[str, object]] = []

    def _matrix_scale(self, parameter: torch.Tensor) -> float:
        if self.config.matrix_scale == "moonlight":
            return 0.2 * math.sqrt(max(parameter.shape))
        return float(self.config.matrix_scale)

    def _transform(self, momentum: torch.Tensor, p: float) -> tuple[torch.Tensor, float]:
        if self.config.transform_backend == "exact_svd":
            return schatten_direction(momentum, p, self.config.eps), 0.0
        if self.config.transform_backend == "reduced":
            return (
                reduced_schatten_direction(
                    momentum,
                    p,
                    ns_steps=self.config.ns_steps,
                    polynomial_degree=self.config.polynomial_degree,
                    polynomial_floor=self.config.polynomial_floor,
                    eps=self.config.eps,
                ),
                0.0,
            )
        return SharedSpectralBasis(momentum, self.config.eps).direction(momentum, p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.last_diagnostics = []
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                use_spectral = group["use_spectral"]
                use_spectral = parameter.ndim == 2 if use_spectral is None else bool(use_spectral)
                state = self.state[parameter]
                if not use_spectral:
                    self._adamw_step(parameter, state, lr, weight_decay)
                    continue
                if parameter.ndim != 2:
                    raise ValueError("spectral parameter groups may contain only matrices")
                if not state:
                    state.update(
                        {
                            "step": 0,
                            "momentum": torch.zeros_like(parameter, dtype=torch.float32),
                            "selected_p": self.config.initial_p
                            if self.config.initial_p in self.config.candidates
                            else self.config.candidates[-1],
                            "secant_steps": [],
                            "secant_changes": [],
                            "curvature_model": None,
                            "previous_gradient": None,
                            "last_parameter_delta": None,
                            "switches": 0,
                            "policy_seconds": 0.0,
                            "transform_seconds": 0.0,
                        }
                    )
                state["step"] += 1
                step = int(state["step"])
                gradient = parameter.grad.detach().float()
                if (
                    state["previous_gradient"] is not None
                    and state["last_parameter_delta"] is not None
                ):
                    state["secant_steps"].append(state["last_parameter_delta"].clone())
                    state["secant_changes"].append(
                        gradient - state["previous_gradient"]
                    )
                    if len(state["secant_steps"]) > self.config.history_size:
                        state["secant_steps"].pop(0)
                        state["secant_changes"].pop(0)
                state["momentum"].mul_(self.config.momentum).add_(
                    gradient, alpha=1.0 - self.config.momentum
                )
                if step % self.config.fit_interval == 0 and state["secant_steps"]:
                    state["curvature_model"] = fit_projected_curvature(
                        state["secant_steps"],
                        state["secant_changes"],
                        basis_rank=self.config.basis_rank,
                        ridge=self.config.ridge,
                        min_curvature=self.config.min_curvature,
                        max_curvature=self.config.max_curvature,
                        perpendicular_curvature=self.config.perpendicular_curvature,
                    )
                old_p = float(state["selected_p"])
                proposal = old_p
                scores = {old_p: 0.0}
                increments = {old_p: []}
                drift = {old_p: 0.0}
                policy_seconds = 0.0
                scheduled = [
                    float(candidate)
                    for schedule_step, candidate in self.config.manual_schedule
                    if step >= schedule_step
                ]
                if scheduled:
                    if scheduled[-1] not in self.config.candidates:
                        raise ValueError("manual schedule values must be configured candidates")
                    state["selected_p"] = scheduled[-1]
                    proposal = scheduled[-1]
                    state["switches"] += int(proposal != old_p)
                elif (
                    len(self.config.candidates) > 1
                    and step >= self.config.warmup_steps
                    and step % self.config.policy_interval == 0
                ):
                    policy_started = time.perf_counter()
                    scale = self._matrix_scale(parameter)
                    scores, increments, drift = rollout_geometry_candidates(
                        gradient,
                        state["momentum"],
                        state["curvature_model"],
                        self.config.candidates,
                        learning_rates=[lr] * self.config.horizon,
                        horizon_weights=self.config.horizon_weights,
                        beta=self.config.momentum,
                        matrix_scale=scale,
                        selected_p=old_p,
                        switch_penalty=self.config.switch_penalty,
                        compute_penalties={
                            candidate: self.config.compute_penalty
                            for candidate in self.config.candidates
                        },
                        backend=self.config.transform_backend,
                        ns_steps=self.config.ns_steps,
                        polynomial_degree=self.config.polynomial_degree,
                        polynomial_floor=self.config.polynomial_floor,
                        eps=self.config.eps,
                    )
                    proposal = min(scores, key=scores.get)
                    if scores[old_p] - scores[proposal] > self.config.switch_margin:
                        state["selected_p"] = proposal
                        state["switches"] += int(proposal != old_p)
                    policy_seconds = time.perf_counter() - policy_started
                    state["policy_seconds"] += policy_seconds
                transform_started = time.perf_counter()
                transform, actual_drift = self._transform(
                    state["momentum"], float(state["selected_p"])
                )
                transform_seconds = time.perf_counter() - transform_started
                state["transform_seconds"] += transform_seconds
                direction = self._matrix_scale(parameter) * transform
                displacement = -lr * direction - lr * weight_decay * parameter.float()
                parameter.add_(displacement.to(parameter.dtype))
                state["last_parameter_delta"] = displacement.detach().clone()
                state["previous_gradient"] = gradient.detach().clone()
                model = state["curvature_model"]
                self.last_diagnostics.append(
                    {
                        "step": step,
                        "selected_p": float(state["selected_p"]),
                        "proposed_p": proposal,
                        "candidate_scores": scores,
                        "horizon_increments": increments,
                        "switches": int(state["switches"]),
                        "curvature_eigenvalues": []
                        if model is None
                        else model["eigenvalues"].detach().cpu().tolist(),
                        "secant_fit_residual": None
                        if model is None
                        else float(model["fit_residual"]),
                        "basis_rank": 0 if model is None else model["basis"].shape[1],
                        "explained_secant_energy": None
                        if model is None
                        else float(model["explained_energy"]),
                        "rollout_basis_drift": drift,
                        "actual_basis_drift": actual_drift,
                        "policy_seconds": policy_seconds,
                        "transform_seconds": transform_seconds,
                    }
                )
        return loss

    def _adamw_step(
        self,
        parameter: torch.Tensor,
        state: dict,
        lr: float,
        weight_decay: float,
    ) -> None:
        beta1, beta2 = self.config.fallback_betas
        if not state:
            state.update(
                {
                    "step": 0,
                    "exp_avg": torch.zeros_like(parameter, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros_like(parameter, dtype=torch.float32),
                }
            )
        state["step"] += 1
        gradient = parameter.grad.detach().float()
        state["exp_avg"].mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(
            gradient, gradient, value=1.0 - beta2
        )
        m_hat = state["exp_avg"] / (1.0 - beta1 ** state["step"])
        v_hat = state["exp_avg_sq"] / (1.0 - beta2 ** state["step"])
        direction = m_hat / (torch.sqrt(v_hat) + self.config.fallback_eps)
        parameter.mul_(1.0 - lr * weight_decay)
        parameter.add_(direction.to(parameter.dtype), alpha=-lr)
