from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .config import GNConfig
from .probes import dot, norm


Direction = dict[str, torch.Tensor]
LinearOperator = Callable[[Mapping[str, torch.Tensor]], Direction]


@dataclass(frozen=True)
class InnerSolveResult:
    direction: Direction
    state: dict[str, dict[str, torch.Tensor | int]]
    iterations: int
    residual_norm: float
    predicted_reduction: float
    finite: bool
    descent: bool


def _zero_power(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    original_dtype = matrix.dtype
    value = matrix.float()
    transposed = value.shape[0] > value.shape[1]
    if transposed:
        value = value.T
    value = value / value.norm().clamp_min(1.0e-7)
    for _ in range(steps):
        gram = value @ value.T
        value = 3.4445 * value + (-4.775 * gram + 2.0315 * (gram @ gram)) @ value
    if transposed:
        value = value.T
    return value.to(original_dtype)


def _solver_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _clone_direction(value: Mapping[str, torch.Tensor]) -> Direction:
    return {
        name: tensor.detach().to(dtype=_solver_dtype(tensor)).clone()
        for name, tensor in value.items()
    }


def _cg_solve(
    gradient: Mapping[str, torch.Tensor],
    operator: LinearOperator,
    warm_start: Mapping[str, torch.Tensor],
    config: GNConfig,
) -> InnerSolveResult:
    delta = {
        name: warm_start.get(name, torch.zeros_like(value)).detach().to(dtype=_solver_dtype(value)).clone()
        for name, value in gradient.items()
    }
    applied = operator(delta)
    residual = {
        name: -gradient[name].to(dtype=_solver_dtype(gradient[name]))
        - applied[name].to(dtype=_solver_dtype(gradient[name]))
        for name in gradient
    }
    search = _clone_direction(residual)
    residual_squared = float(dot(residual, residual).item())
    initial = math.sqrt(max(residual_squared, 0.0))
    iterations = 0
    for iterations in range(1, config.inner_steps + 1):
        product = operator(search)
        curvature = float(dot(search, product).item())
        if not math.isfinite(curvature) or curvature <= 0:
            break
        alpha = residual_squared / max(curvature, 1.0e-20)
        for name in delta:
            delta[name].add_(search[name], alpha=alpha)
            residual[name].add_(product[name], alpha=-alpha)
        next_squared = float(dot(residual, residual).item())
        if math.sqrt(max(next_squared, 0.0)) <= config.relative_residual_tolerance * max(initial, 1.0e-12):
            residual_squared = next_squared
            break
        beta = next_squared / max(residual_squared, 1.0e-20)
        for name in search:
            search[name].mul_(beta).add_(residual[name])
        residual_squared = next_squared
    return _result(gradient, operator, delta, {}, iterations, math.sqrt(max(residual_squared, 0.0)))


def _result(
    gradient: Mapping[str, torch.Tensor],
    operator: LinearOperator,
    delta: Direction,
    state: dict[str, dict[str, torch.Tensor | int]],
    iterations: int,
    residual_norm: float,
) -> InnerSolveResult:
    applied = operator(delta)
    predicted = -float(dot(gradient, delta).item()) - 0.5 * float(dot(delta, applied).item())
    finite = all(torch.isfinite(value).all().item() for value in delta.values()) and math.isfinite(predicted)
    descent = float(dot(gradient, delta).item()) < 0
    return InnerSolveResult(delta, state, iterations, residual_norm, predicted, finite, descent)


def solve_quadratic(
    gradient: Mapping[str, torch.Tensor],
    operator: LinearOperator,
    warm_start: Mapping[str, torch.Tensor],
    inner_state: Mapping[str, dict[str, torch.Tensor | int]],
    muon_names: set[str],
    config: GNConfig,
) -> InnerSolveResult:
    if config.inner_optimizer_matrix == "cg" and config.inner_optimizer_vector == "cg":
        return _cg_solve(gradient, operator, warm_start, config)
    delta = {
        name: warm_start.get(name, torch.zeros_like(value)).detach().to(dtype=_solver_dtype(value)).clone()
        for name, value in gradient.items()
    }
    states: dict[str, dict[str, torch.Tensor | int]] = {}
    for name, value in gradient.items():
        use_muon = name in muon_names and config.inner_optimizer_matrix == "muon"
        previous = inner_state.get(name, {})
        states[name] = {
            key: item.detach().to(dtype=_solver_dtype(value)).clone() if isinstance(item, torch.Tensor) else int(item)
            for key, item in previous.items()
        }
        if use_muon:
            states[name].setdefault("momentum", torch.zeros_like(value, dtype=_solver_dtype(value)))
        else:
            states[name].setdefault("m", torch.zeros_like(value, dtype=_solver_dtype(value)))
            states[name].setdefault("v", torch.zeros_like(value, dtype=_solver_dtype(value)))
            states[name].setdefault("step", 0)
    initial_residual = None
    residual_value = math.inf
    iterations = 0
    for iterations in range(1, config.inner_steps + 1):
        applied = operator(delta)
        residual = {
            name: gradient[name].to(dtype=_solver_dtype(gradient[name]))
            + applied[name].to(dtype=_solver_dtype(gradient[name]))
            for name in gradient
        }
        residual_value = norm(residual)
        initial_residual = residual_value if initial_residual is None else initial_residual
        if residual_value <= config.relative_residual_tolerance * max(initial_residual, 1.0e-12):
            break
        for name, value in residual.items():
            state = states[name]
            if name in muon_names and config.inner_optimizer_matrix == "muon":
                momentum = state["momentum"]
                assert isinstance(momentum, torch.Tensor)
                momentum.mul_(0.95).add_(value)
                update = value + 0.95 * momentum
                matrix = update.reshape(update.shape[0], -1)
                scaled_lr = config.inner_lr * 0.2 * math.sqrt(max(matrix.shape))
                delta[name].add_(_zero_power(matrix).reshape_as(value), alpha=-scaled_lr)
            else:
                m, v = state["m"], state["v"]
                assert isinstance(m, torch.Tensor) and isinstance(v, torch.Tensor)
                step = int(state["step"]) + 1
                state["step"] = step
                m.mul_(0.9).add_(value, alpha=0.1)
                v.mul_(0.999).addcmul_(value, value, value=0.001)
                update = (m / (1 - 0.9**step)) / ((v / (1 - 0.999**step)).sqrt() + 1.0e-8)
                delta[name].add_(update, alpha=-config.inner_lr)
    return _result(gradient, operator, delta, states, iterations, residual_value)
