"""Paper-faithful inner optimization primitives for full Gauss-Newton."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from kronecker_ggn_common.curvature_operator import (
    FunctionalCurvatureBatch,
    GGNFullOperator,
)
from optimizers import Muon, muon_parameter_names


class PaperMuon(Muon):
    """PyTorch equivalent of the Optax Muon used by the paper code."""

    @staticmethod
    def scaled_lr(lr: float, rows: int, columns: int) -> float:
        return lr * math.sqrt(max(1.0, rows / columns))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta = group["momentum"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.reshape(parameter.shape[0], -1)
                state = self.state[parameter]
                momentum = state.setdefault(
                    "momentum_buffer", torch.zeros_like(gradient)
                )
                step_count = int(state.get("step", 0)) + 1
                state["step"] = step_count
                momentum.mul_(beta).add_(gradient, alpha=1.0 - beta)
                if group["nesterov"]:
                    corrected_momentum = momentum / (1.0 - beta ** (step_count + 1))
                    corrected_gradient = gradient / (1.0 - beta**step_count)
                    update = beta * corrected_momentum + (1.0 - beta) * corrected_gradient
                else:
                    update = momentum / (1.0 - beta**step_count)
                update = self.orthogonalize(update, group["ns_steps"]).reshape_as(parameter)
                rows, columns = gradient.shape
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(
                    update,
                    alpha=-self.scaled_lr(group["lr"], rows, columns),
                )
        return loss


def build_paper_inner_optimizers(
    model: nn.Module, *, learning_rate: float, weight_decay: float
) -> dict[str, torch.optim.Optimizer]:
    """Match the paper's Muon matrices and AdamW auxiliary parameters."""
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Learning rate must be positive and decay non-negative")
    selected_names = muon_parameter_names(model)
    muon_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name in selected_names
    ]
    selected_ids = {id(parameter) for parameter in muon_parameters}
    auxiliary_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in selected_ids
    ]
    return {
        "muon": PaperMuon(
            muon_parameters,
            lr=learning_rate,
            weight_decay=0.0,
            momentum=0.95,
        ),
        "adamw_aux": torch.optim.AdamW(
            auxiliary_parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        ),
    }


@dataclass(frozen=True)
class LineSearchResult:
    step_size: float
    loss: float
    candidate_losses: tuple[tuple[float, float], ...]


def _named_trainable_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _matching_parameters(
    reference: nn.Module, inner: nn.Module
) -> tuple[tuple[str, nn.Parameter, nn.Parameter], ...]:
    reference_named = _named_trainable_parameters(reference)
    inner_by_name = dict(_named_trainable_parameters(inner))
    if tuple(name for name, _ in reference_named) != tuple(inner_by_name):
        raise ValueError("Reference and inner models must have identical trainable parameters")
    matched = tuple(
        (name, parameter, inner_by_name[name])
        for name, parameter in reference_named
    )
    if any(reference_parameter.shape != inner_parameter.shape for _, reference_parameter, inner_parameter in matched):
        raise ValueError("Reference and inner parameter shapes must match")
    return matched


def _parameter_delta(reference: nn.Module, inner: nn.Module) -> Tensor:
    matched = _matching_parameters(reference, inner)
    dtype = torch.float64 if matched[0][1].dtype == torch.float64 else torch.float32
    return torch.cat(
        tuple(
            (inner_parameter.detach() - reference_parameter.detach())
            .reshape(-1)
            .to(device=reference_parameter.device, dtype=dtype)
            for _, reference_parameter, inner_parameter in matched
        )
    )


def quadratic_gradient(
    reference: nn.Module,
    inner: nn.Module,
    batch: FunctionalCurvatureBatch,
) -> Tensor:
    """Return ``g0 + J0.T H_loss J0 (inner - reference)`` for one batch."""
    operator = GGNFullOperator(reference, batch)
    delta = _parameter_delta(reference, inner)
    return operator.gradient() + operator.matvec(delta)


def _assign_flat_gradients(model: nn.Module, gradient: Tensor) -> None:
    parameters = tuple(parameter for _, parameter in _named_trainable_parameters(model))
    expected = sum(parameter.numel() for parameter in parameters)
    if gradient.numel() != expected:
        raise ValueError("Quadratic gradient has the wrong number of elements")
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.grad = (
            gradient[offset : offset + size]
            .reshape_as(parameter)
            .to(device=parameter.device, dtype=parameter.dtype)
            .clone()
        )
        offset += size


def inner_cosine_multiplier(inner_step: int, inner_steps: int) -> float:
    if inner_steps <= 0:
        raise ValueError("inner_steps must be positive")
    if not 0 <= inner_step < inner_steps:
        raise ValueError("inner_step must be within the current inner cycle")
    return 0.5 * (1.0 + math.cos(math.pi * inner_step / inner_steps))


def paper_inner_step(
    reference: nn.Module,
    inner: nn.Module,
    batch: FunctionalCurvatureBatch,
    optimizers: dict[str, torch.optim.Optimizer],
    base_learning_rates: dict[str, float],
    *,
    inner_step: int,
    inner_steps: int,
    gradient_clip: float,
) -> float:
    """Take one Muon/AdamW step on the Gauss-Newton quadratic model."""
    return paper_accumulated_inner_step(
        reference,
        inner,
        (batch,),
        optimizers,
        base_learning_rates,
        inner_step=inner_step,
        inner_steps=inner_steps,
        gradient_clip=gradient_clip,
    )


def paper_accumulated_inner_step(
    reference: nn.Module,
    inner: nn.Module,
    batches: tuple[FunctionalCurvatureBatch, ...],
    optimizers: dict[str, torch.optim.Optimizer],
    base_learning_rates: dict[str, float],
    *,
    inner_step: int,
    inner_steps: int,
    gradient_clip: float,
) -> float:
    """Average the full quadratic gradient before one inner optimizer step."""
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    if not batches:
        raise ValueError("At least one inner batch is required")
    if optimizers.keys() != base_learning_rates.keys():
        raise ValueError("Each optimizer requires one base learning rate")
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    gradient = None
    for batch in batches:
        batch_gradient = quadratic_gradient(reference, inner, batch)
        gradient = batch_gradient if gradient is None else gradient.add(batch_gradient)
    assert gradient is not None
    gradient.div_(len(batches))
    gradient_norm = float(gradient.norm().item())
    _assign_flat_gradients(inner, gradient)
    multiplier = inner_cosine_multiplier(inner_step, inner_steps)
    for name, optimizer in optimizers.items():
        for group in optimizer.param_groups:
            group["lr"] = base_learning_rates[name] * multiplier
        group_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )
        torch.nn.utils.clip_grad_norm_(group_parameters, gradient_clip)
        optimizer.step()
    return gradient_norm


def line_search_step_sizes(search_range: int) -> tuple[float, ...]:
    if search_range <= 0:
        raise ValueError("search_range must be positive")
    return tuple(1.0 / math.sqrt(2.0) ** index for index in range(search_range))


def _assign_candidate(
    matched: tuple[tuple[str, nn.Parameter, nn.Parameter], ...], step_size: float
) -> None:
    with torch.no_grad():
        for _, outer_parameter, inner_parameter in matched:
            outer_parameter.lerp_(inner_parameter, step_size)


def _restore_parameters(
    matched: tuple[tuple[str, nn.Parameter, nn.Parameter], ...],
    originals: tuple[Tensor, ...],
) -> None:
    with torch.no_grad():
        for (_, outer_parameter, _), original in zip(matched, originals, strict=True):
            outer_parameter.copy_(original)


def _average_true_loss(
    model: nn.Module, batches: tuple[FunctionalCurvatureBatch, ...]
) -> float:
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            output = model(*batch.args, **batch.kwargs)
            total += float(batch.loss_fn(output).item())
    return total / len(batches)


def held_out_line_search(
    outer: nn.Module,
    inner: nn.Module,
    batches: tuple[FunctionalCurvatureBatch, ...],
    *,
    search_range: int,
) -> LineSearchResult:
    """Select the paper's best true-loss step on entirely held-out batches."""
    if not batches:
        raise ValueError("Held-out line search requires at least one batch")
    matched = _matching_parameters(outer, inner)
    originals = tuple(parameter.detach().clone() for _, parameter, _ in matched)
    losses: list[tuple[float, float]] = []
    try:
        for step_size in line_search_step_sizes(search_range):
            _restore_parameters(matched, originals)
            _assign_candidate(matched, step_size)
            losses.append((step_size, _average_true_loss(outer, batches)))
        best_step, best_loss = min(losses, key=lambda item: item[1])
        _restore_parameters(matched, originals)
        _assign_candidate(matched, best_step)
    except BaseException:
        _restore_parameters(matched, originals)
        raise
    return LineSearchResult(best_step, best_loss, tuple(losses))
