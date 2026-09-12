from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from low_spectral_variance import (
    LowSpectralVariance,
    low_spectral_variance_parameter_names,
)
from srip_band import SRIPBand, srip_band_parameter_names
from effective_rank_half import (
    EffectiveRankHalf,
    EffectiveRankJointNewton,
    EffectiveRankLinear,
    EffectiveRankLinearJointNewton,
    EffectiveRankThird,
    effective_rank,
)
from spectral_sphere_muon import SpectralSphereMuon
from stiefel_muon import StiefelMuon


@torch.compile
def _compiled_zero_power(update: torch.Tensor, steps: int) -> torch.Tensor:
    matrix = update.bfloat16()
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / (matrix.norm() + 1e-7)
    for _ in range(steps):
        gram = matrix @ matrix.T
        matrix = 3.4445 * matrix + (-4.775 * gram + 2.0315 * (gram @ gram)) @ matrix
    return matrix.T if transposed else matrix


@dataclass(frozen=True)
class Qualification:
    ratio: float
    qualified: bool


def qualify_ratio(adamw_seconds: float, muon_seconds: float) -> Qualification:
    ratio = muon_seconds / adamw_seconds
    return Qualification(ratio=ratio, qualified=ratio <= 1.2)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @staticmethod
    def scaled_lr(lr: float, rows: int, columns: int) -> float:
        return lr * 0.2 * math.sqrt(max(rows, columns))

    @staticmethod
    def orthogonalize(update: torch.Tensor, steps: int) -> torch.Tensor:
        return _compiled_zero_power(update, steps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.reshape(parameter.shape[0], -1)
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(gradient))
                buffer.mul_(group["momentum"]).add_(gradient)
                update = gradient.add(buffer, alpha=group["momentum"]) if group["nesterov"] else buffer
                update = self.orthogonalize(update, group["ns_steps"]).reshape_as(parameter)
                rows, columns = gradient.shape
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-self.scaled_lr(group["lr"], rows, columns))
        return loss


class Muown(torch.optim.Optimizer):
    """Muon with optimizer-internal row gains updated by Adam.

    The effective matrix is ``Diag(g / ||R||_row) R``.  ``R`` receives the
    tangent Muon direction while ``g`` receives Adam, so callers continue to
    own and use the ordinary effective weight parameter.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1.0e-8,
    ) -> None:
        if lr <= 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("Muown hyperparameters are invalid")
        super().__init__(
            params,
            dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                betas=betas,
                eps=eps,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.reshape(parameter.shape[0], -1)
                state = self.state[parameter]
                direction = state.setdefault("direction", parameter.detach().clone().reshape_as(gradient))
                row_norm = state.setdefault("direction_row_norm", direction.norm(dim=1).clamp_min(group["eps"]))
                gain = state.setdefault("gain", row_norm.clone())
                unit_direction = direction / row_norm.unsqueeze(1)

                gain_gradient = (gradient * unit_direction).sum(dim=1)
                tangent_gradient = (gradient - gain_gradient.unsqueeze(1) * unit_direction) * (
                    gain / row_norm
                ).unsqueeze(1)
                momentum_buffer = state.setdefault("momentum_buffer", torch.zeros_like(direction))
                momentum_buffer.mul_(group["momentum"]).add_(tangent_gradient)
                directional_update = (
                    tangent_gradient.add(momentum_buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )
                directional_update = Muon.orthogonalize(directional_update, group["ns_steps"])
                rows, columns = direction.shape
                direction.add_(directional_update, alpha=-Muon.scaled_lr(group["lr"], rows, columns))

                gain.mul_(1 - group["lr"] * group["weight_decay"])
                gain_exp_avg = state.setdefault("gain_exp_avg", torch.zeros_like(gain))
                gain_exp_avg_sq = state.setdefault("gain_exp_avg_sq", torch.zeros_like(gain))
                step = int(state.get("gain_step", 0)) + 1
                state["gain_step"] = step
                gain_exp_avg.lerp_(gain_gradient, 1 - beta1)
                gain_exp_avg_sq.lerp_(gain_gradient.square(), 1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                gain.addcdiv_(
                    gain_exp_avg,
                    gain_exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(group["eps"]),
                    value=-group["lr"] / bias_correction1,
                )
                gain.abs_().clamp_min_(group["eps"])
                row_norm.copy_(direction.norm(dim=1).clamp_min(group["eps"]))
                parameter.copy_((gain / row_norm).unsqueeze(1).mul(direction).reshape_as(parameter))
        return loss


def muon_parameter_names(model: nn.Module) -> set[str]:
    convolution_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    first_convolution = next((module for module in model.modules() if isinstance(module, convolution_types)), None)
    first_ids = {id(parameter) for parameter in first_convolution.parameters(recurse=False)} if first_convolution else set()
    embedding_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    ineligible_convolution_ids = set(first_ids)
    for module in model.modules():
        if not isinstance(module, convolution_types):
            continue
        columns = module.in_channels * math.prod(module.kernel_size) // module.groups
        if module.out_channels < 16 or columns < 16 or module.out_channels / columns > 4:
            ineligible_convolution_ids.update(id(parameter) for parameter in module.parameters(recurse=False))
    excluded = ("pos_embedding", "head", "classifier", "output_layer", "ctc_lo", "lm_head")
    selected = set()
    for name, parameter in model.named_parameters():
        if (
            parameter.ndim < 2
            or id(parameter) in ineligible_convolution_ids
            or id(parameter) in embedding_ids
            or any(fragment in name for fragment in excluded)
        ):
            continue
        selected.add(name)
    return selected


def hybrid_muon_parameter_names(model: nn.Module) -> tuple[set[str], set[str]]:
    """Split eligible Transformer matrices into edge-Muon and inner-Stiefel sets."""
    selected = muon_parameter_names(model)
    block_indices = sorted(
        {
            int(name.split(".", 2)[1])
            for name in selected
            if name.startswith("blocks.") and name.split(".", 2)[1].isdigit()
        }
    )
    if len(block_indices) < 3:
        raise ValueError("Hybrid Stiefel-Muon requires at least three Transformer blocks")
    edge_indices = {block_indices[0], block_indices[-1]}
    edge = {
        name
        for name in selected
        if name.startswith("blocks.") and int(name.split(".", 2)[1]) in edge_indices
    }
    middle = selected - edge
    return edge, middle


def build_optimizers(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    auxiliary_lr: float = 3e-4,
    stiefel_lr: float | None = None,
    stiefel_square_lr: float | None = None,
    stiefel_rectangular_lr: float | None = None,
    stiefel_nesterov: bool = False,
    low_spectral_condition_limit: float | None = None,
    low_spectral_dual_steps: int = 8,
    srip_rho: float | None = None,
    srip_dual_steps: int = 8,
    effective_rank_schedule_steps: int | None = None,
    effective_rank_momentum: float = 0.95,
) -> dict[str, torch.optim.Optimizer]:
    if optimizer == "adamw":
        return {"adamw": torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))}
    if optimizer not in {"muon", "muown", "effective_rank_half", "effective_rank_joint_newton", "effective_rank_third", "effective_rank_linear", "effective_rank_linear_joint_newton", "spectral_sphere_muon", "stiefel_muon", "hybrid_stiefel_muon", "low_spectral_variance", "srip_band"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if optimizer == "hybrid_stiefel_muon":
        if stiefel_lr is None or stiefel_lr <= 0:
            raise ValueError("Hybrid Stiefel-Muon requires a positive stiefel_lr")
        edge_names, middle_names = hybrid_muon_parameter_names(model)
        named = dict(model.named_parameters())
        edge = [named[name] for name in sorted(edge_names)]
        middle_square = [
            named[name]
            for name in sorted(middle_names)
            if named[name].shape[0] == named[name].shape[1]
        ]
        middle_rectangular = [
            named[name]
            for name in sorted(middle_names)
            if named[name].shape[0] != named[name].shape[1]
        ]
        for rate in (stiefel_square_lr, stiefel_rectangular_lr):
            if rate is not None and rate <= 0:
                raise ValueError("Geometry-specific Stiefel learning rates must be positive")
        middle = [*middle_square, *middle_rectangular]
        selected_ids = {id(parameter) for parameter in (*edge, *middle)}
        auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in selected_ids]
        stiefel_groups = [
            {
                "params": middle_square,
                "lr": stiefel_lr if stiefel_square_lr is None else stiefel_square_lr,
            },
            {
                "params": middle_rectangular,
                "lr": stiefel_lr
                if stiefel_rectangular_lr is None
                else stiefel_rectangular_lr,
            },
        ]
        return {
            "muon_edge": Muon(edge, lr=lr, weight_decay=weight_decay),
            "stiefel_muon_middle": StiefelMuon(
                stiefel_groups, lr=stiefel_lr, nesterov=stiefel_nesterov
            ),
            "adamw_aux": torch.optim.AdamW(
                auxiliary, lr=auxiliary_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
            ),
        }
    selected = muon_parameter_names(model)
    if optimizer == "low_spectral_variance":
        if low_spectral_condition_limit is None or low_spectral_condition_limit <= 1:
            raise ValueError("Low-Spectral-Variance requires a condition limit above one")
        selected = low_spectral_variance_parameter_names(
            model,
            condition_limit=low_spectral_condition_limit,
            candidates=selected,
        )
    if optimizer == "srip_band":
        if srip_rho is None or not 0 < srip_rho < 1:
            raise ValueError("SRIP-band requires rho strictly between zero and one")
        selected = srip_band_parameter_names(model, rho=srip_rho, candidates=selected)
    if optimizer in {"effective_rank_half", "effective_rank_joint_newton", "effective_rank_third", "effective_rank_linear", "effective_rank_linear_joint_newton"}:
        if not 0 <= effective_rank_momentum < 1:
            raise ValueError("effective-rank momentum must lie in [0, 1)")
        minimum_effective_rank = 0.5 if optimizer in {"effective_rank_half", "effective_rank_joint_newton"} else (1.0 / 3.0 if optimizer == "effective_rank_third" else 0.2)
        named = dict(model.named_parameters())
        selected = {
            name
            for name in selected
            if effective_rank(named[name].detach().reshape(named[name].shape[0], -1))
            >= minimum_effective_rank
        }
    muon_parameters = [parameter for name, parameter in model.named_parameters() if name in selected]
    selected_ids = {id(parameter) for parameter in muon_parameters}
    auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in selected_ids]
    if optimizer == "stiefel_muon":
        matrix_optimizer: torch.optim.Optimizer = StiefelMuon(muon_parameters, lr=lr)
    elif optimizer == "spectral_sphere_muon":
        matrix_optimizer = SpectralSphereMuon(muon_parameters, lr=lr)
    elif optimizer == "low_spectral_variance":
        matrix_optimizer = LowSpectralVariance(
            muon_parameters,
            lr=lr,
            condition_limit=low_spectral_condition_limit,
            dual_steps=low_spectral_dual_steps,
        )
    elif optimizer == "srip_band":
        matrix_optimizer = SRIPBand(
            muon_parameters, lr=lr, rho=srip_rho, dual_steps=srip_dual_steps
        )
    elif optimizer == "muown":
        matrix_optimizer = Muown(muon_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "effective_rank_half":
        matrix_optimizer = EffectiveRankHalf(
            muon_parameters, lr=lr, weight_decay=weight_decay,
            momentum=effective_rank_momentum,
        )
    elif optimizer == "effective_rank_joint_newton":
        matrix_optimizer = EffectiveRankJointNewton(
            muon_parameters, lr=lr, weight_decay=weight_decay,
            momentum=effective_rank_momentum,
        )
    elif optimizer == "effective_rank_third":
        matrix_optimizer = EffectiveRankThird(
            muon_parameters, lr=lr, weight_decay=weight_decay,
            momentum=effective_rank_momentum,
        )
    elif optimizer == "effective_rank_linear":
        matrix_optimizer = EffectiveRankLinear(
            muon_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=effective_rank_momentum,
            schedule_steps=36_250 if effective_rank_schedule_steps is None else effective_rank_schedule_steps,
        )
    elif optimizer == "effective_rank_linear_joint_newton":
        matrix_optimizer = EffectiveRankLinearJointNewton(
            muon_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=effective_rank_momentum,
            schedule_steps=36_250 if effective_rank_schedule_steps is None else effective_rank_schedule_steps,
        )
    else:
        matrix_optimizer = Muon(muon_parameters, lr=lr, weight_decay=weight_decay)
    return {
        optimizer: matrix_optimizer,
        "adamw_aux": torch.optim.AdamW(auxiliary, lr=auxiliary_lr, weight_decay=weight_decay, betas=(0.9, 0.95)),
    }
