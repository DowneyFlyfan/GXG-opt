from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

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
) -> dict[str, torch.optim.Optimizer]:
    if optimizer == "adamw":
        return {"adamw": torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))}
    if optimizer not in {"muon", "stiefel_muon", "hybrid_stiefel_muon"}:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    if optimizer == "hybrid_stiefel_muon":
        if stiefel_lr is None or stiefel_lr <= 0:
            raise ValueError("Hybrid Stiefel-Muon requires a positive stiefel_lr")
        edge_names, middle_names = hybrid_muon_parameter_names(model)
        named = dict(model.named_parameters())
        edge = [named[name] for name in edge_names]
        middle = [named[name] for name in middle_names]
        selected_ids = {id(parameter) for parameter in (*edge, *middle)}
        auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in selected_ids]
        return {
            "muon_edge": Muon(edge, lr=lr, weight_decay=weight_decay),
            "stiefel_muon_middle": StiefelMuon(middle, lr=stiefel_lr),
            "adamw_aux": torch.optim.AdamW(
                auxiliary, lr=auxiliary_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
            ),
        }
    selected = muon_parameter_names(model)
    muon_parameters = [parameter for name, parameter in model.named_parameters() if name in selected]
    selected_ids = {id(parameter) for parameter in muon_parameters}
    auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in selected_ids]
    if optimizer == "stiefel_muon":
        matrix_optimizer: torch.optim.Optimizer = StiefelMuon(muon_parameters, lr=lr)
    else:
        matrix_optimizer = Muon(muon_parameters, lr=lr, weight_decay=weight_decay)
    return {
        optimizer: matrix_optimizer,
        "adamw_aux": torch.optim.AdamW(auxiliary, lr=auxiliary_lr, weight_decay=weight_decay, betas=(0.9, 0.95)),
    }
