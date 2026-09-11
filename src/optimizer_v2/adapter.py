from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path

import torch


def transformer_matrix_parameter_names(model: torch.nn.Module) -> list[str]:
    """Route hidden matrices without exposing names to optimizer controllers."""
    excluded_ids = set()
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        excluded_ids.update(id(parameter) for parameter in input_embeddings.parameters())
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        excluded_ids.update(id(parameter) for parameter in output_embeddings.parameters())
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            excluded_ids.update(id(parameter) for parameter in module.parameters())
    for attribute in ("classifier", "score", "pooler"):
        module = getattr(model, attribute, None)
        if module is not None:
            excluded_ids.update(id(parameter) for parameter in module.parameters())
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.ndim == 2
        and id(parameter) not in excluded_ids
        and "embedding" not in name.lower()
    ]


@dataclass
class Proposal:
    learning: torch.Tensor
    decay: torch.Tensor
    direction: torch.Tensor
    value: torch.Tensor
    next_state: dict


def muon_provenance() -> dict:
    module = importlib.import_module("torch.optim._muon")
    return {
        "torch": str(torch.__version__),
        "torch_revision": torch.version.git_version,
        "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "momentum": "M.lerp_(G, 1-beta); N=G.lerp(M, beta)",
        "nesterov": True,
        "ns_coefficients": [3.4445, -4.775, 2.0315],
        "ns_steps": 5,
        "ns_precision": "bfloat16",
        "eps": 1e-7,
        "adjust_lr_fn": "match_rms_adamw",
        "decay": "unadjusted scalar lr; decoupled",
        "grouping": "full fused hidden matrices; embeddings and vectors use AdamW",
    }


class ProposalAdapter:
    """Pinned stock optimizers plus a non-mutating proposal/one-commit boundary."""

    def __init__(self, model, *, adam_only=False, lr=3e-4, weight_decay=0.1,
                 betas=(0.9, 0.95), momentum=0.95):
        if torch.__version__.split("+")[0] != "2.11.0":
            raise RuntimeError("optimizer 2.0 is pinned to PyTorch 2.11.0")
        self.parameters = dict(model.named_parameters())
        self.matrix_names = set(transformer_matrix_parameter_names(model))
        self.optimizers = []
        self.owners = {}
        matrix = []
        decay, no_decay = [], []
        for name, parameter in self.parameters.items():
            if not parameter.requires_grad:
                continue
            if not adam_only and name in self.matrix_names:
                matrix.append(parameter)
            elif parameter.ndim >= 2:
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        if matrix:
            self.optimizers.append(torch.optim.Muon(
                matrix, lr=lr, weight_decay=weight_decay, momentum=momentum,
                nesterov=True, ns_steps=5, eps=1e-7,
                ns_coefficients=(3.4445, -4.775, 2.0315),
                adjust_lr_fn="match_rms_adamw",
            ))
        self.optimizers.append(torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=tuple(betas), eps=1e-8, foreach=False, fused=False,
        ))
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["peak_lr"] = lr
                for parameter in group["params"]:
                    self.owners[id(parameter)] = (optimizer, group)

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state):
        if len(state) != len(self.optimizers):
            raise ValueError("baseline optimizer layout changed")
        for optimizer, saved in zip(self.optimizers, state):
            optimizer.load_state_dict(saved)
        self.owners = {
            id(parameter): (optimizer, group)
            for optimizer in self.optimizers for group in optimizer.param_groups
            for parameter in group["params"]
        }

    def momentum(self, name):
        parameter = self.parameters[name]
        optimizer, _ = self.owners[id(parameter)]
        return optimizer.state.get(parameter, {}).get(
            "momentum_buffer", torch.zeros_like(parameter)
        )

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    @torch.no_grad()
    def propose(self, momentum_overrides=None):
        from torch.optim._muon import _adjust_lr, _zeropower_via_newtonschulz
        momentum_overrides = momentum_overrides or {}
        proposals = {}
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                continue
            optimizer, group = self.owners[id(parameter)]
            old = optimizer.state.get(parameter, {})
            gradient = parameter.grad
            lr, wd = float(group["lr"]), float(group["weight_decay"])
            value = parameter.detach().clone().mul_(1 - lr * wd)
            decay = parameter.detach() * (-lr * wd)
            if isinstance(optimizer, torch.optim.Muon):
                momentum = momentum_overrides.get(name)
                if momentum is None:
                    momentum = old.get("momentum_buffer", torch.zeros_like(parameter)).clone()
                    momentum.lerp_(gradient, 1 - group["momentum"])
                update = gradient.lerp(momentum, group["momentum"])
                polar = _zeropower_via_newtonschulz(
                    update, group["ns_coefficients"], group["ns_steps"], group["eps"]
                )
                scale = _adjust_lr(1.0, group["adjust_lr_fn"], parameter.shape)
                learning = torch.zeros_like(parameter).add_(polar, alpha=-lr * scale)
                direction = polar.to(parameter.dtype) * scale
                value.add_(polar, alpha=-lr * scale)
                next_state = {"momentum_buffer": momentum}
            else:
                beta1, beta2 = group["betas"]
                step = old.get("step", torch.tensor(0.0)).clone().add_(1)
                first = old.get("exp_avg", torch.zeros_like(parameter)).clone()
                second = old.get("exp_avg_sq", torch.zeros_like(parameter)).clone()
                first.lerp_(gradient, 1 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = (second.sqrt() / (1 - beta2 ** float(step)) ** 0.5).add_(group["eps"])
                step_size = lr / (1 - beta1 ** float(step))
                learning = torch.zeros_like(parameter).addcdiv_(first, denominator, value=-step_size)
                direction = first / denominator / (1 - beta1 ** float(step))
                value.addcdiv_(first, denominator, value=-step_size)
                next_state = {"step": step, "exp_avg": first, "exp_avg_sq": second}
            proposals[name] = Proposal(learning, decay, direction, value, next_state)
        return proposals

    @torch.no_grad()
    def commit(self, proposals, corrections=None):
        corrections = corrections or {}
        for name, proposal in proposals.items():
            parameter = self.parameters[name]
            parameter.copy_(proposal.value)
            if name in corrections:
                parameter.add_(corrections[name])
            optimizer, _ = self.owners[id(parameter)]
            optimizer.state[parameter] = proposal.next_state
