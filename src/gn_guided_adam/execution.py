from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

from .types import FunctionalBatch


@contextmanager
def preserve_execution_state(model: nn.Module):
    modes = [(module, module.training) for module in model.modules()]
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    python_rng = random.getstate()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        import numpy as np

        numpy_rng = np.random.get_state()
    except ImportError:  # pragma: no cover
        np = None
        numpy_rng = None
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.train(training)
        with torch.no_grad():
            live_buffers = dict(model.named_buffers())
            for name, value in buffers.items():
                live_buffers[name].copy_(value)
        random.setstate(python_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        if np is not None and numpy_rng is not None:
            np.random.set_state(numpy_rng)


def functional_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {name: parameter for name, parameter in model.named_parameters()}
    state.update({name: value.detach().clone() for name, value in model.named_buffers()})
    return state


@torch.no_grad()
def candidate_loss(
    model: nn.Module,
    batch: FunctionalBatch,
    direction: dict[str, torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
    weight_decay_names: set[str] | None = None,
) -> float:
    parameters = dict(model.named_parameters())
    if set(direction) != {name for name, parameter in parameters.items() if parameter.requires_grad}:
        raise ValueError("Candidate direction must cover every trainable parameter")
    state: dict[str, Any] = {}
    decay_names = set(direction) if weight_decay_names is None else weight_decay_names
    for name, parameter in parameters.items():
        if parameter.requires_grad:
            decay = weight_decay if name in decay_names else 0.0
            state[name] = parameter.detach() * (1 - lr * decay) + direction[name].to(parameter.dtype)
        else:
            state[name] = parameter
    state.update({name: value.detach().clone() for name, value in model.named_buffers()})
    with preserve_execution_state(model):
        output = functional_call(model, state, batch.args, dict(batch.kwargs), strict=False)
        return float(batch.loss_fn(output).detach().float().item())


@torch.no_grad()
def current_loss(model: nn.Module, batch: FunctionalBatch) -> float:
    state = functional_model_state(model)
    with preserve_execution_state(model):
        output = functional_call(model, state, batch.args, dict(batch.kwargs), strict=False)
        return float(batch.loss_fn(output).detach().float().item())
