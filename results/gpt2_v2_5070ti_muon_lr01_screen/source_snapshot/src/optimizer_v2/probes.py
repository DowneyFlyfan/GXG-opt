from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.func import functional_call

from .linalg import finite


def lm_loss(logits, ids):
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1)
    )


@contextmanager
def diagnostic_mode(model):
    modes = [(module, module.training) for module in model.modules()]
    devices = [next(model.parameters()).device.index] if next(model.parameters()).is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.enable_grad(), torch.autocast(
        next(model.parameters()).device.type, enabled=False
    ):
        model.eval()
        try:
            yield
        finally:
            for module, mode in modes:
                module.training = mode


def functional_logits(model, parameters, ids, *, tie_weights=True):
    return functional_call(model, parameters, (), {"input_ids": ids, "use_cache": False},
                           tie_weights=tie_weights).logits


def collect_factors(model, ids, names, counter=None):
    """autograd.grad on output activations never overwrites training gradients."""
    modules = dict(model.named_modules())
    captured = {}
    handles = []
    for name in names:
        def hook(module, inputs, output, name=name):
            captured[name] = (inputs[0], output)
        handles.append(modules[name].register_forward_hook(hook))
    try:
        with diagnostic_mode(model):
            if counter is not None:
                counter(forward=1)
            loss = lm_loss(model(input_ids=ids, use_cache=False).logits, ids)
            finite(loss)
            if counter is not None:
                counter(backward=1)
            errors = torch.autograd.grad(loss, [captured[name][1] for name in names])
            factors = {
                name: (captured[name][0].detach().reshape(-1, captured[name][0].shape[-1]),
                       error.detach().reshape(-1, error.shape[-1]))
                for name, error in zip(names, errors)
            }
            for values in factors.values():
                finite(*values)
            return factors
    finally:
        for handle in handles:
            handle.remove()


def collect_head_probe(model, ids, block, head, row_count, generator, *, need_error, counter=None):
    if ids.shape[0] != 1:
        raise ValueError("head probes use one complete sequence at a time")
    attention = model.transformer.h[block].attn
    captured = {}
    def input_hook(module, inputs):
        captured["x"] = inputs[0].detach()[0]
    def output_hook(module, inputs, output):
        captured["output"] = output[0]
    handles = [attention.c_attn.register_forward_pre_hook(input_hook),
               attention.register_forward_hook(output_hook)]
    try:
        with diagnostic_mode(model):
            if counter is not None:
                counter(forward=1)
            if need_error:
                loss = lm_loss(model(input_ids=ids, use_cache=False).logits, ids)
                finite(loss)
                if counter is not None:
                    counter(backward=1)
                error, = torch.autograd.grad(loss, captured["output"])
                error = error.detach()[0]
            else:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                error = torch.zeros_like(captured["output"][0])
    finally:
        for handle in handles:
            handle.remove()
    x = captured["x"]
    width = model.config.n_embd
    head_width = width // model.config.n_head
    head_slice = slice(head * head_width, (head + 1) * head_width)
    with torch.no_grad():
        projected = x @ attention.c_attn.weight + attention.c_attn.bias
        q, k, v = (piece[:, head_slice] for piece in projected.split(width, dim=-1))
        finite(x, q, k, v, error)
        rows = torch.randperm(len(x) - 1, generator=generator)[:row_count] + 1
        return {
            "x": x, "q": q, "k": k, "v": v,
            "o": attention.c_proj.weight[head_slice].detach(),
            "error": error, "rows": rows.to(x.device),
        }


def split_path_forward(model, ids, embedding_in, embedding_out):
    parameters = {name: value.detach() for name, value in model.named_parameters()}
    parameters["transformer.wte.weight"] = embedding_in
    parameters["lm_head.weight"] = embedding_out
    return functional_logits(model, parameters, ids, tie_weights=False)


def sample_output_covariance_probe(probabilities, generator):
    finite(probabilities)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    choices = torch.multinomial(flat.detach().cpu(), 1, generator=generator).to(flat.device)
    probe = -flat.detach().clone()
    probe.scatter_add_(1, choices, torch.ones_like(choices, dtype=probe.dtype))
    return (probe / len(flat) ** 0.5).reshape_as(probabilities)


def paired_embedding_sketch(model, ids, count, generator, counter=None):
    with diagnostic_mode(model):
        embedding = model.transformer.wte.weight
        input_leaf = embedding.detach().clone().requires_grad_()
        output_leaf = embedding.detach().clone().requires_grad_()
        if counter is not None:
            counter(forward=1)
        logits = split_path_forward(model, ids, input_leaf, output_leaf)[:, :-1]
        finite(logits)
        probabilities = logits.detach().softmax(-1)
        columns, diagnostics = [], []
        for index in range(count):
            probe = sample_output_covariance_probe(probabilities, generator)
            if counter is not None:
                counter(vjp=1)
            left, right = torch.autograd.grad((logits * probe).sum(), (input_leaf, output_leaf),
                                              retain_graph=index + 1 < count)
            finite(left, right)
            columns.append((left.detach() + right.detach()) / count ** 0.5)
            diagnostics.append({"input_norm": float(left.norm()), "output_norm": float(right.norm()),
                                "paired_inner_product": float((left * right).sum())})
    return columns, diagnostics
