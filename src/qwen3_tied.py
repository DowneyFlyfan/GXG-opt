"""Temporary split-leaf forwards for Qwen's physically tied embeddings."""

from __future__ import annotations

import math

import torch
from torch.func import functional_call

from optimizer_v2.linalg import low_rank_prox


def split_qwen_tied_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    embedding_input: torch.Tensor,
    embedding_output: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Qwen with separate differentiable input/output embedding leaves.

    ``tie_weights=False`` is intentional: this functional call is solely an
    auxiliary differentiation device.  The trained Qwen model retains one
    physical tied parameter and receives one committed optimizer update.
    """
    if embedding_input.shape != embedding_output.shape:
        raise ValueError("split embedding leaves must have the same shape")
    parameters = {name: parameter.detach() for name, parameter in model.named_parameters()}
    parameters["model.embed_tokens.weight"] = embedding_input
    parameters["lm_head.weight"] = embedding_output
    output = functional_call(
        model,
        parameters,
        (),
        {"input_ids": input_ids, "use_cache": False},
        tie_weights=False,
    )
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("split Qwen forward did not return logits")
    return logits


def qwen_paired_embedding_sketch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    """Build the paired categorical-probe sketch for Qwen's tied embedding.

    The same probe is differentiated through the input and output paths.  The
    resulting columns therefore retain the cross terms of the joint output-loss
    generalized Gauss--Newton metric and are stored in FP32 for the small
    proximal solve.
    """
    if count <= 0 or input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("a paired sketch needs positive count and two-token sequences")
    embedding = model.model.embed_tokens.weight
    input_leaf = embedding.detach().clone().requires_grad_()
    output_leaf = embedding.detach().clone().requires_grad_()
    logits = split_qwen_tied_forward(model, input_ids, input_leaf, output_leaf)[:, :-1]
    probabilities = logits.detach().float().softmax(dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    columns: list[torch.Tensor] = []
    diagnostics: list[dict[str, float]] = []
    for index in range(count):
        choices = torch.multinomial(flat.cpu(), 1, generator=generator).to(flat.device)
        probe = -flat.clone()
        probe.scatter_add_(1, choices, torch.ones_like(choices, dtype=probe.dtype))
        probe = (probe / math.sqrt(len(flat))).reshape_as(logits).to(logits.dtype)
        left, right = torch.autograd.grad(
            (logits * probe).sum(),
            (input_leaf, output_leaf),
            retain_graph=index + 1 < count,
        )
        columns.append((left.detach().float() + right.detach().float()) / math.sqrt(count))
        diagnostics.append(
            {
                "input_norm": float(left.norm()),
                "output_norm": float(right.norm()),
                "paired_inner_product": float((left * right).sum()),
            }
        )
    return columns, diagnostics


@torch.no_grad()
def qwen_tied_proximal_correction(
    learning_increment: torch.Tensor,
    columns: list[torch.Tensor],
    *,
    rho: float,
) -> tuple[torch.Tensor, dict]:
    """Return the joint-sketch proximal correction before embedding decay.

    The direct zero-strength branch is intentional: it preserves the baseline
    floating-point trajectory rather than merely relying on a zero Woodbury
    coefficient after extra operations.
    """
    if rho < 0 or not columns or any(column.shape != learning_increment.shape for column in columns):
        raise ValueError("invalid tied-path proximal inputs")
    if rho == 0:
        return torch.zeros_like(learning_increment), {"kappa": 0.0, "gram_eigenvalues": []}
    filtered, diagnostics = low_rank_prox(learning_increment.float(), columns, rho)
    return (filtered - learning_increment.float()).to(learning_increment.dtype), diagnostics
