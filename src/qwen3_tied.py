"""Temporary split-leaf forwards for Qwen's physically tied embeddings."""

from __future__ import annotations

import torch
from torch.func import functional_call


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
