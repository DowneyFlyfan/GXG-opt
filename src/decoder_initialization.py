"""Numerically stable GPT-style initialization for local decoder baselines."""

from __future__ import annotations

import math

import torch

from torch import nn


def initialize_decoder_transformer(model: nn.Module) -> None:
    """Initialize a tied-embedding decoder without saturated initial logits."""
    blocks = getattr(model, "blocks", ())
    residual_scale = 0.02 / math.sqrt(2.0 * max(len(blocks), 1))
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for block in blocks:
            nn.init.normal_(block.projection.weight, mean=0.0, std=residual_scale)
            nn.init.normal_(block.feedforward_out.weight, mean=0.0, std=residual_scale)
        nn.init.normal_(model.pos_embedding, mean=0.0, std=0.01)
