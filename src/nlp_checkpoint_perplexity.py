"""Evaluate saved GPT-2 checkpoints with token-weighted validation perplexity."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import torch

from data import wikitext_loaders
from gpt2_ppl_experiment import validation_nll
from models import create_nlp_model


def model_state_from_checkpoint(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    """Extract the common model-state payload used by historical NLP runs."""
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint does not contain a model state")
    return state


def write_recovery_record(
    output: Path,
    *,
    checkpoint: str,
    validation_batches: int,
    validation_nll: float,
) -> Path:
    """Persist one checkpoint's directly measured PPL without altering it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "checkpoint": checkpoint,
                "validation_batches": validation_batches,
                "validation_nll": validation_nll,
                "perplexity": math.exp(validation_nll),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output


def evaluate_checkpoint_perplexity(
    root: Path,
    checkpoint_path: Path,
    output: Path,
    *,
    validation_batches: int,
    batch_size: int,
    workers: int,
) -> Path:
    """Load one GPT-12x512 checkpoint and measure its held-out PPL on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint perplexity recovery requires CUDA")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_nlp_model("gpt_12x512")
    model.load_state_dict(model_state_from_checkpoint(checkpoint))
    device = torch.device("cuda")
    model.to(device)
    _, loader = wikitext_loaders(root, batch_size=batch_size, workers=workers)
    nll = validation_nll(model, loader, device, maximum_batches=validation_batches)
    del model
    torch.cuda.empty_cache()
    return write_recovery_record(
        output,
        checkpoint=str(checkpoint_path.relative_to(root)),
        validation_batches=validation_batches,
        validation_nll=nll,
    )
