"""Qwen3 model loading and the study's explicit optimizer routing."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from optimizers import Muon, Muown


QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"
_MATRIX_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


def qwen_checkpoint_path(root: Path) -> Path:
    return root / ".cache" / "huggingface" / "models" / "Qwen3-0.6B"


def load_qwen3_model(root: Path) -> nn.Module:
    """Load the requested Qwen model only from the project cache."""
    from transformers import AutoModelForCausalLM

    checkpoint = qwen_checkpoint_path(root)
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"missing {QWEN3_MODEL_ID} below {checkpoint}; download it into the project cache first"
        )
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )


def qwen_muon_parameter_names(model: nn.Module) -> set[str]:
    """Select only interior Qwen projection matrices for Muon-style updates."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) < 3:
        raise ValueError("Qwen routing requires at least three transformer layers")
    named = dict(model.named_parameters())
    selected: set[str] = set()
    for layer_index in range(1, len(layers) - 1):
        prefix = f"model.layers.{layer_index}."
        for suffix in _MATRIX_SUFFIXES:
            name = prefix + suffix
            parameter = named.get(name)
            if parameter is None:
                raise ValueError(f"Qwen layer is missing expected projection: {name}")
            if parameter.ndim != 2:
                raise ValueError(f"Qwen projection must be a matrix: {name}")
            selected.add(name)
    return selected


def build_qwen_optimizers(
    model: nn.Module,
    optimizer_name: str,
    *,
    learning_rate: float | None = None,
    direction_lr: float | None = None,
    gain_lr: float | None = None,
    auxiliary_lr: float,
    weight_decay: float,
) -> dict[str, torch.optim.Optimizer]:
    """Build matched AdamW, Muon, or two-rate Muown optimizer groups for Qwen."""
    if auxiliary_lr <= 0 or weight_decay < 0:
        raise ValueError("auxiliary_lr must be positive and weight_decay non-negative")
    if optimizer_name == "adamw":
        if learning_rate is None or learning_rate <= 0:
            raise ValueError("AdamW requires a positive learning_rate")
        return {
            "adamw": torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.95)
            )
        }
    selected_names = qwen_muon_parameter_names(model)
    named = dict(model.named_parameters())
    matrices = [named[name] for name in sorted(selected_names)]
    matrix_ids = {id(parameter) for parameter in matrices}
    auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in matrix_ids]
    if optimizer_name == "muon":
        if learning_rate is None or learning_rate <= 0:
            raise ValueError("Muon requires a positive learning_rate")
        matrix_optimizer: torch.optim.Optimizer = Muon(
            matrices, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "muown":
        if direction_lr is None or gain_lr is None or direction_lr <= 0 or gain_lr <= 0:
            raise ValueError("Muown requires positive independent direction_lr and gain_lr")
        matrix_optimizer = Muown(
            matrices,
            direction_lr=direction_lr,
            gain_lr=gain_lr,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"unsupported Qwen optimizer: {optimizer_name}")
    return {
        optimizer_name: matrix_optimizer,
        "adamw_aux": torch.optim.AdamW(
            auxiliary, lr=auxiliary_lr, weight_decay=weight_decay, betas=(0.9, 0.95)
        ),
    }
