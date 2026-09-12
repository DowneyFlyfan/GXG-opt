"""Matched GPT2-12x512 experiments measured in validation perplexity."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plot
import torch
import torch.nn.functional as functional

from artifacts import write_metric
from config import FormalTask
from models import parameter_count
from optimizers import build_optimizers
from training import _loaders, _model, configure_reproducibility


DISPLAY_NAMES = {
    "adamw": "AdamW",
    "muon": "Muon",
    "muown": "Muown",
    "effective_rank_linear": "Effective rank (0.2 to 0.8)",
    "effective_rank_linear_joint_newton": "Effective rank (0.2 to 0.8, joint Newton)",
}

# These settings are the previously completed GPT2-12x512 winners.  The
# common effective batch is 48; the PPL runner defaults to micro-batch 8 with
# six accumulated gradients, matching the effective-rank and Muown finals.
FORMAL_PPL_HYPERPARAMETERS = {
    "adamw": {"learning_rate": 1.5e-4, "weight_decay": 0.01, "auxiliary_lr": None},
    "muon": {"learning_rate": 2.5e-3, "weight_decay": 0.01, "auxiliary_lr": 5.0e-4},
    "muown": {"learning_rate": 5.0e-3, "weight_decay": 0.0, "auxiliary_lr": 3.0e-4},
    "effective_rank_linear": {
        "learning_rate": 1.25e-3,
        "weight_decay": 0.0,
        "auxiliary_lr": 3.0e-4,
    },
    "effective_rank_linear_joint_newton": {
        "learning_rate": 1.25e-3,
        "weight_decay": 0.0,
        "auxiliary_lr": 3.0e-4,
    },
}


@dataclass(frozen=True)
class PPLTrialPaths:
    metric: Path
    result: Path
    checkpoint: Path


def ppl_trial_paths(root: Path, optimizer: str, *, run_label: str) -> PPLTrialPaths:
    """Keep recoverable states in ``.cache`` and compact evidence outside it."""
    stem = f"nlp_gpt_12x512__{run_label}__{optimizer}"
    return PPLTrialPaths(
        metric=root / "metrics" / "nlp" / f"{stem}.ppl.jsonl",
        result=root / "results" / "nlp" / f"{stem}.ppl.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.ppl.checkpoint.pt",
    )


def perplexity_from_nll(validation_nll: float) -> float:
    """Convert the mean next-token negative log likelihood to perplexity."""
    return math.exp(validation_nll)


@torch.no_grad()
def validation_nll(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    maximum_batches: int,
) -> float:
    """Return the token-weighted next-token negative log likelihood."""
    if maximum_batches <= 0:
        raise ValueError("maximum_batches must be positive")
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        token_ids, targets = (value.to(device, non_blocking=True) for value in batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(token_ids)
            logits = output.logits if hasattr(output, "logits") else output
        total_nll += float(
            functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum"
            )
        )
        total_tokens += targets.numel()
    if total_tokens == 0:
        raise RuntimeError("validation loader produced no tokens")
    return total_nll / total_tokens


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    *,
    completed_epoch: int,
    completed_updates: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            "completed_epoch": completed_epoch,
            "completed_updates": completed_updates,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def _restore_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
) -> tuple[int, int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(state["schedulers"][name])
    return (
        int(state["completed_epoch"]),
        int(state["completed_updates"]),
        float(state["elapsed_seconds"]),
    )


def _optimizer_diagnostics(
    optimizer_name: str, optimizers: dict[str, torch.optim.Optimizer]
) -> dict[str, float | int]:
    """Expose the effective-rank scheduler and its finite-step decisions."""
    if optimizer_name not in {"effective_rank_linear", "effective_rank_linear_joint_newton"}:
        return {}
    optimizer = optimizers[optimizer_name]
    accepted = skipped = projected = 0
    projection_frobenius = 0.0
    joint_newton = unconstrained = fallback = 0
    for state in optimizer.state.values():
        accepted += int(state.get("accepted_steps", 0))
        skipped += int(state.get("skipped_steps", 0))
        projected += int(state.get("projected_steps", 0))
        projection_frobenius += float(state.get("projection_frobenius", 0.0))
        joint_newton += int(state.get("joint_newton_steps", 0))
        unconstrained += int(state.get("unconstrained_steps", 0))
        fallback += int(state.get("certified_fallback_steps", 0))
    return {
        "effective_rank_floor": float(optimizer.minimum_effective_rank),
        "effective_rank_schedule_step": int(optimizer.schedule_step),
        "effective_rank_accepted_steps": accepted,
        "effective_rank_skipped_steps": skipped,
        "effective_rank_projected_steps": projected,
        "effective_rank_projection_frobenius": projection_frobenius,
        "effective_rank_joint_newton_steps": joint_newton,
        "effective_rank_unconstrained_steps": unconstrained,
        "effective_rank_certified_fallback_steps": fallback,
    }


def run_ppl_trial(
    task: FormalTask,
    optimizer_name: str,
    root: Path,
    *,
    run_label: str,
    learning_rate: float,
    workers: int = 4,
    seed: int = 1337,
    maximum_epochs: int = 5,
    maximum_updates: int | None = None,
    validation_batches: int = 64,
    weight_decay: float = 0.0,
    auxiliary_learning_rate: float | None = None,
) -> dict:
    """Train one matched run and record measured PPL at actual optimizer steps.

    ``maximum_updates`` is only for short tuning screens.  A final result uses
    ``None`` and therefore completes the requested five epochs.
    """
    if optimizer_name not in DISPLAY_NAMES:
        raise ValueError(f"unsupported PPL optimizer: {optimizer_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < maximum_epochs <= task.estimated_epochs:
        raise ValueError("maximum_epochs must lie within the task epoch budget")
    if maximum_updates is not None and maximum_updates <= 0:
        raise ValueError("maximum_updates must be positive")
    if (
        learning_rate <= 0
        or weight_decay < 0
        or (auxiliary_learning_rate is not None and auxiliary_learning_rate <= 0)
    ):
        raise ValueError("learning_rate must be positive and weight_decay non-negative")

    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    updates_per_epoch = len(train_loader) // task.gradient_accumulation
    scheduled_updates = (
        maximum_updates
        if maximum_updates is not None
        else maximum_epochs * updates_per_epoch
    )
    optimizers = build_optimizers(
        model,
        optimizer_name,
        learning_rate,
        weight_decay,
        task.muon_aux_lr if auxiliary_learning_rate is None else auxiliary_learning_rate,
        effective_rank_schedule_steps=scheduled_updates,
    )
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=task.estimated_epochs)
        for name, optimizer in optimizers.items()
    }
    paths = ppl_trial_paths(root, optimizer_name, run_label=run_label)
    completed_epoch = completed_updates = 0
    elapsed_seconds = 0.0
    if maximum_updates is None and paths.checkpoint.exists():
        completed_epoch, completed_updates, elapsed_seconds = _restore_checkpoint(
            paths.checkpoint, model, optimizers, schedulers
        )
    elif paths.metric.exists():
        paths.metric.unlink()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stopped_early = False
    for epoch in range(completed_epoch + 1, maximum_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            token_ids, targets = (value.to(device, non_blocking=True) for value in batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(token_ids)
                logits = output.logits if hasattr(output, "logits") else output
                loss = functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            (loss / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation != 0:
                continue
            for optimizer in optimizers.values():
                optimizer.step()
            completed_updates += 1
            if maximum_updates is not None and completed_updates >= maximum_updates:
                stopped_early = True
                break
        if not stopped_early:
            for scheduler in schedulers.values():
                scheduler.step()
            completed_epoch = epoch
        elapsed_seconds += time.perf_counter() - started
        nll = validation_nll(model, validation_loader, device, maximum_batches=validation_batches)
        record = {
            "epoch": epoch,
            "step": completed_updates,
            "elapsed_seconds": elapsed_seconds,
            "validation_nll": nll,
            "perplexity": perplexity_from_nll(nll),
        }
        record.update(_optimizer_diagnostics(optimizer_name, optimizers))
        write_metric(paths.metric, record)
        print(json.dumps({"optimizer": optimizer_name, **record}, sort_keys=True), flush=True)
        if maximum_updates is None:
            _save_checkpoint(
                paths.checkpoint,
                model,
                optimizers,
                schedulers,
                completed_epoch=completed_epoch,
                completed_updates=completed_updates,
                elapsed_seconds=elapsed_seconds,
            )
        if stopped_early:
            break
        started = time.perf_counter()

    result = {
        "task": task.identifier,
        "model": task.model,
        "optimizer": optimizer_name,
        "parameters": parameter_count(model),
        "epochs_completed": completed_epoch,
        "updates": completed_updates,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "validation_batches": validation_batches,
        "final_validation_nll": nll,
        "final_perplexity": perplexity_from_nll(nll),
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "screen_completed" if stopped_early else "completed",
    }
    result.update(_optimizer_diagnostics(optimizer_name, optimizers))
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not stopped_early:
        paths.checkpoint.unlink(missing_ok=True)
    return result


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_ppl_comparison_plots(
    root: Path, *, label: str, methods: Iterable[str]
) -> tuple[Path, Path]:
    """Render only actual validation perplexities against steps and elapsed time."""
    method_list = tuple(methods)
    if not method_list or any(method not in DISPLAY_NAMES for method in method_list):
        raise ValueError("the PPL comparison requires registered methods")
    traces = [
        (DISPLAY_NAMES[method], _read_records(ppl_trial_paths(root, method, run_label=label).metric))
        for method in method_list
    ]
    if any(not records for _, records in traces):
        raise ValueError("every plotted PPL method needs measured records")
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"gpt2_ppl_{label}_steps.png",
        output_root / f"gpt2_ppl_{label}_time.png",
    )
    for output, key, xlabel, scale in (
        (outputs[0], "step", "Completed optimizer step", 1.0),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)", 3600.0),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for name, records in traces:
            axis.plot(
                [record[key] / scale for record in records],
                [record["perplexity"] for record in records],
                marker="o",
                label=name,
            )
        axis.set(
            xlabel=xlabel,
            ylabel="Validation perplexity (lower is better)",
            title="GPT2-12x512 matched optimizer comparison",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs
