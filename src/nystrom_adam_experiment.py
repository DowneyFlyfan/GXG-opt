"""Two-stage Nyström-Generalized-Gauss-Newton then AdamW GPT experiment."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from artifacts import write_metric
from gn_experiment import (
    LANGUAGE_MODEL_GN_TASK,
    ArtifactPaths,
    _read_metrics,
    artifact_paths,
)
from new_curvature_experiment import (
    run_nystrom_ggn_trial,
    write_new_curvature_comparison_plots,
)
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


def nystrom_adam_paths(root: Path, label: str) -> ArtifactPaths:
    """Return isolated artifacts for the explicit curvature-to-AdamW handoff."""
    return artifact_paths(root, "nystrom_adam", run_label=f"nystrom_adam_{label}")


def fresh_adamw_after_nystrom(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Start AdamW from the curvature-stage weights with no invented moments."""
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("AdamW handoff hyperparameters are invalid")
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )


def write_nystrom_adam_comparison_plots(root: Path, *, label: str) -> tuple[Path, Path] | None:
    """Compare the combined two-stage trace with the selected AdamW/Muon runs."""
    return write_new_curvature_comparison_plots(
        root,
        candidate="nystrom_adam",
        label=f"nystrom_adam_{label}",
        display_name="Nyström-GGN → AdamW",
    )


def _save_hybrid_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    completed_epoch: int,
    completed_adam_steps: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "completed_epoch": completed_epoch,
            "completed_adam_steps": completed_adam_steps,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def run_nystrom_adam_trial(
    root: Path,
    *,
    label: str,
    nystrom_initial_step_scale: float = 1.5e-5,
    nystrom_outer_steps: int = 1,
    adamw_learning_rate: float = 1.5e-4,
    maximum_seconds: float = 14_400.0,
    workers: int = 4,
    seed: int = 1337,
    fresh: bool = False,
    write_plots: bool = False,
) -> dict:
    """Run one or more Nyström-GGN updates then continue with fresh AdamW.

    The stage checkpoint transfers model parameters only.  The AdamW optimizer
    is intentionally initialized after loading that checkpoint so that neither
    moment estimates nor scheduler state are fabricated from curvature data.
    """
    if nystrom_outer_steps <= 0 or nystrom_initial_step_scale <= 0:
        raise ValueError("Nyström stage hyperparameters must be positive")
    if adamw_learning_rate <= 0 or maximum_seconds <= 0:
        raise ValueError("AdamW learning rate and maximum seconds must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Nyström-to-AdamW trial")
    stage_label = f"nystrom_adam_{label}_stage1"
    stage_result = run_nystrom_ggn_trial(
        root,
        label=stage_label,
        initial_step_scale=nystrom_initial_step_scale,
        maximum_outer_steps=nystrom_outer_steps,
        maximum_seconds=maximum_seconds,
        workers=workers,
        seed=seed,
        fresh=fresh,
    )
    stage_paths = artifact_paths(root, "nystrom_ggn", run_label=stage_label)
    stage_records = _read_metrics(stage_paths.metric)
    if not stage_records or any(record["line_search_step_size"] <= 0 for record in stage_records):
        raise RuntimeError("Nyström stage made no nonzero held-out update; AdamW handoff is not run")
    paths = nystrom_adam_paths(root, label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    task = replace(LANGUAGE_MODEL_GN_TASK, micro_batch_size=12, gradient_accumulation=4)
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    stage_payload = torch.load(stage_paths.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(stage_payload["model"])
    optimizer = fresh_adamw_after_nystrom(
        model,
        learning_rate=adamw_learning_rate,
        weight_decay=task.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=task.estimated_epochs
    )
    completed_epoch = completed_adam_steps = 0
    elapsed_seconds = float(stage_result["post_warmup_seconds"])
    if paths.checkpoint.exists():
        payload = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        completed_epoch = int(payload["completed_epoch"])
        completed_adam_steps = int(payload["completed_adam_steps"])
        elapsed_seconds = float(payload["elapsed_seconds"])
    elif not paths.metric.exists():
        stage_record = stage_records[-1]
        write_metric(
            paths.metric,
            {
                "stage": "nystrom_ggn",
                "step": int(stage_record["step"]),
                "metric": stage_record["metric"],
                "elapsed_seconds": elapsed_seconds,
                "nystrom_initial_step_scale": nystrom_initial_step_scale,
                "nystrom_line_search_step_size": stage_record["line_search_step_size"],
            },
        )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    timed_out = False
    for epoch in range(completed_epoch + 1, task.estimated_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index % task.gradient_accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation:
                continue
            optimizer.step()
            completed_adam_steps += 1
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if elapsed_seconds >= maximum_seconds:
                timed_out = True
                break
        if not timed_out:
            scheduler.step()
            completed_epoch = epoch
        metric = _evaluate(task, model, validation_loader, device)
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        write_metric(
            paths.metric,
            {
                "stage": "adamw",
                "epoch": epoch,
                "step": nystrom_outer_steps + completed_adam_steps,
                "adamw_steps": completed_adam_steps,
                "metric": metric,
                "elapsed_seconds": elapsed_seconds,
                "adamw_learning_rate": adamw_learning_rate,
            },
        )
        _save_hybrid_checkpoint(
            paths.checkpoint,
            model,
            optimizer,
            scheduler,
            completed_epoch=completed_epoch,
            completed_adam_steps=completed_adam_steps,
            elapsed_seconds=elapsed_seconds,
        )
        if timed_out:
            break
    records = _read_metrics(paths.metric)
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "nystrom_ggn_then_adamw",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "nystrom_outer_steps": nystrom_outer_steps,
        "nystrom_initial_step_scale": nystrom_initial_step_scale,
        "nystrom_selected_step_sizes": [record["line_search_step_size"] for record in stage_records],
        "adamw_learning_rate": adamw_learning_rate,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "completed_adam_epochs": completed_epoch,
        "completed_adam_steps": completed_adam_steps,
        "final_metric": records[-1]["metric"],
        "seconds": elapsed_seconds,
        "nystrom_post_warmup_seconds": stage_result["post_warmup_seconds"],
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "time_limit_checkpointed" if timed_out else "completed",
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if write_plots:
        write_nystrom_adam_comparison_plots(root, label=label)
    return result
