"""Resumable matrix-free generalized Gauss--Newton language-model trials."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional

from artifacts import write_metric
from full_ggn import FullGGNConfig, FullGGNState, full_ggn_step
from gn_experiment import LANGUAGE_MODEL_GN_TASK, artifact_paths, write_gn_comparison_plots
from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch, GGNFullOperator
from training import _evaluate, _loaders, _model, configure_reproducibility


def full_gn_task(*, batch_size: int):
    """Use a full-GGN batch without gradient accumulation."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=batch_size,
        gradient_accumulation=1,
    )


def prepare_full_gn_batch(
    batch: tuple[torch.Tensor, torch.Tensor], *, batch_size: int, sequence_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a fixed token budget for an exact GGN curvature step."""
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    token_ids, targets = batch
    if token_ids.size(0) < batch_size or token_ids.size(1) < sequence_length:
        raise ValueError("Loader batch is smaller than the requested GGN batch")
    return token_ids[:batch_size, :sequence_length], targets[:batch_size, :sequence_length]


def _loss(model, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(token_ids)
    if hasattr(logits, "logits"):
        logits = logits.logits
    return functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


def _checkpoint(
    path: Path,
    model,
    state: FullGGNState,
    *,
    next_epoch: int,
    next_batch: int,
    completed_steps: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "damping": state.damping,
            "previous_direction": (
                None
                if state.previous_direction is None
                else state.previous_direction.detach().cpu()
            ),
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "completed_steps": completed_steps,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def _restore(path: Path, model, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    previous = payload["previous_direction"]
    return (
        FullGGNState(
            damping=float(payload["damping"]),
            previous_direction=None if previous is None else previous.to(device),
        ),
        int(payload["next_epoch"]),
        int(payload["next_batch"]),
        int(payload["completed_steps"]),
        float(payload["elapsed_seconds"]),
    )


def run_full_gn_trial(
    root: Path,
    *,
    batch_size: int,
    sequence_length: int = 1024,
    initial_damping: float,
    maximum_cg_iterations: int,
    maximum_seconds: float = 14_400.0,
    evaluation_interval_steps: int = 256,
    workers: int = 4,
    seed: int = 1337,
    fresh: bool = False,
    label: str | None = None,
) -> dict:
    """Train the retained model with exact full-GGN Hessian-free updates."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-GGN experiment")
    task = full_gn_task(batch_size=batch_size)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    paths = artifact_paths(root, "full_ggn", run_label=label or "full_ggn")
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    state = FullGGNState(damping=initial_damping)
    next_epoch = 1
    resume_batch = 0
    completed_steps = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        state, next_epoch, resume_batch, completed_steps, elapsed_seconds = _restore(
            paths.checkpoint, model, device
        )
    elif paths.metric.exists():
        paths.metric.unlink()
    config = FullGGNConfig(maximum_cg_iterations=maximum_cg_iterations)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(next_epoch, task.estimated_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == next_epoch and batch_index < resume_batch:
                continue
            token_ids, targets = (
                value.to(device, non_blocking=True)
                for value in prepare_full_gn_batch(
                    batch, batch_size=batch_size, sequence_length=sequence_length
                )
            )
            operator = GGNFullOperator(
                model,
                FunctionalCurvatureBatch(
                    args=(token_ids,),
                    loss_fn=lambda output: functional.cross_entropy(
                        (output.logits if hasattr(output, "logits") else output)
                        .float()
                        .reshape(-1, (output.logits if hasattr(output, "logits") else output).size(-1)),
                        targets.reshape(-1),
                    ),
                    batch_id=f"full-gn-{completed_steps}",
                ),
            )
            step = full_ggn_step(
                operator,
                lambda: _loss(model, token_ids, targets),
                state=state,
                config=config,
            )
            completed_steps += 1
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            timed_out = elapsed_seconds >= maximum_seconds
            if completed_steps % evaluation_interval_steps == 0 or timed_out:
                metric = _evaluate(task, model, validation_loader, device)
                write_metric(
                    paths.metric,
                    {
                        "step": completed_steps,
                        "metric": metric,
                        "elapsed_seconds": elapsed_seconds,
                        "damping": state.damping,
                        "accepted": float(step.accepted),
                        "step_scale": step.step_scale,
                        "reduction_ratio": step.reduction_ratio,
                        "cg_iterations": step.cg_iterations,
                        "predicted_reduction": step.predicted_reduction,
                    },
                )
                _checkpoint(
                    paths.checkpoint,
                    model,
                    state,
                    next_epoch=epoch,
                    next_batch=batch_index + 1,
                    completed_steps=completed_steps,
                    elapsed_seconds=elapsed_seconds,
                )
                write_gn_comparison_plots(root)
                print(
                    f"task={task.identifier} optimizer=full_ggn step={completed_steps} "
                    f"metric={metric:.6f} damping={state.damping:.6g}",
                    flush=True,
                )
            if timed_out:
                result = {
                    "task": task.identifier,
                    "domain": task.domain,
                    "model": task.model,
                    "optimizer": "full_ggn",
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "maximum_cg_iterations": maximum_cg_iterations,
                    "completed_steps": completed_steps,
                    "seconds": elapsed_seconds,
                    "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    "status": "time_limit_checkpointed",
                }
                paths.result.parent.mkdir(parents=True, exist_ok=True)
                paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                return result
        resume_batch = 0
        next_epoch = epoch + 1
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "full_ggn",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "maximum_cg_iterations": maximum_cg_iterations,
        "completed_steps": completed_steps,
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "completed",
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    paths.checkpoint.unlink(missing_ok=True)
    write_gn_comparison_plots(root)
    return result
