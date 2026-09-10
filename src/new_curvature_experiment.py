"""Controlled GPT-2 experiments for MKOR, RACS, and Nyström-GGN."""

from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plot
import torch
from torch import Tensor, nn

from artifacts import write_metric
from gn_experiment import (
    LANGUAGE_MODEL_GN_TASK,
    _comparison_records,
    _read_metrics,
    artifact_paths,
)
from gpt_baseline_selection import selected_baseline_label, selected_baseline_paths
from kronecker_ggn_common.curvature_operator import GGNFullOperator
from mkor import MKOR, eligible_linear_modules
from nystrom_ggn import NystromState, build_nystrom_state
from paper_full_gn_experiment import (
    _infinite_batches,
    _load_complete_warmup,
    common_warmup_path,
    language_model_curvature_batch,
    paper_gn_time_limit_reached,
)
from racs import RACS
from recycled_low_rank_gn_experiment import (
    ExactBatchStream,
    SampleWeightedGGNOperator,
    _held_out_batches,
    _outer_gradient_and_curvature_batches,
    held_out_direction_line_search,
)
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility
from models import parameter_count


def _candidate_output_paths(root: Path, label: str) -> tuple[Path, Path]:
    output_root = root / "results" / "nlp"
    return (
        output_root / f"{label}_metric_steps.png",
        output_root / f"{label}_metric_time.png",
    )


def write_new_curvature_comparison_plots(
    root: Path,
    *,
    candidate: str,
    label: str,
    display_name: str,
) -> tuple[Path, Path] | None:
    """Plot the candidate against the immutable tuned GPT baseline records."""
    candidate_path = artifact_paths(root, candidate, run_label=label).metric
    if not candidate_path.exists():
        return None
    traces: list[tuple[str, list[dict]]] = []
    for optimizer in ("adamw", "muon", "muown"):
        metric, result = selected_baseline_paths(root, optimizer)
        if metric.exists() and result.exists():
            traces.append((selected_baseline_label(optimizer), _comparison_records(root, optimizer)))
    candidate_records = _read_metrics(candidate_path)
    if len(traces) != 3 or not candidate_records:
        return None
    traces.append((display_name, candidate_records))
    outputs = _candidate_output_paths(root, label)
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for output, key, axis_label in (
        (outputs[0], "step", "Completed optimizer step / outer step"),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)"),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for name, records in traces:
            values = [record for record in records if key in record]
            axis.plot(
                [record[key] / 3600 if key == "elapsed_seconds" else record[key] for record in values],
                [record["metric"] for record in values],
                label=name,
            )
        if key == "step":
            axis.set_xscale("log")
        axis.set(
            xlabel=axis_label,
            ylabel="Validation next-token accuracy",
            title=f"{display_name} versus tuned GPT-2 baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def _matrix_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    selected_names = eligible_linear_modules(model)
    named = dict(model.named_parameters())
    matrix_parameters = [named[f"{name}.weight"] for name in sorted(selected_names)]
    selected_ids = {id(parameter) for parameter in matrix_parameters}
    auxiliary = [parameter for parameter in model.parameters() if id(parameter) not in selected_ids]
    return matrix_parameters, auxiliary


def _save_matrix_checkpoint(
    path: Path,
    model: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    *,
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            "epoch": epoch,
            "step": step,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def _run_matrix_trial(
    root: Path,
    *,
    candidate: str,
    label: str,
    matrix_learning_rate: float,
    auxiliary_learning_rate: float,
    maximum_seconds: float,
    workers: int,
    seed: int,
    fresh: bool,
    maximum_epochs: int | None = None,
    write_plots: bool = False,
    racs_scale: float = 0.05,
) -> dict:
    if candidate not in {"mkor", "racs"}:
        raise ValueError("matrix candidate must be mkor or racs")
    if maximum_epochs is not None and not 0 < maximum_epochs <= LANGUAGE_MODEL_GN_TASK.estimated_epochs:
        raise ValueError("maximum_epochs must lie within the formal epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the curvature experiments")
    configure_reproducibility(seed)
    device = torch.device("cuda")
    task = replace(LANGUAGE_MODEL_GN_TASK, micro_batch_size=12, gradient_accumulation=4)
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    matrices, auxiliary = _matrix_parameters(model)
    if candidate == "mkor":
        matrix_optimizer: torch.optim.Optimizer = MKOR(matrices, lr=matrix_learning_rate, momentum=0.9, weight_decay=task.weight_decay)
        matrix_optimizer.attach(model)
    else:
        matrix_optimizer = RACS(matrices, lr=matrix_learning_rate, beta=0.9, scale=racs_scale, limiter=1.01, weight_decay=task.weight_decay)
    optimizers = {
        candidate: matrix_optimizer,
        "adamw_aux": torch.optim.AdamW(auxiliary, lr=auxiliary_learning_rate, weight_decay=task.weight_decay, betas=(0.9, 0.95)),
    }
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=task.estimated_epochs) for name, optimizer in optimizers.items()}
    paths = artifact_paths(root, candidate, run_label=label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    completed_epoch = completed_steps = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        payload = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        completed_epoch, completed_steps = int(payload["epoch"]), int(payload["step"])
        elapsed_seconds = float(payload["elapsed_seconds"])
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    timed_out = False
    target_epochs = task.estimated_epochs if maximum_epochs is None else maximum_epochs
    for epoch in range(completed_epoch + 1, target_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.step()
                completed_steps += 1
                elapsed_seconds += time.perf_counter() - started
                started = time.perf_counter()
                if paper_gn_time_limit_reached(elapsed_seconds, maximum_seconds):
                    timed_out = True
                    break
        if timed_out:
            break
        for scheduler in schedulers.values():
            scheduler.step()
        completed_epoch = epoch
        metric = _evaluate(task, model, validation_loader, device)
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        write_metric(paths.metric, {"epoch": epoch, "step": completed_steps, "metric": metric, "elapsed_seconds": elapsed_seconds})
        _save_matrix_checkpoint(paths.checkpoint, model, optimizers, schedulers, epoch=completed_epoch, step=completed_steps, elapsed_seconds=elapsed_seconds)
    final_metric = _read_metrics(paths.metric)[-1]["metric"] if paths.metric.exists() else None
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": candidate,
        "parameters": parameter_count(model),
        "matrix_learning_rate": matrix_learning_rate,
        "auxiliary_learning_rate": auxiliary_learning_rate,
        "racs_scale": racs_scale if candidate == "racs" else None,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "completed_epochs": completed_epoch,
        "completed_steps": completed_steps,
        "final_metric": final_metric,
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "time_limit_checkpointed" if timed_out else ("completed" if target_epochs == task.estimated_epochs else "screen_complete"),
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    display = "MKOR" if candidate == "mkor" else "RACS"
    if write_plots:
        write_new_curvature_comparison_plots(root, candidate=candidate, label=label, display_name=display)
    if candidate == "mkor":
        matrix_optimizer.detach()
    return result


def run_mkor_trial(root: Path, **kwargs) -> dict:
    return _run_matrix_trial(root, candidate="mkor", **kwargs)


def run_racs_trial(root: Path, **kwargs) -> dict:
    return _run_matrix_trial(root, candidate="racs", **kwargs)


def _nystrom_checkpoint(path: Path, model: nn.Module, *, state: NystromState | None, step: int, consumed_sequences: int, elapsed_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "basis": None if state is None else state.basis.detach().cpu(), "indices": None if state is None else state.indices, "step": step, "consumed_sequences": consumed_sequences, "elapsed_seconds": elapsed_seconds}, path)


def run_nystrom_ggn_trial(
    root: Path,
    *,
    label: str,
    physical_batch_size: int = 1,
    outer_effective_batch_size: int = 3904,
    curvature_batch_size: int = 64,
    nystrom_rank: int = 4,
    damping: float = 0.1,
    refresh_interval: int = 4,
    initial_step_scale: float = 1.0,
    maximum_outer_steps: int = 8,
    maximum_seconds: float = 14_400.0,
    workers: int = 4,
    seed: int = 1337,
    fresh: bool = False,
) -> dict:
    if physical_batch_size != 1 or curvature_batch_size != 64 or nystrom_rank != 4 or damping != 0.1 or refresh_interval != 4:
        raise ValueError("formal Nys-GGN must use b1/c64/r4/d0.1/refresh4")
    if initial_step_scale <= 0:
        raise ValueError("Nyström-GGN initial step scale must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the curvature experiments")
    configure_reproducibility(seed)
    device = torch.device("cuda")
    task = replace(LANGUAGE_MODEL_GN_TASK, micro_batch_size=physical_batch_size, gradient_accumulation=1)
    train_loader, validation_loader = _loaders(task, root, workers, seed + 17)
    model = _model(task).to(device)
    warmup = _load_complete_warmup(common_warmup_path(root), model)
    paths = artifact_paths(root, "nystrom_ggn", run_label=label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    completed_steps = consumed_sequences = 0
    elapsed_seconds = 0.0
    state: NystromState | None = None
    if paths.checkpoint.exists():
        payload = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        completed_steps, consumed_sequences = int(payload["step"]), int(payload["consumed_sequences"])
        elapsed_seconds = float(payload["elapsed_seconds"])
        if payload["basis"] is not None:
            state = NystromState(payload["basis"].to(device), damping, tuple(payload["indices"]))
    stream = ExactBatchStream(_infinite_batches(train_loader))
    if consumed_sequences:
        stream.skip(consumed_sequences)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_metric = None
    total_matvecs = 0
    while completed_steps < maximum_outer_steps:
        gradient, curvature_batches, _ = _outer_gradient_and_curvature_batches(
            model, stream.take(outer_effective_batch_size), curvature_batch_size=curvature_batch_size, device=device, collect_second_moment=False
        )
        consumed_sequences += outer_effective_batch_size
        counts = tuple(batch.args[0].shape[0] for batch in curvature_batches)
        operator = SampleWeightedGGNOperator(tuple(GGNFullOperator(model, batch) for batch in curvature_batches), sample_counts=counts)
        matvecs = 0

        def matvec(vector: Tensor) -> Tensor:
            nonlocal matvecs
            matvecs += 1
            return operator.matvec(vector)

        if state is None or completed_steps % refresh_interval == 0:
            indices = random.Random(seed + 29 + completed_steps).sample(
                range(gradient.numel()), nystrom_rank
            )
            state = build_nystrom_state(matvec, dimension=gradient.numel(), indices=indices, rank=nystrom_rank, damping=damping, dtype=gradient.dtype, storage_dtype=torch.bfloat16, device=device)
            refreshed = True
        else:
            refreshed = False
        direction = -state.inverse_action(gradient)
        if not torch.isfinite(direction).all() or torch.dot(gradient, direction) >= 0:
            raise RuntimeError("Nyström-GGN failed to produce a finite descent direction")
        del gradient, operator, curvature_batches
        held_out = _held_out_batches(stream.take(outer_effective_batch_size), device=device)
        consumed_sequences += outer_effective_batch_size
        line_search = held_out_direction_line_search(model, direction, held_out, search_range=5, initial_step_scale=initial_step_scale, screening_sequences=64, finalists=2, include_zero_step=True)
        del direction, held_out
        completed_steps += 1
        total_matvecs += matvecs
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        final_metric = _evaluate(task, model, validation_loader, device)
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        write_metric(paths.metric, {"step": completed_steps, "metric": final_metric, "elapsed_seconds": float(warmup["elapsed_seconds"]) + elapsed_seconds, "post_warmup_elapsed_seconds": elapsed_seconds, "damping": damping, "nystrom_rank": state.rank, "nystrom_columns": nystrom_rank, "refresh_interval": refresh_interval, "initial_step_scale": initial_step_scale, "refreshed": refreshed, "averaged_curvature_matvecs": matvecs, "line_search_step_size": line_search.step_size, "line_search_loss": line_search.loss, "outer_effective_batch_size": outer_effective_batch_size, "curvature_batch_size": curvature_batch_size})
        _nystrom_checkpoint(paths.checkpoint, model, state=state, step=completed_steps, consumed_sequences=consumed_sequences, elapsed_seconds=elapsed_seconds)
        write_new_curvature_comparison_plots(root, candidate="nystrom_ggn", label=label, display_name="Nyström-GGN")
        torch.cuda.empty_cache()
        if paper_gn_time_limit_reached(elapsed_seconds, maximum_seconds):
            break
    result = {"task": task.identifier, "domain": task.domain, "model": task.model, "optimizer": "nystrom_ggn", "parameters": parameter_count(model), "warmup_tokens": int(warmup["processed_tokens"]), "physical_batch_size": physical_batch_size, "outer_effective_batch_size": outer_effective_batch_size, "curvature_batch_size": curvature_batch_size, "nystrom_rank": nystrom_rank, "damping": damping, "refresh_interval": refresh_interval, "initial_step_scale": initial_step_scale, "completed_outer_steps": completed_steps, "consumed_training_sequences": consumed_sequences, "averaged_curvature_matvecs": total_matvecs, "final_metric": final_metric, "seconds": float(warmup["elapsed_seconds"]) + elapsed_seconds, "post_warmup_seconds": elapsed_seconds, "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2, "status": "completed" if completed_steps >= maximum_outer_steps else "time_limit_checkpointed"}
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
