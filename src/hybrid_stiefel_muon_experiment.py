"""Muon on the edge Transformer blocks and Stiefel-Muon in the interior."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK, _read_metrics
from optimizers import build_optimizers, hybrid_muon_parameter_names
from stiefel_muon_experiment import (
    _historical_baseline_records,
    _load_checkpoint,
    _needs_epoch_end_evaluation,
    _save_checkpoint,
    stiefel_task,
)
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


@dataclass(frozen=True)
class HybridStiefelMuonPaths:
    metric: Path
    result: Path
    checkpoint: Path


def hybrid_stiefel_muon_paths(root: Path, label: str) -> HybridStiefelMuonPaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__hybrid_stiefel_muon_{label}"
    return HybridStiefelMuonPaths(
        metric=root / "metrics" / "nlp" / f"{stem}.jsonl",
        result=root / "results" / "nlp" / f"{stem}.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def write_hybrid_stiefel_muon_comparison_plots(
    root: Path, *, label: str
) -> tuple[Path, Path] | None:
    candidate = hybrid_stiefel_muon_paths(root, label)
    if not candidate.metric.exists():
        return None
    traces = []
    for display, optimizer in (("AdamW", "adamw"), ("Muon", "muon")):
        baseline = root / "metrics" / "nlp" / f"nlp_gpt_12x512__{optimizer}.jsonl"
        if baseline.exists():
            traces.append((display, _historical_baseline_records(root, optimizer)))
    traces.append(("Hybrid Stiefel-Muon", _read_metrics(candidate.metric)))
    if len(traces) != 3:
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"hybrid_stiefel_muon_{label}_metric_steps.png",
        output_root / f"hybrid_stiefel_muon_{label}_metric_time.png",
    )
    for output, key, xlabel in (
        (outputs[0], "step", "Optimizer step"),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)"),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for display, records in traces:
            values = [record for record in records if key in record]
            x_values = [
                record[key] / 3600 if key == "elapsed_seconds" else record[key]
                for record in values
            ]
            axis.plot(x_values, [record["metric"] for record in values], label=display)
        axis.set(xlabel=xlabel, ylabel="Validation next-token accuracy")
        axis.set_title("Hybrid Stiefel-Muon versus retained AdamW and Muon baselines")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def _write_hybrid_metric(
    paths: HybridStiefelMuonPaths,
    *,
    epoch: int,
    steps: int,
    metric: float,
    elapsed_seconds: float,
    muon_learning_rate: float,
    stiefel_learning_rate: float,
    momentum: float,
    ns_steps: int,
) -> None:
    write_metric(
        paths.metric,
        {
            "epoch": epoch,
            "step": steps,
            "metric": metric,
            "elapsed_seconds": elapsed_seconds,
            "muon_learning_rate": muon_learning_rate,
            "stiefel_learning_rate": stiefel_learning_rate,
            "momentum": momentum,
            "ns_steps": ns_steps,
        },
    )


def run_hybrid_stiefel_muon_trial(
    root: Path,
    *,
    label: str,
    muon_learning_rate: float,
    stiefel_learning_rate: float,
    stiefel_square_learning_rate: float | None = None,
    stiefel_rectangular_learning_rate: float | None = None,
    stiefel_nesterov: bool = False,
    momentum: float = 0.95,
    ns_steps: int = 5,
    scheduler_t_max: int | None = None,
    workers: int = 4,
    maximum_seconds: float = 14_400.0,
    maximum_steps: int | None = None,
    evaluation_interval_steps: int = 1_024,
    micro_batch_size: int = 4,
    gradient_accumulation: int = 2,
    seed: int = 1337,
    fresh: bool = False,
) -> dict:
    """Train a 12-block decoder with ordinary Muon on blocks 0/11 only."""
    if min(muon_learning_rate, stiefel_learning_rate) <= 0:
        raise ValueError("Both learning rates must be positive")
    for rate in (stiefel_square_learning_rate, stiefel_rectangular_learning_rate):
        if rate is not None and rate <= 0:
            raise ValueError("Geometry-specific Stiefel learning rates must be positive")
    if not 0 <= momentum < 1 or ns_steps <= 0:
        raise ValueError("momentum and Newton--Schulz steps are invalid")
    if scheduler_t_max is not None and scheduler_t_max <= 0:
        raise ValueError("scheduler_t_max must be positive when provided")
    if maximum_seconds <= 0 or evaluation_interval_steps <= 0:
        raise ValueError("time and evaluation intervals must be positive")
    if maximum_steps is not None and maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive when provided")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Hybrid Stiefel-Muon")

    task = stiefel_task(
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    paths = hybrid_stiefel_muon_paths(root, label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    edge_names, middle_names = hybrid_muon_parameter_names(model)
    optimizers = build_optimizers(
        model,
        "hybrid_stiefel_muon",
        muon_learning_rate,
        task.weight_decay,
        task.muon_aux_lr,
        stiefel_lr=stiefel_learning_rate,
        stiefel_square_lr=stiefel_square_learning_rate,
        stiefel_rectangular_lr=stiefel_rectangular_learning_rate,
        stiefel_nesterov=stiefel_nesterov,
    )
    for name in ("muon_edge", "stiefel_muon_middle"):
        for group in optimizers[name].param_groups:
            group["momentum"] = momentum
            group["ns_steps"] = ns_steps
    scheduler_horizon = task.estimated_epochs if scheduler_t_max is None else scheduler_t_max
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_horizon
        )
        for name, optimizer in optimizers.items()
    }
    epoch, resume_batch, steps, elapsed_seconds = (1, 0, 0, 0.0)
    if paths.checkpoint.exists():
        epoch, resume_batch, steps, elapsed_seconds = _load_checkpoint(
            paths.checkpoint, model, optimizers, schedulers
        )
    elif paths.metric.exists():
        paths.metric.unlink()
    last_evaluated_steps = (
        int(_read_metrics(paths.metric)[-1]["step"]) if paths.metric.exists() else 0
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stop = False
    final_metric = None
    while epoch <= task.estimated_epochs and not stop:
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index < resume_batch:
                continue
            if batch_index % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation:
                continue
            for optimizer in optimizers.values():
                optimizer.step()
            steps += 1
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            time_limit = elapsed_seconds >= maximum_seconds
            step_limit = maximum_steps is not None and steps >= maximum_steps
            epoch_end = batch_index + 1 == len(train_loader)
            due = steps % evaluation_interval_steps == 0 or time_limit or step_limit or epoch_end
            if due:
                final_metric = _evaluate(task, model, validation_loader, device)
                elapsed_seconds += time.perf_counter() - started
                started = time.perf_counter()
                _write_hybrid_metric(
                    paths,
                    epoch=epoch,
                    steps=steps,
                    metric=final_metric,
                    elapsed_seconds=elapsed_seconds,
                    muon_learning_rate=muon_learning_rate,
                    stiefel_learning_rate=stiefel_learning_rate,
                    momentum=momentum,
                    ns_steps=ns_steps,
                )
                last_evaluated_steps = steps
                _save_checkpoint(
                    paths.checkpoint,
                    model,
                    optimizers,
                    schedulers,
                    epoch=epoch,
                    batch=batch_index + 1,
                    steps=steps,
                    elapsed_seconds=elapsed_seconds,
                )
                write_hybrid_stiefel_muon_comparison_plots(root, label=label)
                print(
                    f"task={task.identifier} optimizer=hybrid_stiefel_muon step={steps} "
                    f"metric={final_metric:.6f} elapsed={elapsed_seconds:.1f}",
                    flush=True,
                )
            if time_limit or step_limit:
                stop = True
                break
        if stop:
            break
        if _needs_epoch_end_evaluation(
            steps=steps, last_evaluated_steps=last_evaluated_steps
        ):
            final_metric = _evaluate(task, model, validation_loader, device)
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            _write_hybrid_metric(
                paths,
                epoch=epoch,
                steps=steps,
                metric=final_metric,
                elapsed_seconds=elapsed_seconds,
                muon_learning_rate=muon_learning_rate,
                stiefel_learning_rate=stiefel_learning_rate,
                momentum=momentum,
                ns_steps=ns_steps,
            )
            last_evaluated_steps = steps
            _save_checkpoint(
                paths.checkpoint,
                model,
                optimizers,
                schedulers,
                epoch=epoch,
                batch=len(train_loader),
                steps=steps,
                elapsed_seconds=elapsed_seconds,
            )
            write_hybrid_stiefel_muon_comparison_plots(root, label=label)
            print(
                f"task={task.identifier} optimizer=hybrid_stiefel_muon step={steps} "
                f"metric={final_metric:.6f} elapsed={elapsed_seconds:.1f}",
                flush=True,
            )
        for scheduler in schedulers.values():
            scheduler.step()
        epoch += 1
        resume_batch = 0

    if final_metric is None:
        final_metric = _evaluate(task, model, validation_loader, device)
    status = "completed" if epoch > task.estimated_epochs else "time_limit_checkpointed"
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "hybrid_stiefel_muon",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "muon_edge_matrix_count": len(edge_names),
        "stiefel_middle_matrix_count": len(middle_names),
        "muon_learning_rate": muon_learning_rate,
        "stiefel_learning_rate": stiefel_learning_rate,
        "stiefel_square_learning_rate": stiefel_square_learning_rate,
        "stiefel_rectangular_learning_rate": stiefel_rectangular_learning_rate,
        "stiefel_nesterov": stiefel_nesterov,
        "momentum": momentum,
        "ns_steps": ns_steps,
        "scheduler_t_max": scheduler_horizon,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "completed_steps": steps,
        "final_metric": final_metric,
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": status,
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_hybrid_stiefel_muon_comparison_plots(root, label=label)
    return result
