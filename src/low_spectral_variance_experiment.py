"""Controlled GPT-2 12x512 experiment for Low-Spectral-Variance."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK, _read_metrics
from gpt_baseline_selection import selected_baseline_label, selected_baseline_paths
from low_spectral_variance import low_spectral_variance_parameter_names
from optimizers import build_optimizers, muon_parameter_names
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


@dataclass(frozen=True)
class LowSpectralVariancePaths:
    metric: Path
    result: Path
    checkpoint: Path


def low_spectral_variance_paths(root: Path, label: str) -> LowSpectralVariancePaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__low_spectral_variance_{label}"
    return LowSpectralVariancePaths(
        metric=root / "metrics" / "nlp" / f"{stem}.jsonl",
        result=root / "results" / "nlp" / f"{stem}.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def low_spectral_variance_task(*, gradient_accumulation: int):
    """Return the fixed GPT workload with an explicitly chosen effective batch."""
    if gradient_accumulation <= 0:
        raise ValueError("gradient accumulation must be positive")
    return replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=12,
        gradient_accumulation=gradient_accumulation,
    )


def _baseline_records(root: Path, optimizer: str) -> list[dict]:
    metric_path, result_path = selected_baseline_paths(root, optimizer)
    records = _read_metrics(metric_path)
    if not records or "step" in records[0]:
        return records
    result = json.loads(result_path.read_text())
    steps_per_epoch = math.ceil(12_207 / int(result["gradient_accumulation"]))
    return [
        {
            **record,
            "step": int(record["epoch"]) * steps_per_epoch,
            "elapsed_seconds": result["seconds"] * int(record["epoch"]) / int(result["epochs"]),
        }
        for record in records
    ]


def write_low_spectral_variance_comparison_plots(
    root: Path, *, label: str
) -> tuple[Path, Path] | None:
    paths = low_spectral_variance_paths(root, label)
    if not paths.metric.exists():
        return None
    candidate = _read_metrics(paths.metric)
    if not candidate:
        return None
    traces = [
        (selected_baseline_label("adamw"), _baseline_records(root, "adamw")),
        (selected_baseline_label("muon"), _baseline_records(root, "muon")),
        (selected_baseline_label("muown"), _baseline_records(root, "muown")),
        ("Low-Spectral-Variance", candidate),
    ]
    if not all(records for _, records in traces):
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"low_spectral_variance_{label}_metric_steps.png",
        output_root / f"low_spectral_variance_{label}_metric_time.png",
    )
    for output, key, xlabel in (
        (outputs[0], "step", "Completed optimizer step"),
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
        axis.set(
            xlabel=xlabel,
            ylabel="Validation next-token accuracy",
            title="Low-Spectral-Variance versus tuned GPT-2 baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def _save_checkpoint(
    path: Path,
    model,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    *,
    epoch: int,
    steps: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            "epoch": epoch,
            "steps": steps,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def run_low_spectral_variance_trial(
    root: Path,
    *,
    label: str,
    learning_rate: float,
    condition_limit: float,
    dual_steps: int = 8,
    momentum: float = 0.95,
    workers: int = 4,
    maximum_seconds: float = 14_400.0,
    maximum_epochs: int | None = None,
    gradient_accumulation: int = 4,
    seed: int = 1337,
    fresh: bool = False,
    write_plots: bool = False,
) -> dict:
    """Train the source-derived condition-capped matrix optimizer.

    Matrix parameters outside the requested cap remain in the AdamW auxiliary
    group.  This is necessary because the source method requires a feasible
    starting point; no model weights are silently projected at initialization.
    """
    if learning_rate <= 0 or condition_limit <= 1 or dual_steps <= 0:
        raise ValueError("learning rate, condition limit, and dual steps must be positive")
    if not 0 <= momentum < 1 or maximum_seconds <= 0:
        raise ValueError("momentum and maximum seconds are invalid")
    task = low_spectral_variance_task(gradient_accumulation=gradient_accumulation)
    target_epochs = task.estimated_epochs if maximum_epochs is None else maximum_epochs
    if not 0 < target_epochs <= task.estimated_epochs:
        raise ValueError("maximum epochs must lie within the formal five-epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Low-Spectral-Variance")
    paths = low_spectral_variance_paths(root, label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    candidates = muon_parameter_names(model)
    constrained_names = low_spectral_variance_parameter_names(
        model, condition_limit=condition_limit, candidates=candidates
    )
    if not constrained_names:
        raise ValueError("no GPT-2 matrix is feasible under this condition-number cap")
    optimizers = build_optimizers(
        model,
        "low_spectral_variance",
        learning_rate,
        task.weight_decay,
        task.muon_aux_lr,
        low_spectral_condition_limit=condition_limit,
        low_spectral_dual_steps=dual_steps,
    )
    for group in optimizers["low_spectral_variance"].param_groups:
        group["momentum"] = momentum
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=task.estimated_epochs)
        for name, optimizer in optimizers.items()
    }
    completed_epoch = completed_steps = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        payload = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        completed_epoch = int(payload["epoch"])
        completed_steps = int(payload["steps"])
        elapsed_seconds = float(payload["elapsed_seconds"])
    steps_per_epoch = len(train_loader) // task.gradient_accumulation
    partial_steps = completed_steps - completed_epoch * steps_per_epoch
    if not 0 <= partial_steps < steps_per_epoch:
        raise ValueError("checkpoint has an invalid partial-epoch optimizer-step cursor")
    if partial_steps:
        # A time-limit checkpoint can be taken within an epoch.  The model and
        # optimizer state are exact at ``partial_steps``; restart the loader's
        # deterministic order and skip those already-applied accumulation
        # windows rather than replaying them.  The provisional evaluation is
        # not an epoch metric and must not appear in the final learning curve.
        retained = [record for record in _read_metrics(paths.metric) if int(record["epoch"]) <= completed_epoch]
        paths.metric.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in retained))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    timed_out = False
    for epoch in range(completed_epoch + 1, target_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == completed_epoch + 1 and batch_index < partial_steps * task.gradient_accumulation:
                continue
            if batch_index % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation:
                continue
            for optimizer in optimizers.values():
                optimizer.step()
            completed_steps += 1
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if elapsed_seconds >= maximum_seconds:
                timed_out = True
                break
        if not timed_out:
            for scheduler in schedulers.values():
                scheduler.step()
            completed_epoch = epoch
            partial_steps = 0
        metric = _evaluate(task, model, validation_loader, device)
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        write_metric(
            paths.metric,
            {
                "epoch": epoch,
                "step": completed_steps,
                "metric": metric,
                "elapsed_seconds": elapsed_seconds,
                "learning_rate": learning_rate,
                "condition_limit": condition_limit,
                "dual_steps": dual_steps,
                "constrained_matrix_count": len(constrained_names),
            },
        )
        _save_checkpoint(
            paths.checkpoint,
            model,
            optimizers,
            schedulers,
            epoch=completed_epoch,
            steps=completed_steps,
            elapsed_seconds=elapsed_seconds,
        )
        if timed_out:
            break
    records = _read_metrics(paths.metric)
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "low_spectral_variance",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": learning_rate,
        "condition_limit": condition_limit,
        "dual_steps": dual_steps,
        "momentum": momentum,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "constrained_matrix_count": len(constrained_names),
        "adamw_auxiliary_parameter_count": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name not in constrained_names
        ),
        "completed_epochs": completed_epoch,
        "completed_steps": completed_steps,
        "final_metric": records[-1]["metric"],
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "time_limit_checkpointed" if timed_out else (
            "completed" if target_epochs == task.estimated_epochs else "screen_complete"
        ),
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if write_plots:
        write_low_spectral_variance_comparison_plots(root, label=label)
    return result
