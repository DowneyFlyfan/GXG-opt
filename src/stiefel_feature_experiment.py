"""Matched GPT2-12x512 experiment for Stiefel feature-steepest descent."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK
from gn_experiment import _read_metrics
from gpt_baseline_selection import selected_baseline_label
from low_spectral_variance_experiment import _baseline_records, low_spectral_variance_task
from stiefel_feature import (
    InputCovarianceCollector,
    column_polar,
    feature_polar_update,
    stiefel_feature_parameter_names,
)
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


@dataclass(frozen=True)
class StiefelFeaturePaths:
    metric: Path
    result: Path
    checkpoint: Path


def stiefel_feature_paths(root: Path, label: str) -> StiefelFeaturePaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__stiefel_feature_{label}"
    return StiefelFeaturePaths(
        metric=root / "metrics" / "nlp" / f"{stem}.jsonl",
        result=root / "results" / "nlp" / f"{stem}.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def stiefel_feature_task(*, gradient_accumulation: int):
    return low_spectral_variance_task(gradient_accumulation=gradient_accumulation)


def write_stiefel_feature_comparison_plots(
    root: Path, *, label: str
) -> tuple[Path, Path] | None:
    paths = stiefel_feature_paths(root, label)
    candidate = _read_metrics(paths.metric) if paths.metric.exists() else []
    traces = [
        (selected_baseline_label("adamw"), _baseline_records(root, "adamw")),
        (selected_baseline_label("muon"), _baseline_records(root, "muon")),
        (selected_baseline_label("muown"), _baseline_records(root, "muown")),
        ("Stiefel Feature-Steepest Descent", candidate),
    ]
    if not all(records for _, records in traces):
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"stiefel_feature_{label}_metric_steps.png",
        output_root / f"stiefel_feature_{label}_metric_time.png",
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
                [record["metric"] for record in values], label=name,
            )
        axis.set(
            xlabel=xlabel,
            ylabel="Validation next-token accuracy",
            title="Stiefel Feature-Steepest Descent versus tuned GPT-2 baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def _selected_linear_modules(model: torch.nn.Module, names: set[str]) -> dict[str, torch.nn.Linear]:
    modules = {
        f"{module_name}.weight": module
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and f"{module_name}.weight" in names
    }
    if set(modules) != names:
        raise ValueError("selected Stiefel parameter names do not resolve to linear modules")
    return modules


def run_stiefel_feature_trial(
    root: Path,
    *,
    label: str,
    alpha: float,
    auxiliary_learning_rate: float | None = None,
    workers: int = 4,
    maximum_seconds: float = 14_400.0,
    maximum_epochs: int | None = None,
    gradient_accumulation: int = 4,
    seed: int = 1337,
    fresh: bool = False,
    write_plots: bool = False,
) -> dict:
    """Run real-covariance polar updates plus AdamW auxiliary parameters."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if maximum_seconds <= 0:
        raise ValueError("maximum seconds must be positive")
    task = stiefel_feature_task(gradient_accumulation=gradient_accumulation)
    target_epochs = task.estimated_epochs if maximum_epochs is None else maximum_epochs
    if not 0 < target_epochs <= task.estimated_epochs:
        raise ValueError("maximum epochs must lie within the five-epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stiefel feature descent")
    paths = stiefel_feature_paths(root, label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    selected_names = stiefel_feature_parameter_names(model)
    modules = _selected_linear_modules(model, selected_names)
    named = dict(model.named_parameters())
    selected_ids = {id(named[name]) for name in selected_names}
    auxiliary = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in selected_ids]
    auxiliary_lr = task.muon_aux_lr if auxiliary_learning_rate is None else auxiliary_learning_rate
    if auxiliary_lr <= 0:
        raise ValueError("auxiliary learning rate must be positive")
    auxiliary_optimizer = torch.optim.AdamW(
        auxiliary, lr=auxiliary_lr, weight_decay=task.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(auxiliary_optimizer, T_max=task.estimated_epochs)
    completed_epoch = completed_steps = 0
    elapsed_seconds = 0.0
    checkpoint = None
    if paths.checkpoint.exists():
        checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        auxiliary_optimizer.load_state_dict(checkpoint["auxiliary_optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        completed_epoch = int(checkpoint["epoch"])
        completed_steps = int(checkpoint["steps"])
        elapsed_seconds = float(checkpoint["elapsed_seconds"])
    else:
        with torch.no_grad():
            for name in selected_names:
                named[name].copy_(column_polar(named[name]))
    collector = InputCovarianceCollector(modules)
    steps_per_epoch = len(train_loader) // task.gradient_accumulation
    partial_steps = completed_steps - completed_epoch * steps_per_epoch
    if not 0 <= partial_steps < steps_per_epoch:
        collector.close()
        raise ValueError("checkpoint has an invalid partial-epoch optimizer-step cursor")
    if partial_steps:
        retained = [record for record in _read_metrics(paths.metric) if int(record["epoch"]) <= completed_epoch]
        paths.metric.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in retained))
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    timed_out = False
    last_budget = 0.0
    last_orthogonality = 0.0
    try:
        for epoch in range(completed_epoch + 1, target_epochs + 1):
            model.train()
            for batch_index, batch in enumerate(train_loader):
                if epoch == completed_epoch + 1 and batch_index < partial_steps * task.gradient_accumulation:
                    continue
                if batch_index % task.gradient_accumulation == 0:
                    auxiliary_optimizer.zero_grad(set_to_none=True)
                (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
                if (batch_index + 1) % task.gradient_accumulation:
                    continue
                budgets, errors = [], []
                with torch.no_grad():
                    for name in sorted(selected_names):
                        covariance = collector.consume(name)
                        if covariance is None:
                            raise RuntimeError(f"missing input covariance for {name}")
                        parameter = named[name]
                        updated, diagnostics = feature_polar_update(parameter, parameter.grad, covariance, alpha=alpha)
                        parameter.copy_(updated)
                        budgets.append(diagnostics.feature_budget)
                        errors.append(diagnostics.orthogonality_error)
                auxiliary_optimizer.step()
                last_budget = sum(budgets) / len(budgets)
                last_orthogonality = max(errors)
                completed_steps += 1
                elapsed_seconds += time.perf_counter() - started
                started = time.perf_counter()
                if elapsed_seconds >= maximum_seconds:
                    timed_out = True
                    break
            if not timed_out:
                scheduler.step()
                completed_epoch = epoch
                partial_steps = 0
            collector.clear()
            metric = _evaluate(task, model, validation_loader, device)
            collector.clear()
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            write_metric(paths.metric, {
                "epoch": epoch, "step": completed_steps, "metric": metric,
                "elapsed_seconds": elapsed_seconds, "alpha": alpha,
                "stiefel_matrix_count": len(selected_names),
                "mean_feature_budget": last_budget,
                "max_orthogonality_error": last_orthogonality,
            })
            paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(), "auxiliary_optimizer": auxiliary_optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": completed_epoch,
                "steps": completed_steps, "elapsed_seconds": elapsed_seconds,
            }, paths.checkpoint)
            if timed_out:
                break
    finally:
        collector.close()
    records = _read_metrics(paths.metric)
    result = {
        "task": task.identifier, "domain": task.domain, "model": task.model,
        "optimizer": "stiefel_feature_steepest_descent", "parameters": sum(p.numel() for p in model.parameters()),
        "alpha": alpha, "auxiliary_learning_rate": auxiliary_lr,
        "micro_batch_size": task.micro_batch_size, "gradient_accumulation": task.gradient_accumulation,
        "stiefel_matrix_count": len(selected_names),
        "adamw_auxiliary_parameter_count": sum(p.numel() for p in auxiliary),
        "completed_epochs": completed_epoch, "completed_steps": completed_steps,
        "final_metric": records[-1]["metric"], "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "time_limit_checkpointed" if timed_out else ("completed" if target_epochs == task.estimated_epochs else "screen_complete"),
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if write_plots:
        write_stiefel_feature_comparison_plots(root, label=label)
    return result
