"""Matched-schedule GPT2-12x512 evaluation for Multi-Step Spectral Geometry."""

from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from gn_experiment import _read_metrics
from gn_experiment import LANGUAGE_MODEL_GN_TASK, language_model_task
from gpt_baseline_selection import selected_baseline_label
from low_spectral_variance_experiment import _baseline_records, _save_checkpoint, low_spectral_variance_task
from multi_step_spectral_geometry import MultiStepSpectralOptimizer, SpectralPolicyConfig
from optimizers import muon_parameter_names
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


@dataclass(frozen=True)
class MultiStepSpectralPaths:
    metric: Path
    result: Path
    checkpoint: Path


def multi_step_spectral_paths(root: Path, label: str) -> MultiStepSpectralPaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__multi_step_spectral_{label}"
    return MultiStepSpectralPaths(
        metric=root / "metrics" / "nlp" / f"{stem}.jsonl",
        result=root / "results" / "nlp" / f"{stem}.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def matched_multi_step_task(*, micro_batch_size: int = 12, gradient_accumulation: int):
    """Return the exact five-epoch GPT2-12x512 baseline workload."""
    return language_model_task(
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
    )


def build_multi_step_spectral_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    auxiliary_learning_rate: float,
    weight_decay: float,
    momentum: float,
    initial_p: float = math.inf,
    warmup_steps: int = 500,
    policy_interval: int = 100,
    switch_margin: float = 0.005,
) -> tuple[MultiStepSpectralOptimizer, set[str]]:
    """Route the same matrices as Muon through the fetched spectral policy."""
    if (
        learning_rate <= 0
        or auxiliary_learning_rate <= 0
        or weight_decay < 0
        or not 0 <= momentum < 1
        or warmup_steps < 0
        or policy_interval <= 0
        or switch_margin < 0
    ):
        raise ValueError("Multi-Step Spectral Geometry hyperparameters are invalid")
    selected = muon_parameter_names(model)
    named = dict(model.named_parameters())
    matrices = [named[name] for name in sorted(selected) if named[name].requires_grad]
    matrix_ids = {id(parameter) for parameter in matrices}
    auxiliary = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in matrix_ids
    ]
    if not matrices:
        raise ValueError("no Muon-eligible matrices are available for the spectral policy")
    config = SpectralPolicyConfig(
        lr=learning_rate,
        fallback_lr=auxiliary_learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        matrix_scale="moonlight",
        candidates=(2.0, 4.0, 8.0, math.inf),
        transform_backend="reduced",
        ns_steps=8,
        polynomial_degree=12,
        polynomial_floor=1.0e-4,
        warmup_steps=warmup_steps,
        policy_interval=policy_interval,
        horizon=4,
        horizon_weights=(1.0, 0.8, 0.6, 0.4),
        switch_penalty=0.01,
        switch_margin=switch_margin,
        compute_penalty=0.0,
        history_size=8,
        basis_rank=8,
        fit_interval=50,
        ridge=1.0e-4,
        min_curvature=0.0,
        max_curvature=100.0,
        perpendicular_curvature="median_secant",
        initial_p=initial_p,
    )
    return (
        MultiStepSpectralOptimizer(
            [
                {"params": matrices, "use_spectral": True},
                {"params": auxiliary, "use_spectral": False},
            ],
            config,
        ),
        selected,
    )


def write_multi_step_spectral_comparison_plots(
    root: Path, *, label: str
) -> tuple[Path, Path] | None:
    paths = multi_step_spectral_paths(root, label)
    candidate = _read_metrics(paths.metric) if paths.metric.exists() else []
    traces = [
        (selected_baseline_label("adamw"), _baseline_records(root, "adamw")),
        (selected_baseline_label("muon"), _baseline_records(root, "muon")),
        (selected_baseline_label("muown"), _baseline_records(root, "muown")),
        ("Multi-Step Spectral Geometry", candidate),
    ]
    if not all(records for _, records in traces):
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"multi_step_spectral_{label}_metric_steps.png",
        output_root / f"multi_step_spectral_{label}_metric_time.png",
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
            title="Multi-Step Spectral Geometry versus tuned GPT-2 baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def _diagnostic_summary(optimizer: MultiStepSpectralOptimizer) -> dict[str, object]:
    diagnostics = optimizer.last_diagnostics
    if not diagnostics:
        return {"policy_matrix_count": 0, "selected_p": {}, "switches": 0}
    selected: dict[str, int] = {}
    for item in diagnostics:
        value = str(item["selected_p"])
        selected[value] = selected.get(value, 0) + 1
    return {
        "policy_matrix_count": len(diagnostics),
        "selected_p": selected,
        "switches": sum(int(item["switches"]) for item in diagnostics),
        "policy_seconds": sum(float(item["policy_seconds"]) for item in diagnostics),
        "transform_seconds": sum(float(item["transform_seconds"]) for item in diagnostics),
    }


def run_multi_step_spectral_trial(
    root: Path,
    *,
    label: str,
    learning_rate: float,
    auxiliary_learning_rate: float | None = None,
    momentum: float = 0.95,
    initial_p: float = math.inf,
    warmup_steps: int = 500,
    policy_interval: int = 100,
    switch_margin: float = 0.005,
    workers: int = 4,
    maximum_seconds: float = 14_400.0,
    maximum_epochs: int | None = None,
    micro_batch_size: int = 12,
    gradient_accumulation: int = 4,
    seed: int = 1337,
    fresh: bool = False,
    write_plots: bool = False,
) -> dict:
    """Run the formula-compatible policy on the baseline-matched GPT workload."""
    if learning_rate <= 0 or not 0 <= momentum < 1 or maximum_seconds <= 0:
        raise ValueError("Multi-Step Spectral Geometry hyperparameters are invalid")
    task = matched_multi_step_task(
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    target_epochs = task.estimated_epochs if maximum_epochs is None else maximum_epochs
    if not 0 < target_epochs <= task.estimated_epochs:
        raise ValueError("maximum epochs must lie within the five-epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Multi-Step Spectral Geometry")
    paths = multi_step_spectral_paths(root, label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    resolved_auxiliary_learning_rate = (
        task.muon_aux_lr if auxiliary_learning_rate is None else auxiliary_learning_rate
    )
    optimizer, spectral_names = build_multi_step_spectral_optimizer(
        model,
        learning_rate=learning_rate,
        auxiliary_learning_rate=resolved_auxiliary_learning_rate,
        weight_decay=task.weight_decay,
        momentum=momentum,
        initial_p=initial_p,
        warmup_steps=warmup_steps,
        policy_interval=policy_interval,
        switch_margin=switch_margin,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=task.estimated_epochs
    )
    completed_epoch = completed_steps = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        payload = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        completed_epoch = int(payload["epoch"])
        completed_steps = int(payload["steps"])
        elapsed_seconds = float(payload["elapsed_seconds"])
    steps_per_epoch = len(train_loader) // task.gradient_accumulation
    partial_steps = completed_steps - completed_epoch * steps_per_epoch
    if not 0 <= partial_steps < steps_per_epoch:
        raise ValueError("checkpoint has an invalid partial-epoch optimizer-step cursor")
    if partial_steps:
        retained = [
            record for record in _read_metrics(paths.metric)
            if int(record["epoch"]) <= completed_epoch
        ]
        paths.metric.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in retained)
        )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    timed_out = False
    final_diagnostics: dict[str, object] = {}
    for epoch in range(completed_epoch + 1, target_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if epoch == completed_epoch + 1 and batch_index < partial_steps * task.gradient_accumulation:
                continue
            if batch_index % task.gradient_accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation:
                continue
            optimizer.step()
            final_diagnostics = _diagnostic_summary(optimizer)
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
                "momentum": momentum,
                "spectral_matrix_count": len(spectral_names),
                **final_diagnostics,
            },
        )
        paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": completed_epoch,
                "steps": completed_steps,
                "elapsed_seconds": elapsed_seconds,
            },
            paths.checkpoint,
        )
        if timed_out:
            break
    records = _read_metrics(paths.metric)
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "multi_step_spectral_geometry",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": learning_rate,
        "auxiliary_learning_rate": resolved_auxiliary_learning_rate,
        "momentum": momentum,
        "matrix_scale": optimizer.config.matrix_scale,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "spectral_matrix_count": len(spectral_names),
        "adamw_auxiliary_parameter_count": sum(
            parameter.numel() for name, parameter in model.named_parameters() if name not in spectral_names
        ),
        "completed_epochs": completed_epoch,
        "completed_steps": completed_steps,
        "final_metric": records[-1]["metric"],
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "time_limit_checkpointed" if timed_out else (
            "completed" if target_epochs == task.estimated_epochs else "screen_complete"
        ),
        "last_policy_diagnostics": final_diagnostics,
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if write_plots:
        write_multi_step_spectral_comparison_plots(root, label=label)
    return result
