"""Matched-schedule GPT-2 evaluation for the strict-feasible SRIP band."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK, _read_metrics
from gpt_baseline_selection import selected_baseline_label
from low_spectral_variance_experiment import _baseline_records, _save_checkpoint, low_spectral_variance_task
from optimizers import build_optimizers, muon_parameter_names
from srip_band import srip_band_parameter_names
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


@dataclass(frozen=True)
class SRIPBandPaths:
    metric: Path
    result: Path
    checkpoint: Path


def srip_band_paths(root: Path, label: str) -> SRIPBandPaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__srip_band_{label}"
    return SRIPBandPaths(
        root / "metrics" / "nlp" / f"{stem}.jsonl",
        root / "results" / "nlp" / f"{stem}.json",
        root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def write_srip_band_comparison_plots(root: Path, *, label: str) -> tuple[Path, Path] | None:
    paths = srip_band_paths(root, label)
    candidate = _read_metrics(paths.metric) if paths.metric.exists() else []
    traces = [
        (selected_baseline_label("adamw"), _baseline_records(root, "adamw")),
        (selected_baseline_label("muon"), _baseline_records(root, "muon")),
        (selected_baseline_label("muown"), _baseline_records(root, "muown")),
        ("SRIP-band strict-feasible", candidate),
    ]
    if not all(records for _, records in traces):
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"srip_band_{label}_metric_steps.png",
        output_root / f"srip_band_{label}_metric_time.png",
    )
    for output, key, xlabel in (
        (outputs[0], "step", "Completed optimizer step"),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)"),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for name, records in traces:
            axis.plot(
                [record[key] / 3600 if key == "elapsed_seconds" else record[key] for record in records],
                [record["metric"] for record in records], label=name,
            )
        axis.set(xlabel=xlabel, ylabel="Validation next-token accuracy",
                 title="SRIP-band strict-feasible versus tuned GPT-2 baselines")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def run_srip_band_trial(
    root: Path, *, label: str, learning_rate: float, rho: float, dual_steps: int = 8,
    momentum: float = 0.95, workers: int = 4, maximum_seconds: float = 14_400.0,
    maximum_epochs: int | None = None, gradient_accumulation: int = 4,
    seed: int = 1337, fresh: bool = False, write_plots: bool = False,
) -> dict:
    """Run SRIP-band and AdamW auxiliary parameters on the fixed GPT workload."""
    if learning_rate <= 0 or not 0 < rho < 1 or dual_steps <= 0 or not 0 <= momentum < 1:
        raise ValueError("SRIP-band hyperparameters are invalid")
    if maximum_seconds <= 0:
        raise ValueError("maximum_seconds must be positive")
    task = low_spectral_variance_task(gradient_accumulation=gradient_accumulation)
    target_epochs = task.estimated_epochs if maximum_epochs is None else maximum_epochs
    if not 0 < target_epochs <= task.estimated_epochs:
        raise ValueError("maximum epochs must lie within the five-epoch budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SRIP-band")
    paths = srip_band_paths(root, label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    constrained_names = srip_band_parameter_names(
        model, rho=rho, candidates=muon_parameter_names(model)
    )
    if not constrained_names:
        raise ValueError("no GPT-2 matrix is initially feasible for the SRIP band")
    optimizers = build_optimizers(
        model, "srip_band", learning_rate, task.weight_decay, task.muon_aux_lr,
        srip_rho=rho, srip_dual_steps=dual_steps,
    )
    for group in optimizers["srip_band"].param_groups:
        group["momentum"] = momentum
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=task.estimated_epochs)
                  for name, optimizer in optimizers.items()}
    completed_epoch = completed_steps = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        payload = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(payload["schedulers"][name])
        completed_epoch, completed_steps = int(payload["epoch"]), int(payload["steps"])
        elapsed_seconds = float(payload["elapsed_seconds"])
    steps_per_epoch = len(train_loader) // task.gradient_accumulation
    partial_steps = completed_steps - completed_epoch * steps_per_epoch
    if not 0 <= partial_steps < steps_per_epoch:
        raise ValueError("checkpoint has an invalid partial-epoch optimizer-step cursor")
    if partial_steps:
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
        write_metric(paths.metric, {"epoch": epoch, "step": completed_steps, "metric": metric,
                     "elapsed_seconds": elapsed_seconds, "learning_rate": learning_rate,
                     "rho": rho, "dual_steps": dual_steps,
                     "constrained_matrix_count": len(constrained_names)})
        _save_checkpoint(paths.checkpoint, model, optimizers, schedulers, epoch=completed_epoch,
                         steps=completed_steps, elapsed_seconds=elapsed_seconds)
        if timed_out:
            break
    records = _read_metrics(paths.metric)
    result = {"task": task.identifier, "domain": task.domain, "model": task.model,
              "optimizer": "srip_band", "parameters": sum(p.numel() for p in model.parameters()),
              "learning_rate": learning_rate, "rho": rho, "dual_steps": dual_steps, "momentum": momentum,
              "micro_batch_size": task.micro_batch_size, "gradient_accumulation": task.gradient_accumulation,
              "constrained_matrix_count": len(constrained_names),
              "adamw_auxiliary_parameter_count": sum(p.numel() for name, p in model.named_parameters() if name not in constrained_names),
              "completed_epochs": completed_epoch, "completed_steps": completed_steps,
              "final_metric": records[-1]["metric"], "seconds": elapsed_seconds,
              "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
              "status": "time_limit_checkpointed" if timed_out else ("completed" if target_epochs == task.estimated_epochs else "screen_complete")}
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if write_plots:
        write_srip_band_comparison_plots(root, label=label)
    return result
