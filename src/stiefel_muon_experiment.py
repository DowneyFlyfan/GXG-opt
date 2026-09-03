"""Checkpointable Stiefel-Muon trial on the retained GPT-12x512 baseline."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK, _read_metrics
from optimizers import build_optimizers, muon_parameter_names
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


@dataclass(frozen=True)
class StiefelMuonPaths:
    metric: Path
    result: Path
    checkpoint: Path


def stiefel_task(*, micro_batch_size: int, gradient_accumulation: int):
    if micro_batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("microbatch and accumulation must be positive")
    return replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
    )


def stiefel_muon_paths(root: Path, label: str) -> StiefelMuonPaths:
    return constrained_muon_paths(root, optimizer_name="stiefel_muon", label=label)


def spectral_sphere_muon_paths(root: Path, label: str) -> StiefelMuonPaths:
    return constrained_muon_paths(
        root, optimizer_name="spectral_sphere_muon", label=label
    )


def constrained_muon_paths(
    root: Path, *, optimizer_name: str, label: str
) -> StiefelMuonPaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__{optimizer_name}_{label}"
    return StiefelMuonPaths(
        metric=root / "metrics" / "nlp" / f"{stem}.jsonl",
        result=root / "results" / "nlp" / f"{stem}.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def _save_checkpoint(
    path: Path,
    model,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    *,
    epoch: int,
    batch: int,
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
            "batch": batch,
            "steps": steps,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def _load_checkpoint(path: Path, model, optimizers, schedulers) -> tuple[int, int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(payload["schedulers"][name])
    return (
        int(payload["epoch"]),
        int(payload["batch"]),
        int(payload["steps"]),
        float(payload["elapsed_seconds"]),
    )


def _historical_baseline_records(root: Path, optimizer: str) -> list[dict]:
    """Put retained epoch metrics on their actual optimizer-step timeline."""
    metric_path = root / "metrics" / "nlp" / f"nlp_gpt_12x512__{optimizer}.jsonl"
    result_path = root / "results" / "nlp" / f"nlp_gpt_12x512__{optimizer}.json"
    records = _read_metrics(metric_path)
    if not records or "step" in records[0]:
        return records
    result = json.loads(result_path.read_text())
    steps_per_epoch = math.ceil(12_207 / int(result["gradient_accumulation"]))
    return [
        {
            **record,
            "step": int(record["epoch"]) * steps_per_epoch,
            "elapsed_seconds": result["seconds"]
            * int(record["epoch"])
            / int(result["epochs"]),
        }
        for record in records
    ]


def _needs_epoch_end_evaluation(*, steps: int, last_evaluated_steps: int) -> bool:
    """A partial accumulation tail must not hide the final completed update."""
    return steps > last_evaluated_steps


def _write_constrained_muon_comparison_plots(
    root: Path, *, optimizer_name: str, display_name: str, label: str
) -> tuple[Path, Path] | None:
    candidate = constrained_muon_paths(root, optimizer_name=optimizer_name, label=label)
    if not candidate.metric.exists():
        return None
    traces = []
    for display, optimizer in (("AdamW", "adamw"), ("Muon", "muon")):
        baseline = root / "metrics" / "nlp" / f"nlp_gpt_12x512__{optimizer}.jsonl"
        if baseline.exists():
            traces.append((display, _historical_baseline_records(root, optimizer)))
    traces.append((display_name, _read_metrics(candidate.metric)))
    if len(traces) != 3:
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"{optimizer_name}_{label}_metric_steps.png",
        output_root / f"{optimizer_name}_{label}_metric_time.png",
    )
    for output, key, xlabel in (
        (outputs[0], "step", "Optimizer step"),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)"),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for display, records in traces:
            values = [record for record in records if key in record]
            x_values = [record[key] / 3600 if key == "elapsed_seconds" else record[key] for record in values]
            axis.plot(x_values, [record["metric"] for record in values], label=display)
        axis.set(xlabel=xlabel, ylabel="Validation next-token accuracy")
        axis.set_title(f"{display_name} versus retained AdamW and Muon baselines")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def write_stiefel_muon_comparison_plots(root: Path, *, label: str) -> tuple[Path, Path] | None:
    return _write_constrained_muon_comparison_plots(
        root,
        optimizer_name="stiefel_muon",
        display_name="Stiefel-Muon",
        label=label,
    )


def write_spectral_sphere_muon_comparison_plots(
    root: Path, *, label: str
) -> tuple[Path, Path] | None:
    return _write_constrained_muon_comparison_plots(
        root,
        optimizer_name="spectral_sphere_muon",
        display_name="Spectral-Sphere Muon",
        label=label,
    )


def run_stiefel_muon_trial(
    root: Path,
    *,
    label: str,
    learning_rate: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
    workers: int = 4,
    maximum_seconds: float = 14_400.0,
    maximum_steps: int | None = None,
    evaluation_interval_steps: int = 1_024,
    micro_batch_size: int = 4,
    gradient_accumulation: int = 2,
    training_epochs: int | None = None,
    scheduler_epochs: int | None = None,
    seed: int = 1337,
    fresh: bool = False,
    optimizer_name: str = "stiefel_muon",
) -> dict:
    """Run a constrained-Muon variant until the epoch, step, or time boundary."""
    if learning_rate <= 0 or not 0 <= momentum < 1 or ns_steps <= 0:
        raise ValueError("learning rate, momentum, and Newton--Schulz steps are invalid")
    if maximum_seconds <= 0 or evaluation_interval_steps <= 0:
        raise ValueError("time and evaluation intervals must be positive")
    if maximum_steps is not None and maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive when provided")
    if training_epochs is not None and training_epochs <= 0:
        raise ValueError("training_epochs must be positive when provided")
    if scheduler_epochs is not None and scheduler_epochs <= 0:
        raise ValueError("scheduler_epochs must be positive when provided")
    if optimizer_name not in {"stiefel_muon", "spectral_sphere_muon"}:
        raise ValueError("unsupported constrained Muon optimizer")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stiefel-Muon")

    task = stiefel_task(
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    epoch_limit = task.estimated_epochs if training_epochs is None else training_epochs
    scheduler_horizon = epoch_limit if scheduler_epochs is None else scheduler_epochs
    paths = constrained_muon_paths(root, optimizer_name=optimizer_name, label=label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    selected_names = muon_parameter_names(model)
    optimizers = build_optimizers(
        model,
        optimizer_name,
        learning_rate,
        task.weight_decay,
        task.muon_aux_lr,
    )
    for group in optimizers[optimizer_name].param_groups:
        group["momentum"] = momentum
        group["ns_steps"] = ns_steps
    schedulers = {
        name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_horizon)
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
    while epoch <= epoch_limit and not stop:
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
                write_metric(
                    paths.metric,
                    {
                        "epoch": epoch,
                        "step": steps,
                        "metric": final_metric,
                        "elapsed_seconds": elapsed_seconds,
                        "learning_rate": learning_rate,
                        "momentum": momentum,
                        "ns_steps": ns_steps,
                    },
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
                _write_constrained_muon_comparison_plots(
                    root,
                    optimizer_name=optimizer_name,
                    display_name=(
                        "Stiefel-Muon"
                        if optimizer_name == "stiefel_muon"
                        else "Spectral-Sphere Muon"
                    ),
                    label=label,
                )
                print(
                    f"task={task.identifier} optimizer={optimizer_name} step={steps} "
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
            write_metric(
                paths.metric,
                {
                    "epoch": epoch,
                    "step": steps,
                    "metric": final_metric,
                    "elapsed_seconds": elapsed_seconds,
                    "learning_rate": learning_rate,
                    "momentum": momentum,
                    "ns_steps": ns_steps,
                },
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
            _write_constrained_muon_comparison_plots(
                root,
                optimizer_name=optimizer_name,
                display_name=(
                    "Stiefel-Muon"
                    if optimizer_name == "stiefel_muon"
                    else "Spectral-Sphere Muon"
                ),
                label=label,
            )
            print(
                f"task={task.identifier} optimizer={optimizer_name} step={steps} "
                f"metric={final_metric:.6f} elapsed={elapsed_seconds:.1f}",
                flush=True,
            )
        for scheduler in schedulers.values():
            scheduler.step()
        epoch += 1
        resume_batch = 0

    if final_metric is None:
        final_metric = _evaluate(task, model, validation_loader, device)
    status = "completed" if epoch > epoch_limit else "time_limit_checkpointed"
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": optimizer_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        # Both variants constrain exactly the Muon-eligible matrix set.  Naming
        # this generically prevents a spectral-sphere result from being
        # misreported as a Stiefel experiment.
        "constrained_matrix_count": len(selected_names),
        "learning_rate": learning_rate,
        "momentum": momentum,
        "ns_steps": ns_steps,
        "direction_msign": "newton_schulz",
        "lambda_steps": 10 if optimizer_name == "spectral_sphere_muon" else None,
        "spectral_power_iterations": (
            10 if optimizer_name == "spectral_sphere_muon" else None
        ),
        "training_epochs": epoch_limit,
        "scheduler_epochs": scheduler_horizon,
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
    _write_constrained_muon_comparison_plots(
        root,
        optimizer_name=optimizer_name,
        display_name=(
            "Stiefel-Muon"
            if optimizer_name == "stiefel_muon"
            else "Spectral-Sphere Muon"
        ),
        label=label,
    )
    return result


def run_spectral_sphere_muon_trial(root: Path, **kwargs) -> dict:
    """Run the article-11241 optimizer with the retained trial protocol."""
    return run_stiefel_muon_trial(
        root, optimizer_name="spectral_sphere_muon", **kwargs
    )
