from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(
    Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plot
import torch

from artifacts import write_metric
from config import FormalTask
from gn_training import nlp_mc_ggn_update
from optimizer_registry import build_optimizer
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


# This is the retained 54.68M language-model baseline.  Its cached WikiText-103
# stream and AdamW/Muon traces already exist, so GN measurements can be compared
# without changing the model, tokenizer, data order, or token budget.
LANGUAGE_MODEL_GN_TASK = FormalTask(
    "nlp_gpt_12x512",
    "nlp",
    "gpt_12x512",
    5,
    4,
    2,
    3.0e-4,
    3.0e-4,
    weight_decay=0.01,
)


def language_model_task(
    *, micro_batch_size: int | None = None, gradient_accumulation: int | None = None
) -> FormalTask:
    """Return the retained task with an explicitly selected effective batch."""
    if micro_batch_size is not None and micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if gradient_accumulation is not None and gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    return replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=(
            LANGUAGE_MODEL_GN_TASK.micro_batch_size
            if micro_batch_size is None
            else micro_batch_size
        ),
        gradient_accumulation=(
            LANGUAGE_MODEL_GN_TASK.gradient_accumulation
            if gradient_accumulation is None
            else gradient_accumulation
        ),
    )


def _common_config(overrides: dict | None = None) -> dict:
    config = {
        "learning_rate": 0.03,
        "curvature_mode": "mc_ggn",
        "damping": 0.03,
        "factor_decay": 0.95,
        "factor_update_interval": 64,
        "spectral_update_interval": 256,
        "fallback_learning_rate": 3.0e-4,
        "fallback_betas": (0.9, 0.95),
        "weight_decay": LANGUAGE_MODEL_GN_TASK.weight_decay,
        "linear_algebra_dtype": "float32",
        "trust_clip": 0.02,
    }
    if overrides:
        config.update(overrides)
    return config


def build_gn_optimizer(model, optimizer_name: str, *, config_overrides: dict | None = None):
    """Build the separately tuned baseline GN or low-rank DG-GN optimizer."""
    config = _common_config(config_overrides)
    if optimizer_name == "kronecker_ggn":
        return build_optimizer(optimizer_name, model, config)
    if optimizer_name == "low_rank_corrected_kronecker_ggn":
        config.update(
            {
                "correction_rank": 1,
                # A rank-one Ritz space cannot certify its own residual.  The
                # eight-dimensional pilot gave a 0.0088 relative residual,
                # whereas three dimensions gave 0.249; retain rank one but
                # use eight Krylov vectors and accept only residuals <= 0.02.
                "correction_oversampling": 7,
                "lanczos_steps": 8,
                "lanczos_tolerance": 0.02,
                "correction_warmup_steps": 2048,
                "correction_refresh_interval": 2048,
                "correction_max_age": 4096,
                "correction_memory_budget_mb": 512.0,
                "active_layer_policy": "largest_parameter_count",
                "active_layer_count": 1,
            }
        )
        return build_optimizer(optimizer_name, model, config)
    raise ValueError(f"Unsupported GN experiment optimizer: {optimizer_name}")


@dataclass(frozen=True)
class ArtifactPaths:
    metric: Path
    result: Path
    checkpoint: Path


def artifact_paths(root: Path, optimizer_name: str, *, run_label: str | None = None) -> ArtifactPaths:
    stem = f"{LANGUAGE_MODEL_GN_TASK.identifier}__{run_label or optimizer_name}"
    return ArtifactPaths(
        metric=root / "metrics" / "nlp" / f"{stem}.jsonl",
        result=root / "results" / "nlp" / f"{stem}.json",
        checkpoint=root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def _checkpoint(
    path: Path,
    model,
    optimizer,
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
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "completed_steps": completed_steps,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def _restore(path: Path, model, optimizer) -> tuple[int, int, int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return (
        int(state["next_epoch"]),
        int(state["next_batch"]),
        int(state["completed_steps"]),
        float(state["elapsed_seconds"]),
    )


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _comparison_records(
    root: Path, optimizer_name: str, *, run_label: str | None = None
) -> list[dict]:
    paths = artifact_paths(root, optimizer_name, run_label=run_label)
    records = _read_metrics(paths.metric)
    if not records or "step" in records[0]:
        return records
    result = json.loads(paths.result.read_text())
    steps_per_epoch = 12_207
    return [
        {
            **record,
            "step": int(record["epoch"]) * steps_per_epoch,
            "elapsed_seconds": result["seconds"] * int(record["epoch"]) / result["epochs"],
        }
        for record in records
    ]


def write_gn_comparison_plots(
    root: Path,
    *,
    guided_run_label: str | None = None,
    full_ggn_run_label: str | None = None,
) -> tuple[Path, Path] | None:
    names = [("AdamW", "adamw", None), ("Muon", "muon", None)]
    if guided_run_label is None:
        names.extend(
            [
                ("Full GGN", "full_ggn", full_ggn_run_label),
                ("Kronecker GGN", "kronecker_ggn", None),
                ("Low-rank DG-GN", "low_rank_corrected_kronecker_ggn", None),
            ]
        )
    traces = []
    for label, name, run_label in names:
        path = artifact_paths(root, name, run_label=run_label).metric
        if path.exists():
            traces.append(
                (label, _comparison_records(root, name, run_label=run_label))
            )
    if guided_run_label is not None:
        guided_path = artifact_paths(
            root, "gn_guided_adamw", run_label=guided_run_label
        ).metric
        if guided_path.exists():
            traces.append(("GGN-guided AdamW", _read_metrics(guided_path)))
    if len(traces) < 3:
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / "nlp_gpt_12x512_gn_metric_steps.png",
        output_root / "nlp_gpt_12x512_gn_metric_time.png",
    )
    for output, axis_name, record_name in zip(
        outputs,
        ("Optimizer step", "Wall-clock time (hours)"),
        ("step", "elapsed_seconds"),
        strict=True,
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for label, records in traces:
            if records:
                x_values = [
                    item[record_name] / 3600
                    if record_name == "elapsed_seconds"
                    else item[record_name]
                    for item in records
                ]
                axis.plot(x_values, [item["metric"] for item in records], label=label)
        axis.set(xlabel=axis_name, ylabel="Validation next-token accuracy")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def run_gn_trial(
    root: Path,
    optimizer_name: str,
    *,
    workers: int = 4,
    maximum_seconds: float = 14_400.0,
    evaluation_interval_steps: int = 2048,
    seed: int = 1337,
    fresh: bool = False,
    micro_batch_size: int | None = None,
    gradient_accumulation: int | None = None,
    config_overrides: dict | None = None,
    run_label: str | None = None,
) -> dict:
    """Run a checkpointable GN/DG-GN language-model trial for at most four hours."""
    if optimizer_name not in {"kronecker_ggn", "low_rank_corrected_kronecker_ggn"}:
        raise ValueError(f"Unsupported GN experiment optimizer: {optimizer_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the language-model GN experiment")
    task = language_model_task(
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    configure_reproducibility(seed)
    device = torch.device("cuda")
    paths = artifact_paths(root, optimizer_name, run_label=run_label)
    if fresh:
        # A curvature implementation change invalidates both the optimizer
        # state and its metric trace.  Restart from the same data seed rather
        # than silently resuming an incompatible checkpoint.
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    optimizer = build_gn_optimizer(model, optimizer_name, config_overrides=config_overrides)
    next_epoch, resume_batch, completed_steps, elapsed_seconds = (1, 0, 0, 0.0)
    if paths.checkpoint.exists():
        next_epoch, resume_batch, completed_steps, elapsed_seconds = _restore(
            paths.checkpoint, model, optimizer
        )
    elif paths.metric.exists():
        paths.metric.unlink()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    include_exact_operator = optimizer_name == "low_rank_corrected_kronecker_ggn"
    for epoch in range(next_epoch, task.estimated_epochs + 1):
        model.train()
        step_batches = []
        for batch_index, batch in enumerate(train_loader):
            if epoch == next_epoch and batch_index < resume_batch:
                continue
            if batch_index % task.gradient_accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                step_batches = []
                if optimizer.should_update_curvature():
                    optimizer.update_curvature(
                        lambda current_model, current_batch, registry: nlp_mc_ggn_update(
                            current_model,
                            current_batch,
                            registry,
                            include_exact_residual_operator=include_exact_operator,
                            seed=seed + completed_steps,
                            residual_batch_size=1,
                            residual_sequence_length=128,
                        ),
                        batch,
                    )
            step_batches.append(batch)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            completed_batch = (batch_index + 1) % task.gradient_accumulation == 0
            if not completed_batch:
                continue
            acceptance_closure = None
            if (
                optimizer.config.adaptive_damping
                and optimizer.step_count
                % optimizer.config.damping_adaptation_interval
                == 0
            ):
                acceptance_batches = tuple(step_batches)

                def acceptance_closure():
                    return sum(
                        _loss(task, model, item, device)
                        for item in acceptance_batches
                    ) / len(acceptance_batches)
            optimizer.step(acceptance_closure=acceptance_closure)
            completed_steps += 1
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            timed_out = elapsed_seconds >= maximum_seconds
            evaluate = completed_steps % evaluation_interval_steps == 0 or timed_out
            if evaluate:
                metric = _evaluate(task, model, validation_loader, device)
                write_metric(
                    paths.metric,
                    {
                        "step": completed_steps,
                        "metric": metric,
                        "elapsed_seconds": elapsed_seconds,
                        **optimizer.get_metrics(),
                    },
                )
                _checkpoint(
                    paths.checkpoint,
                    model,
                    optimizer,
                    next_epoch=epoch,
                    next_batch=batch_index + 1,
                    completed_steps=completed_steps,
                    elapsed_seconds=elapsed_seconds,
                )
                write_gn_comparison_plots(root)
                print(
                    f"task={task.identifier} optimizer={optimizer_name} "
                    f"step={completed_steps} metric={metric:.6f}",
                    flush=True,
                )
            if timed_out:
                result = {
                    "task": task.identifier,
                    "domain": task.domain,
                    "model": task.model,
                    "optimizer": optimizer_name,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
        metric = _evaluate(task, model, validation_loader, device)
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        write_metric(
            paths.metric,
            {
                "step": completed_steps,
                "metric": metric,
                "elapsed_seconds": elapsed_seconds,
                **optimizer.get_metrics(),
            },
        )
        _checkpoint(
            paths.checkpoint,
            model,
            optimizer,
            next_epoch=next_epoch,
            next_batch=0,
            completed_steps=completed_steps,
            elapsed_seconds=elapsed_seconds,
        )
        write_gn_comparison_plots(root)
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": optimizer_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
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
