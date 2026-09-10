"""Controlled generalized Gauss--Newton-guided AdamW language-model trials."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK, artifact_paths, write_gn_comparison_plots
from gn_guided_adam import FunctionalBatch, GNGuidedAdamW, GuidedStepContext
from gn_guided_adam.config import AdamWConfig, GNConfig, GuidedAdamConfig
from models import CONTEXT_LENGTH
from training import _evaluate, _loaders, _model, configure_reproducibility


MINIMUM_EFFECTIVE_BATCH_SIZE = 60
DEFAULT_GUIDED_BLOCK_PATTERNS = (r"^blocks\.11\.feedforward_out\.weight$",)


def validate_guided_gn_contract(
    *,
    micro_batch_size: int,
    sequence_length: int,
    gradient_accumulation: int,
    curvature_accumulation: int,
) -> dict[str, int]:
    """Keep model, context, and statistical batch fixed across tuning runs."""
    for name, value in {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation": gradient_accumulation,
        "curvature_accumulation": curvature_accumulation,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if sequence_length != CONTEXT_LENGTH:
        raise ValueError(
            f"sequence_length must remain {CONTEXT_LENGTH} for the retained baseline"
        )
    if curvature_accumulation > gradient_accumulation:
        raise ValueError("curvature accumulation cannot exceed gradient accumulation")
    gradient_effective_batch_size = micro_batch_size * gradient_accumulation
    curvature_effective_batch_size = micro_batch_size * curvature_accumulation
    if gradient_effective_batch_size < MINIMUM_EFFECTIVE_BATCH_SIZE:
        raise ValueError("gradient effective batch must be at least 60")
    if curvature_effective_batch_size < MINIMUM_EFFECTIVE_BATCH_SIZE:
        raise ValueError("curvature effective batch must be at least 60")
    return {
        "sequence_length": sequence_length,
        "gradient_effective_batch_size": gradient_effective_batch_size,
        "curvature_effective_batch_size": curvature_effective_batch_size,
    }


def guided_gn_config(
    *,
    learning_rate: float,
    weight_decay: float,
    curvature_accumulation: int,
    guided_block_patterns: tuple[str, ...] = DEFAULT_GUIDED_BLOCK_PATTERNS,
    rank: int = 2,
    refresh_interval: int = 200,
    initial_damping: float = 1.0e-2,
    warmup_steps: int = 0,
    trust_radius: float = 1.0,
    max_relative_block_update: float = 1.0e-3,
    alpha_max: float = 1.0,
    max_basis_age: int | None = None,
    max_parameter_drift: float = 1.0e-2,
    rho_min: float = 0.0,
    acceptance_margin: float = 0.0,
) -> GuidedAdamConfig:
    if not guided_block_patterns:
        raise ValueError("At least one guided Transformer matrix must be selected")
    return GuidedAdamConfig(
        adamw=AdamWConfig(
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=weight_decay,
        ),
        gn=GNConfig(
            warmup_steps=warmup_steps,
            rank=rank,
            refresh_interval=refresh_interval,
            curvature_batches=curvature_accumulation,
            acceptance_batches=1,
            initial_damping=initial_damping,
            min_damping=min(initial_damping, 1.0e-6),
            trust_radius=trust_radius,
            max_relative_block_update=max_relative_block_update,
            alpha_max=alpha_max,
            max_basis_age=(refresh_interval if max_basis_age is None else max_basis_age),
            max_parameter_drift=max_parameter_drift,
            rho_min=rho_min,
            acceptance_margin=acceptance_margin,
            min_block_numel=256,
            include_output_projection=False,
            guided_block_patterns=guided_block_patterns,
        ),
    )


def _model_logits(output: object) -> torch.Tensor:
    return output.logits if hasattr(output, "logits") else output  # type: ignore[return-value]


def functional_curvature_batches(
    batches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    *,
    completed_steps: int,
) -> tuple[FunctionalBatch, ...]:
    if not batches:
        raise ValueError("At least one curvature microbatch is required")
    result = []
    for microbatch_index, (token_ids, targets) in enumerate(batches):
        result.append(
            FunctionalBatch(
                args=(token_ids,),
                loss_fn=lambda output, selected_targets=targets: functional.cross_entropy(
                    _model_logits(output).float().reshape(-1, _model_logits(output).size(-1)),
                    selected_targets.reshape(-1),
                ),
                batch_id=f"step-{completed_steps}-micro-{microbatch_index}",
            )
        )
    return tuple(result)


def _training_loss(model, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = _model_logits(model(token_ids))
        return functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )


def _save_checkpoint(
    path: Path,
    model,
    optimizer: GNGuidedAdamW,
    *,
    next_epoch: int,
    next_batch: int,
    completed_steps: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "completed_steps": completed_steps,
            "elapsed_seconds": elapsed_seconds,
        },
        temporary,
    )
    temporary.replace(path)


def _restore_checkpoint(path: Path, model, optimizer: GNGuidedAdamW) -> tuple[int, int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["next_epoch"]),
        int(payload["next_batch"]),
        int(payload["completed_steps"]),
        float(payload["elapsed_seconds"]),
    )


def _set_epoch_learning_rate(
    optimizer: GNGuidedAdamW, *, initial_learning_rate: float, epoch: int, epochs: int
) -> float:
    learning_rate = initial_learning_rate * 0.5 * (
        1.0 + math.cos(math.pi * (epoch - 1) / epochs)
    )
    optimizer.param_groups[0]["lr"] = learning_rate
    return learning_rate


def _best_baseline_metric(root: Path) -> float:
    best = 0.0
    for name in ("adamw", "muon"):
        path = artifact_paths(root, name).metric
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                best = max(best, float(json.loads(line)["metric"]))
    return best


def run_guided_gn_trial(
    root: Path,
    *,
    micro_batch_size: int,
    gradient_accumulation: int,
    curvature_accumulation: int,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 0.01,
    guided_block_patterns: tuple[str, ...] = DEFAULT_GUIDED_BLOCK_PATTERNS,
    rank: int = 2,
    refresh_interval: int = 200,
    initial_damping: float = 1.0e-2,
    warmup_steps: int = 0,
    trust_radius: float = 1.0,
    max_relative_block_update: float = 1.0e-3,
    alpha_max: float = 1.0,
    max_basis_age: int | None = None,
    max_parameter_drift: float = 1.0e-2,
    rho_min: float = 0.0,
    acceptance_margin: float = 0.0,
    maximum_seconds: float = 14_400.0,
    maximum_steps: int | None = None,
    evaluation_interval_steps: int = 128,
    workers: int = 4,
    seed: int = 1337,
    fresh: bool = False,
    label: str = "gn_guided_adamw",
) -> dict:
    contract = validate_guided_gn_contract(
        micro_batch_size=micro_batch_size,
        sequence_length=CONTEXT_LENGTH,
        gradient_accumulation=gradient_accumulation,
        curvature_accumulation=curvature_accumulation,
    )
    if maximum_seconds <= 0 or maximum_seconds > 14_400:
        raise ValueError("maximum_seconds must be in (0, 14400]")
    if maximum_steps is not None and (
        not isinstance(maximum_steps, int)
        or isinstance(maximum_steps, bool)
        or maximum_steps <= 0
    ):
        raise ValueError("maximum_steps must be a positive integer when provided")
    if evaluation_interval_steps <= 0:
        raise ValueError("evaluation_interval_steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the guided-GGN language-model experiment")

    task = replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=micro_batch_size,
        gradient_accumulation=gradient_accumulation,
        adamw_lr=learning_rate,
        weight_decay=weight_decay,
    )
    configure_reproducibility(seed)
    device = torch.device("cuda")
    paths = artifact_paths(root, "gn_guided_adamw", run_label=label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    config = guided_gn_config(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        curvature_accumulation=curvature_accumulation,
        guided_block_patterns=guided_block_patterns,
        rank=rank,
        refresh_interval=refresh_interval,
        initial_damping=initial_damping,
        warmup_steps=warmup_steps,
        trust_radius=trust_radius,
        max_relative_block_update=max_relative_block_update,
        alpha_max=alpha_max,
        max_basis_age=max_basis_age,
        max_parameter_drift=max_parameter_drift,
        rho_min=rho_min,
        acceptance_margin=acceptance_margin,
    )
    optimizer = GNGuidedAdamW(model, config)
    next_epoch, resume_batch, completed_steps, elapsed_base = (1, 0, 0, 0.0)
    if paths.checkpoint.exists():
        next_epoch, resume_batch, completed_steps, elapsed_base = _restore_checkpoint(
            paths.checkpoint, model, optimizer
        )
    elif paths.metric.exists():
        paths.metric.unlink()

    torch.cuda.reset_peak_memory_stats(device)
    session_started = time.perf_counter()
    last_metric: float | None = None
    last_recorded_step = -1
    for epoch in range(next_epoch, task.estimated_epochs + 1):
        learning_rate_now = _set_epoch_learning_rate(
            optimizer,
            initial_learning_rate=learning_rate,
            epoch=epoch,
            epochs=task.estimated_epochs,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for batch_index, cpu_batch in enumerate(train_loader):
            if epoch == next_epoch and batch_index < resume_batch:
                continue
            device_batch = tuple(
                value.to(device, non_blocking=True) for value in cpu_batch
            )
            step_batches.append(device_batch)
            token_ids, targets = device_batch
            (_training_loss(model, token_ids, targets) / gradient_accumulation).backward()
            if len(step_batches) < gradient_accumulation:
                continue

            curvature_batches = functional_curvature_batches(
                tuple(step_batches[:curvature_accumulation]),
                completed_steps=completed_steps,
            )
            context = GuidedStepContext(
                curvature_batch=curvature_batches,
                acceptance_batch=curvature_batches[0],
                curvature_reuses_training_data=True,
                acceptance_reuses_other_data=False,
                gradient_accumulation=gradient_accumulation,
                tokens=micro_batch_size * gradient_accumulation * CONTEXT_LENGTH,
                epoch=epoch - 1,
            )
            step_result = optimizer.step(context)
            optimizer.zero_grad(set_to_none=True)
            step_batches.clear()
            completed_steps += 1
            elapsed_seconds = elapsed_base + time.perf_counter() - session_started
            timed_out = elapsed_seconds >= maximum_seconds
            step_limited = maximum_steps is not None and completed_steps >= maximum_steps
            stopped = timed_out or step_limited
            evaluate = completed_steps % evaluation_interval_steps == 0 or stopped
            if evaluate:
                last_metric = _evaluate(task, model, validation_loader, device)
                elapsed_seconds = elapsed_base + time.perf_counter() - session_started
                timed_out = timed_out or elapsed_seconds >= maximum_seconds
                stopped = timed_out or step_limited
                metrics = optimizer.get_metrics()
                write_metric(
                    paths.metric,
                    {
                        "step": completed_steps,
                        "metric": last_metric,
                        "elapsed_seconds": elapsed_seconds,
                        "epoch": epoch,
                        "learning_rate": learning_rate_now,
                        "guidance_accepted": float(step_result.guidance_accepted),
                        **metrics,
                    },
                )
                last_recorded_step = completed_steps
                _save_checkpoint(
                    paths.checkpoint,
                    model,
                    optimizer,
                    next_epoch=epoch,
                    next_batch=batch_index + 1,
                    completed_steps=completed_steps,
                    elapsed_seconds=elapsed_seconds,
                )
                write_gn_comparison_plots(root, guided_run_label=label)
                print(
                    f"task={task.identifier} optimizer=gn_guided_adamw "
                    f"step={completed_steps} metric={last_metric:.6f} "
                    f"accepted={int(step_result.guidance_accepted)}",
                    flush=True,
                )
            if stopped:
                return _write_result(
                    paths.result,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    contract=contract,
                    completed_steps=completed_steps,
                    elapsed_seconds=elapsed_base + time.perf_counter() - session_started,
                    peak_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
                    peak_reserved_memory_mb=torch.cuda.max_memory_reserved(device) / 1024**2,
                    last_metric=last_metric,
                    baseline_metric=_best_baseline_metric(root),
                    status=(
                        "time_limit_checkpointed"
                        if timed_out
                        else "step_limit_checkpointed"
                    ),
                    label=label,
                )
        resume_batch = 0
        next_epoch = epoch + 1
        if completed_steps != last_recorded_step:
            last_metric = _evaluate(task, model, validation_loader, device)
            elapsed_seconds = elapsed_base + time.perf_counter() - session_started
            write_metric(
                paths.metric,
                {
                    "step": completed_steps,
                    "metric": last_metric,
                    "elapsed_seconds": elapsed_seconds,
                    "epoch": epoch,
                    "learning_rate": learning_rate_now,
                    **optimizer.get_metrics(),
                },
            )
            last_recorded_step = completed_steps
            _save_checkpoint(
                paths.checkpoint,
                model,
                optimizer,
                next_epoch=next_epoch,
                next_batch=0,
                completed_steps=completed_steps,
                elapsed_seconds=elapsed_seconds,
            )
            write_gn_comparison_plots(root, guided_run_label=label)

    elapsed_seconds = elapsed_base + time.perf_counter() - session_started
    result = _write_result(
        paths.result,
        model=model,
        optimizer=optimizer,
        config=config,
        contract=contract,
        completed_steps=completed_steps,
        elapsed_seconds=elapsed_seconds,
        peak_memory_mb=torch.cuda.max_memory_allocated(device) / 1024**2,
        peak_reserved_memory_mb=torch.cuda.max_memory_reserved(device) / 1024**2,
        last_metric=last_metric,
        baseline_metric=_best_baseline_metric(root),
        status="completed",
        label=label,
    )
    paths.checkpoint.unlink(missing_ok=True)
    return result


def _write_result(
    path: Path,
    *,
    model,
    optimizer: GNGuidedAdamW,
    config: GuidedAdamConfig,
    contract: dict[str, int],
    completed_steps: int,
    elapsed_seconds: float,
    peak_memory_mb: float,
    peak_reserved_memory_mb: float,
    last_metric: float | None,
    baseline_metric: float,
    status: str,
    label: str,
) -> dict:
    result = {
        "task": LANGUAGE_MODEL_GN_TASK.identifier,
        "domain": LANGUAGE_MODEL_GN_TASK.domain,
        "model": LANGUAGE_MODEL_GN_TASK.model,
        "optimizer": "gn_guided_adamw",
        "label": label,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        **contract,
        "completed_steps": completed_steps,
        "seconds": elapsed_seconds,
        "peak_memory_mb": peak_memory_mb,
        "peak_reserved_memory_mb": peak_reserved_memory_mb,
        "last_metric": last_metric,
        "baseline_best_metric": baseline_metric,
        "target_beaten": bool(last_metric is not None and last_metric > baseline_metric),
        "guided_blocks": [spec.name for spec in optimizer.registry.enabled],
        "optimizer_config": config.to_dict(),
        "status": status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
