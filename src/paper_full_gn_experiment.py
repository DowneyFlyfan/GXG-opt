"""Scaled reproduction of the paper's Full Gauss-Newton language-model run."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as functional

from artifacts import write_metric
from gn_experiment import LANGUAGE_MODEL_GN_TASK, artifact_paths, write_gn_comparison_plots
from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch
from models import CONTEXT_LENGTH, parameter_count
from paper_full_gn import (
    build_paper_inner_optimizers,
    held_out_line_search,
    paper_accumulated_inner_step,
)
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


MINIMUM_OUTER_EFFECTIVE_BATCH_SIZE = 60
COMMON_WARMUP_NAME = "nlp_gpt_12x512__paper_common_adamw_warmup.pt"


def paper_gn_time_limit_reached(
    elapsed_seconds: float, maximum_seconds: float
) -> bool:
    """Return whether a completed operation has exhausted the run budget."""
    return elapsed_seconds >= maximum_seconds


def validate_paper_gn_contract(
    *,
    micro_batch_size: int,
    sequence_length: int,
    inner_steps: int,
    inner_gradient_accumulation: int,
) -> dict[str, int]:
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if sequence_length != CONTEXT_LENGTH:
        raise ValueError(f"sequence_length must remain {CONTEXT_LENGTH}")
    if inner_steps <= 0:
        raise ValueError("inner_steps must be positive")
    if inner_gradient_accumulation <= 0:
        raise ValueError("inner_gradient_accumulation must be positive")
    inner_effective_batch_size = micro_batch_size * inner_gradient_accumulation
    outer_effective_batch_size = inner_effective_batch_size * inner_steps
    if outer_effective_batch_size < MINIMUM_OUTER_EFFECTIVE_BATCH_SIZE:
        raise ValueError("outer effective batch must be at least 60")
    return {
        "sequence_length": sequence_length,
        "inner_effective_batch_size": inner_effective_batch_size,
        "outer_effective_batch_size": outer_effective_batch_size,
        "outer_effective_tokens": outer_effective_batch_size * sequence_length,
        "line_search_effective_batch_size": outer_effective_batch_size,
    }


def chinchilla_warmup_tokens(
    parameter_count: int,
    *,
    tokens_per_parameter: float = 20.0,
    warmup_fraction: float = 0.05,
) -> int:
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if tokens_per_parameter <= 0 or not 0 < warmup_fraction <= 1:
        raise ValueError("Chinchilla scale and warmup fraction must be positive")
    return round(parameter_count * tokens_per_parameter * warmup_fraction)


def language_model_curvature_batch(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> FunctionalCurvatureBatch:
    token_ids, targets = (
        value.to(device, non_blocking=device.type == "cuda") for value in batch
    )

    def loss_fn(output, selected_targets=targets):
        logits = output.logits if hasattr(output, "logits") else output
        return functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            selected_targets.reshape(-1),
        )

    return FunctionalCurvatureBatch(args=(token_ids,), loss_fn=loss_fn)


def save_paper_gn_checkpoint(
    path: Path,
    outer_model: torch.nn.Module,
    inner_model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    completed_outer_steps: int,
    consumed_training_batches: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "outer_model": outer_model.state_dict(),
            "inner_model": inner_model.state_dict(),
            "optimizers": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "completed_outer_steps": completed_outer_steps,
            "consumed_training_batches": consumed_training_batches,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def load_paper_gn_checkpoint(
    path: Path,
    outer_model: torch.nn.Module,
    inner_model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    outer_model.load_state_dict(payload["outer_model"])
    inner_model.load_state_dict(payload["inner_model"])
    if optimizers.keys() != payload["optimizers"].keys():
        raise ValueError("Checkpoint optimizer names do not match the current run")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
    return (
        int(payload["completed_outer_steps"]),
        int(payload["consumed_training_batches"]),
        float(payload["elapsed_seconds"]),
    )


def common_warmup_path(root: Path) -> Path:
    return root / ".cache" / "nlp" / "checkpoints" / COMMON_WARMUP_NAME


def _save_warmup_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    completed_optimizer_steps: int,
    consumed_batches: int,
    processed_tokens: int,
    target_tokens: int,
    elapsed_seconds: float,
    metric: float | None,
    complete: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_optimizer_steps": completed_optimizer_steps,
            "consumed_batches": consumed_batches,
            "processed_tokens": processed_tokens,
            "target_tokens": target_tokens,
            "elapsed_seconds": elapsed_seconds,
            "metric": metric,
            "complete": complete,
        },
        path,
    )


def _infinite_batches(loader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def prepare_common_adamw_warmup(
    root: Path,
    *,
    maximum_seconds: float = 14_400.0,
    workers: int = 4,
    seed: int = 1337,
    checkpoint_interval_steps: int = 512,
    fresh: bool = False,
) -> dict:
    """Create the paper's shared 5%-Chinchilla AdamW starting checkpoint."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper reproduction")
    configure_reproducibility(seed)
    device = torch.device("cuda")
    task = LANGUAGE_MODEL_GN_TASK
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=task.adamw_lr,
        weight_decay=task.weight_decay,
        betas=(0.9, 0.95),
    )
    target_tokens = chinchilla_warmup_tokens(parameter_count(model))
    path = common_warmup_path(root)
    if fresh:
        path.unlink(missing_ok=True)
    completed_optimizer_steps = 0
    consumed_batches = 0
    processed_tokens = 0
    elapsed_seconds = 0.0
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if bool(payload["complete"]):
            return {
                "status": "complete",
                "checkpoint": str(path),
                "processed_tokens": int(payload["processed_tokens"]),
                "target_tokens": int(payload["target_tokens"]),
                "metric": float(payload["metric"]),
                "seconds": float(payload["elapsed_seconds"]),
            }
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        completed_optimizer_steps = int(payload["completed_optimizer_steps"])
        consumed_batches = int(payload["consumed_batches"])
        processed_tokens = int(payload["processed_tokens"])
        elapsed_seconds = float(payload["elapsed_seconds"])

    batches = _infinite_batches(train_loader)
    for _ in range(consumed_batches):
        next(batches)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    micro_in_step = consumed_batches % task.gradient_accumulation
    model.train()
    while processed_tokens < target_tokens:
        batch = next(batches)
        consumed_batches += 1
        batch_tokens = int(batch[0].numel())
        processed_tokens += batch_tokens
        (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
        micro_in_step += 1
        if micro_in_step == task.gradient_accumulation:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_in_step = 0
            completed_optimizer_steps += 1
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        timed_out = elapsed_seconds >= maximum_seconds and micro_in_step == 0
        checkpoint_due = (
            completed_optimizer_steps > 0
            and completed_optimizer_steps % checkpoint_interval_steps == 0
            and micro_in_step == 0
        )
        if timed_out or checkpoint_due:
            _save_warmup_checkpoint(
                path,
                model,
                optimizer,
                completed_optimizer_steps=completed_optimizer_steps,
                consumed_batches=consumed_batches,
                processed_tokens=processed_tokens,
                target_tokens=target_tokens,
                elapsed_seconds=elapsed_seconds,
                metric=None,
                complete=False,
            )
        if timed_out:
            return {
                "status": "time_limit_checkpointed",
                "checkpoint": str(path),
                "processed_tokens": processed_tokens,
                "target_tokens": target_tokens,
                "seconds": elapsed_seconds,
            }

    if micro_in_step:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        completed_optimizer_steps += 1
    metric = _evaluate(task, model, validation_loader, device)
    elapsed_seconds += time.perf_counter() - started
    _save_warmup_checkpoint(
        path,
        model,
        optimizer,
        completed_optimizer_steps=completed_optimizer_steps,
        consumed_batches=consumed_batches,
        processed_tokens=processed_tokens,
        target_tokens=target_tokens,
        elapsed_seconds=elapsed_seconds,
        metric=metric,
        complete=True,
    )
    return {
        "status": "complete",
        "checkpoint": str(path),
        "processed_tokens": processed_tokens,
        "target_tokens": target_tokens,
        "metric": metric,
        "seconds": elapsed_seconds,
    }


def _load_complete_warmup(path: Path, model: torch.nn.Module) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing common AdamW warmup checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not bool(payload.get("complete", False)):
        raise RuntimeError("Common AdamW warmup checkpoint is incomplete")
    model.load_state_dict(payload["model"])
    return payload


def run_paper_full_gn_trial(
    root: Path,
    *,
    micro_batch_size: int = 1,
    sequence_length: int = 1024,
    inner_steps: int = 122,
    inner_gradient_accumulation: int = 1,
    inner_learning_rate: float = 0.01,
    optimizer_weight_decay: float = 0.001,
    gradient_clip: float = 1.0,
    line_search_range: int = 5,
    maximum_outer_steps: int = 10_000,
    maximum_seconds: float = 14_400.0,
    evaluation_interval_steps: int = 1,
    workers: int = 4,
    seed: int = 1337,
    label: str = "paper_full_gn",
    fresh: bool = False,
) -> dict:
    """Run the official Muon-inner Full-GN algorithm on the retained task."""
    contract = validate_paper_gn_contract(
        micro_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        inner_steps=inner_steps,
        inner_gradient_accumulation=inner_gradient_accumulation,
    )
    if inner_learning_rate <= 0 or optimizer_weight_decay < 0:
        raise ValueError("Inner learning rate must be positive and decay non-negative")
    if gradient_clip <= 0 or line_search_range <= 0:
        raise ValueError("Gradient clip and line-search range must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper reproduction")

    configure_reproducibility(seed)
    device = torch.device("cuda")
    task = replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=micro_batch_size,
        gradient_accumulation=1,
    )
    train_loader, validation_loader = _loaders(task, root, workers, seed + 17)
    outer_model = _model(task).to(device)
    warmup = _load_complete_warmup(common_warmup_path(root), outer_model)
    inner_model = _model(task).to(device)
    inner_model.load_state_dict(outer_model.state_dict())
    optimizers = build_paper_inner_optimizers(
        inner_model,
        learning_rate=inner_learning_rate,
        weight_decay=optimizer_weight_decay,
    )
    base_learning_rates = {name: inner_learning_rate for name in optimizers}
    paths = artifact_paths(root, "full_ggn", run_label=label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    completed_outer_steps = 0
    consumed_training_batches = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        completed_outer_steps, consumed_training_batches, elapsed_seconds = (
            load_paper_gn_checkpoint(
                paths.checkpoint,
                outer_model,
                inner_model,
                optimizers,
            )
        )
    elif paths.metric.exists():
        paths.metric.unlink()

    batches = _infinite_batches(train_loader)
    for _ in range(consumed_training_batches):
        next(batches)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    while completed_outer_steps < maximum_outer_steps:
        outer_model.train()
        inner_model.train()
        final_inner_gradient_norm = float("nan")
        for inner_step in range(inner_steps):
            accumulated = tuple(
                language_model_curvature_batch(next(batches), device)
                for _ in range(inner_gradient_accumulation)
            )
            consumed_training_batches += inner_gradient_accumulation
            final_inner_gradient_norm = paper_accumulated_inner_step(
                outer_model,
                inner_model,
                accumulated,
                optimizers,
                base_learning_rates,
                inner_step=inner_step,
                inner_steps=inner_steps,
                gradient_clip=gradient_clip,
            )
            del accumulated
            progress_interval = max(inner_steps // 10, 1)
            if (inner_step + 1) % progress_interval == 0:
                print(
                    f"task={task.identifier} optimizer=paper_full_gn "
                    f"outer_step={completed_outer_steps + 1} "
                    f"inner_step={inner_step + 1}/{inner_steps} "
                    f"inner_gradient_norm={final_inner_gradient_norm:.6g}",
                    flush=True,
                )

        held_out_batches = tuple(
            language_model_curvature_batch(next(batches), device)
            for _ in range(inner_steps * inner_gradient_accumulation)
        )
        consumed_training_batches += len(held_out_batches)
        line_search = held_out_line_search(
            outer_model,
            inner_model,
            held_out_batches,
            search_range=line_search_range,
        )
        del held_out_batches
        completed_outer_steps += 1
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        timed_out = paper_gn_time_limit_reached(elapsed_seconds, maximum_seconds)
        should_evaluate = (
            completed_outer_steps % evaluation_interval_steps == 0 or timed_out
        )
        if should_evaluate:
            metric = _evaluate(task, outer_model, validation_loader, device)
            elapsed_seconds += time.perf_counter() - started
            started = time.perf_counter()
            timed_out = paper_gn_time_limit_reached(
                elapsed_seconds, maximum_seconds
            )
            write_metric(
                paths.metric,
                {
                    "step": completed_outer_steps,
                    "metric": metric,
                    "elapsed_seconds": float(warmup["elapsed_seconds"])
                    + elapsed_seconds,
                    "post_warmup_elapsed_seconds": elapsed_seconds,
                    "line_search_step_size": line_search.step_size,
                    "line_search_loss": line_search.loss,
                    "inner_gradient_norm": final_inner_gradient_norm,
                    **contract,
                },
            )
            save_paper_gn_checkpoint(
                paths.checkpoint,
                outer_model,
                inner_model,
                optimizers,
                completed_outer_steps=completed_outer_steps,
                consumed_training_batches=consumed_training_batches,
                elapsed_seconds=elapsed_seconds,
            )
            write_gn_comparison_plots(root, paper_full_gn_run_label=label)
            print(
                f"task={task.identifier} optimizer=paper_full_gn "
                f"step={completed_outer_steps} metric={metric:.6f} "
                f"line_search_step={line_search.step_size:.6g}",
                flush=True,
            )
        if timed_out:
            break

    status = (
        "completed"
        if completed_outer_steps >= maximum_outer_steps
        else "time_limit_checkpointed"
    )
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "paper_full_gn",
        "parameters": parameter_count(outer_model),
        "warmup_tokens": int(warmup["processed_tokens"]),
        "micro_batch_size": micro_batch_size,
        "inner_steps": inner_steps,
        "inner_gradient_accumulation": inner_gradient_accumulation,
        "inner_learning_rate": inner_learning_rate,
        "optimizer_weight_decay": optimizer_weight_decay,
        "gradient_clip": gradient_clip,
        "line_search_range": line_search_range,
        **contract,
        "completed_outer_steps": completed_outer_steps,
        "consumed_training_batches": consumed_training_batches,
        "seconds": float(warmup["elapsed_seconds"]) + elapsed_seconds,
        "post_warmup_seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": status,
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
