"""Resumable GPT2-12x512 screens for the spectral-unit-ball full GGN method."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional

from artifacts import write_metric
from gpt2_ppl_experiment import perplexity_from_nll, validation_nll
from gn_experiment import language_model_task
from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch, GGNFullOperator
from models import parameter_count
from optimizers import muon_parameter_names
from spectral_unit_ball_gn import SpectralUnitBallConfig, spectral_unit_ball_step
from training import _loaders, _model, configure_reproducibility


def _paths(root: Path, label: str) -> tuple[Path, Path, Path]:
    stem = f"nlp_gpt_12x512__{label}__spectral_unit_ball_full_ggn"
    return (
        root / "metrics" / "nlp" / f"{stem}.ppl.jsonl",
        root / "results" / "nlp" / f"{stem}.ppl.json",
        root / ".cache" / "nlp" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def _loss(model: torch.nn.Module, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = model(token_ids)
    logits = output.logits if hasattr(output, "logits") else output
    return functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))


def _batch(token_ids: torch.Tensor, targets: torch.Tensor) -> FunctionalCurvatureBatch:
    def loss_fn(output, selected_targets=targets):
        logits = output.logits if hasattr(output, "logits") else output
        return functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), selected_targets.reshape(-1)
        )

    def output_hvp_fn(output, tangent):
        logits = output.logits if hasattr(output, "logits") else output
        tangent_logits = tangent.logits if hasattr(tangent, "logits") else tangent
        output_shape = logits.shape
        logits = logits.reshape(-1, logits.size(-1)).float()
        tangent_logits = tangent_logits.reshape_as(logits).float()
        probabilities = logits.softmax(dim=-1)
        return (
            probabilities
            * (
                tangent_logits
                - (probabilities * tangent_logits).sum(dim=-1, keepdim=True)
            )
            / logits.size(0)
        ).reshape(output_shape)

    return FunctionalCurvatureBatch(
        args=(token_ids,), loss_fn=loss_fn, output_hvp_fn=output_hvp_fn
    )


def _save_checkpoint(
    path: Path, model, auxiliary_optimizer, *, step: int, elapsed_seconds: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "auxiliary_optimizer": (
                None if auxiliary_optimizer is None else auxiliary_optimizer.state_dict()
            ),
            "step": step,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def run_spectral_unit_ball_screen(
    root: Path,
    *,
    label: str,
    maximum_steps: int,
    inner_iterations: int = 1,
    initial_beta: float = 1.0,
    micro_batch_size: int = 1,
    validation_batches: int = 4,
    auxiliary_learning_rate: float = 0.0,
    auxiliary_weight_decay: float = 0.0,
    workers: int = 2,
    seed: int = 1337,
    fresh: bool = False,
) -> dict:
    """Screen the exact restricted full GGN method; this is not a final run."""
    if maximum_steps <= 0 or micro_batch_size <= 0 or validation_batches <= 0:
        raise ValueError("step, batch, and validation counts must be positive")
    if auxiliary_learning_rate < 0 or auxiliary_weight_decay < 0:
        raise ValueError("auxiliary AdamW learning rate and weight decay must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a full-GGN screen")
    configure_reproducibility(seed)
    task = language_model_task(micro_batch_size=micro_batch_size, gradient_accumulation=1)
    metric_path, result_path, checkpoint_path = _paths(root, label)
    if fresh:
        metric_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
        checkpoint_path.unlink(missing_ok=True)
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    device = torch.device("cuda")
    model = _model(task).to(device)
    completed_steps = 0
    elapsed_seconds = 0.0
    checkpoint_payload = None
    if checkpoint_path.exists():
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint_payload["model"])
        completed_steps, elapsed_seconds = int(checkpoint_payload["step"]), float(checkpoint_payload["elapsed_seconds"])
    selected_names = tuple(sorted(muon_parameter_names(model)))
    selected_parameters = dict(model.named_parameters())
    if not selected_names or any(selected_parameters[name].ndim != 2 for name in selected_names):
        raise RuntimeError("spectral GGN routing must select only matrix parameters")
    config = SpectralUnitBallConfig(
        inner_iterations=inner_iterations, initial_beta=initial_beta
    )
    selected_ids = {id(selected_parameters[name]) for name in selected_names}
    auxiliary_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in selected_ids
    ]
    auxiliary_optimizer = (
        None
        if auxiliary_learning_rate == 0.0
        else torch.optim.AdamW(
            auxiliary_parameters,
            lr=auxiliary_learning_rate,
            weight_decay=auxiliary_weight_decay,
            betas=(0.9, 0.95),
        )
    )
    if checkpoint_payload is not None and auxiliary_optimizer is not None:
        optimizer_state = checkpoint_payload.get("auxiliary_optimizer")
        if optimizer_state is None:
            raise RuntimeError("checkpoint has no AdamW state for the requested auxiliary path")
        auxiliary_optimizer.load_state_dict(optimizer_state)
    iterator = iter(train_loader)
    for _ in range(completed_steps):
        next(iterator)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_ppl = float("nan")
    last_step = None
    while completed_steps < maximum_steps:
        model.train()
        token_ids, targets = (value.to(device, non_blocking=True) for value in next(iterator))
        if auxiliary_optimizer is not None:
            auxiliary_optimizer.zero_grad(set_to_none=True)
            _loss(model, token_ids, targets).backward()
        operator = GGNFullOperator(
            model, _batch(token_ids, targets), parameter_names=selected_names
        )
        last_step = spectral_unit_ball_step(
            operator,
            lambda: _loss(model, token_ids, targets),
            config=config,
        )
        if auxiliary_optimizer is not None:
            auxiliary_optimizer.step()
        completed_steps += 1
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        nll = validation_nll(model, validation_loader, device, maximum_batches=validation_batches)
        final_ppl = perplexity_from_nll(nll)
        record = {
            "step": completed_steps,
            "elapsed_seconds": elapsed_seconds,
            "validation_nll": nll,
            "perplexity": final_ppl,
            "selected_matrix_blocks": len(selected_names),
            "selected_parameters": sum(selected_parameters[name].numel() for name in selected_names),
            "auxiliary_learning_rate": auxiliary_learning_rate,
            **asdict(last_step),
        }
        write_metric(metric_path, record)
        _save_checkpoint(
            checkpoint_path,
            model,
            auxiliary_optimizer,
            step=completed_steps,
            elapsed_seconds=elapsed_seconds,
        )
        print(json.dumps({"optimizer": "spectral_unit_ball_full_ggn", **record}, sort_keys=True), flush=True)
        started = time.perf_counter()
    result = {
        "task": task.identifier,
        "model": task.model,
        "optimizer": "spectral_unit_ball_full_ggn",
        "status": "screen_completed",
        "parameters": parameter_count(model),
        "selected_matrix_blocks": len(selected_names),
        "selected_parameters": sum(selected_parameters[name].numel() for name in selected_names),
        "auxiliary_parameters": sum(parameter.numel() for parameter in auxiliary_parameters),
        "micro_batch_size": micro_batch_size,
        "maximum_steps": maximum_steps,
        "inner_iterations": inner_iterations,
        "initial_beta": initial_beta,
        "validation_batches": validation_batches,
        "auxiliary_learning_rate": auxiliary_learning_rate,
        "auxiliary_weight_decay": auxiliary_weight_decay,
        "final_perplexity": final_ppl,
        "seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "last_step": {} if last_step is None else asdict(last_step),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
