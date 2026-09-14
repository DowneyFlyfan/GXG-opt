"""Metric paths and rendering for matched Qwen3 validation perplexity trials."""

from __future__ import annotations

import json
import hashlib
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plot
import torch
import torch.nn.functional as functional

from qwen3_data import load_qwen_token_cache, qwen_block_loaders
from qwen3_model import build_qwen_optimizers, load_qwen3_model


BASELINE_DISPLAY_NAMES = {"adamw": "AdamW", "muon": "Muon", "muown": "Muown"}
TRIAL_DISPLAY_NAMES = {
    **BASELINE_DISPLAY_NAMES,
    "proposal_notch_v1": "Proposal-notch Muon",
    "routing_resistance_v1": "Routing-resistance Muon",
    "tied_path_curvature_v1": "Tied-path curvature",
}


@dataclass(frozen=True)
class QwenTrialPaths:
    metric: Path
    result: Path
    checkpoint: Path


@dataclass(frozen=True)
class QwenTrialConfig:
    root: Path
    optimizer: str
    run_label: str
    learning_rate: float | None = None
    direction_lr: float | None = None
    gain_lr: float | None = None
    auxiliary_lr: float = 3.0e-4
    weight_decay: float = 0.1
    routing_rho: float = 1.0
    routing_interval: int = 8
    routing_query_rows: int = 4
    routing_edges_per_row: int = 4
    routing_mixture: float = 0.05
    tied_rho: float = 1.0
    tied_probes: int = 2
    tied_interval: int = 16
    tied_max_age: int = 16
    micro_batch_size: int = 1
    gradient_accumulation: int = 1
    maximum_epochs: int = 3
    maximum_updates: int | None = None
    validation_batches: int = 1
    evaluation_interval_updates: int = 1_000
    workers: int = 0
    seed: int = 1337
    device: str = "cuda"


def qwen_trial_paths(root: Path, optimizer: str, run_label: str) -> QwenTrialPaths:
    """Keep durable checkpoints in project cache and lightweight evidence outside it."""
    if optimizer not in TRIAL_DISPLAY_NAMES:
        raise ValueError(f"unsupported Qwen trial optimizer: {optimizer}")
    stem = f"qwen3_0p6b__{run_label}__{optimizer}"
    return QwenTrialPaths(
        metric=root / "metrics" / "nlp" / f"{stem}.ppl.jsonl",
        result=root / "results" / "nlp" / f"{stem}.ppl.json",
        checkpoint=root / ".cache" / "qwen3_0p6b" / "checkpoints" / f"{stem}.checkpoint.pt",
    )


def _metric_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing metric trace: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f"metric trace is empty: {path}")
    required = {"step", "elapsed_seconds", "perplexity"}
    if any(not required <= record.keys() for record in records):
        raise ValueError(f"metric trace lacks perplexity/step/time fields: {path}")
    return records


def render_qwen_comparison(root: Path, *, run_label: str) -> tuple[Path, Path]:
    """Render the required baseline perplexity curves versus steps and time."""
    traces = {
        BASELINE_DISPLAY_NAMES[optimizer]: _metric_records(qwen_trial_paths(root, optimizer, run_label).metric)
        for optimizer in BASELINE_DISPLAY_NAMES
    }
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"qwen3_0p6b_{run_label}_metric_steps.png",
        output_root / f"qwen3_0p6b_{run_label}_metric_time.png",
    )
    for output, x_key, x_label, scale in (
        (outputs[0], "step", "Completed optimizer step", 1.0),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)", 3600.0),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for label, records in traces.items():
            axis.plot(
                [float(record[x_key]) / scale for record in records],
                [float(record["perplexity"]) for record in records],
                marker="o",
                label=label,
            )
        axis.set(
            xlabel=x_label,
            ylabel="Validation perplexity (lower is better)",
            title="Qwen3-0.6B matched optimizer baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def _logits(output: object) -> torch.Tensor:
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("causal language model did not return logits")
    return logits


def _manifest_sha256(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@torch.no_grad()
def _validation_perplexity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    maximum_batches: int,
) -> float:
    if maximum_batches <= 0:
        raise ValueError("validation_batches must be positive")
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        with autocast:
            logits = _logits(model(input_ids=input_ids.to(device), use_cache=False))
        loss = functional.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.to(device).reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += labels.numel()
    if total_tokens == 0:
        raise RuntimeError("validation loader emitted no tokens")
    return math.exp(total_loss / total_tokens)


def run_qwen_trial(config: QwenTrialConfig) -> dict:
    """Run one matched baseline and bind its checkpoint to the token manifest."""
    if config.optimizer not in TRIAL_DISPLAY_NAMES:
        raise ValueError(f"unsupported Qwen trial optimizer: {config.optimizer}")
    if config.micro_batch_size <= 0 or config.gradient_accumulation <= 0:
        raise ValueError("batch and accumulation values must be positive")
    if (
        config.maximum_epochs <= 0
        or config.validation_batches <= 0
        or config.evaluation_interval_updates <= 0
    ):
        raise ValueError("epoch, validation, and evaluation interval values must be positive")
    if config.maximum_updates is not None and config.maximum_updates <= 0:
        raise ValueError("maximum_updates must be positive when provided")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)
    cache = load_qwen_token_cache(config.root)
    manifest_digest = _manifest_sha256(cache.manifest)
    paths = qwen_trial_paths(config.root, config.optimizer, config.run_label)
    if any(path.exists() for path in (paths.metric, paths.result, paths.checkpoint)):
        raise FileExistsError(f"refusing to overwrite Qwen trial {config.run_label}/{config.optimizer}")
    train_loader, validation_loader = qwen_block_loaders(
        cache,
        micro_batch_size=config.micro_batch_size,
        workers=config.workers,
        seed=config.seed,
    )
    model = load_qwen3_model(config.root).to(device)
    optimizers = build_qwen_optimizers(
        model,
        config.optimizer,
        learning_rate=config.learning_rate,
        direction_lr=config.direction_lr,
        gain_lr=config.gain_lr,
        auxiliary_lr=config.auxiliary_lr,
        weight_decay=config.weight_decay,
        routing_rho=config.routing_rho,
        routing_interval=config.routing_interval,
        routing_query_rows=config.routing_query_rows,
        routing_edges_per_row=config.routing_edges_per_row,
        routing_mixture=config.routing_mixture,
        tied_rho=config.tied_rho,
        tied_probes=config.tied_probes,
        tied_interval=config.tied_interval,
        tied_max_age=config.tied_max_age,
    )
    paths.metric.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed_updates = 0
    completed_epochs = 0
    stopped_early = False
    final_perplexity: float | None = None
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        for batch_index, (input_ids, labels) in enumerate(train_loader):
            if batch_index % config.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            for optimizer in optimizers.values():
                prepare_forward = getattr(optimizer, "prepare_forward", None)
                if prepare_forward is not None:
                    prepare_forward()
                prepare_batch = getattr(optimizer, "prepare_batch", None)
                if prepare_batch is not None:
                    prepare_batch(input_ids)
            with autocast:
                logits = _logits(model(input_ids=input_ids.to(device), use_cache=False))
                loss = functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)), labels.to(device).reshape(-1)
                )
            (loss / config.gradient_accumulation).backward()
            if (batch_index + 1) % config.gradient_accumulation:
                continue
            for optimizer in optimizers.values():
                optimizer.step()
            completed_updates += 1
            if completed_updates % config.evaluation_interval_updates == 0:
                elapsed_seconds = time.perf_counter() - started
                final_perplexity = _validation_perplexity(
                    model, validation_loader, device, config.validation_batches
                )
                peak_memory_mib = (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else None
                )
                record = {
                    "epoch": epoch,
                    "step": completed_updates,
                    "elapsed_seconds": elapsed_seconds,
                    "token_exposure": (
                        completed_updates
                        * config.micro_batch_size
                        * config.gradient_accumulation
                        * cache.sequence_length
                    ),
                    "perplexity": final_perplexity,
                    "data_manifest_sha256": manifest_digest,
                    "peak_memory_mib": peak_memory_mib,
                }
                with paths.metric.open("a") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                model.train()
            if config.maximum_updates is not None and completed_updates >= config.maximum_updates:
                stopped_early = True
                break
        if stopped_early:
            break
        completed_epochs = epoch
    elapsed_seconds = time.perf_counter() - started
    if final_perplexity is None or completed_updates % config.evaluation_interval_updates:
        final_perplexity = _validation_perplexity(
            model, validation_loader, device, config.validation_batches
        )
        record = {
            "epoch": completed_epochs,
            "step": completed_updates,
            "elapsed_seconds": time.perf_counter() - started,
            "token_exposure": (
                completed_updates
                * config.micro_batch_size
                * config.gradient_accumulation
                * cache.sequence_length
            ),
            "perplexity": final_perplexity,
            "data_manifest_sha256": manifest_digest,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            ),
        }
        with paths.metric.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    token_exposure = completed_updates * config.micro_batch_size * config.gradient_accumulation * cache.sequence_length
    checkpoint = {
        "model": model.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "completed_epochs": completed_epochs,
        "completed_updates": completed_updates,
        "data_manifest_sha256": manifest_digest,
        "peak_memory_mib": peak_memory_mib,
        "config": asdict(config),
    }
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, paths.checkpoint)
    result = {
        "optimizer": config.optimizer,
        "run_label": config.run_label,
        "completed_epochs": completed_epochs,
        "completed_updates": completed_updates,
        "final_perplexity": final_perplexity,
        "elapsed_seconds": elapsed_seconds,
        "data_manifest_sha256": manifest_digest,
        "peak_memory_mib": peak_memory_mib,
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
