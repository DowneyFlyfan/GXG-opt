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
    resume: bool = False


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


def qwen_full_evaluation_path(root: Path, optimizer: str, run_label: str) -> Path:
    """Keep a post-training full-validation result separate from the curve trace."""
    paths = qwen_trial_paths(root, optimizer, run_label)
    return paths.result.with_name(paths.result.stem + ".full_validation.ppl.json")


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


def _require_completed_qwen_baselines(
    root: Path,
    *,
    run_label: str,
    manifest_digest: str,
    expected_epochs: int,
) -> None:
    """Reject a proposal until the three matching formal baselines are final."""
    for optimizer in BASELINE_DISPLAY_NAMES:
        path = qwen_trial_paths(root, optimizer, run_label).result
        if not path.is_file():
            raise RuntimeError(f"missing completed formal baseline: {optimizer}/{run_label}")
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid formal baseline result: {path}") from error
        if (
            result.get("optimizer") != optimizer
            or result.get("run_label") != run_label
            or result.get("data_manifest_sha256") != manifest_digest
            or result.get("completed_epochs") != expected_epochs
            or int(result.get("completed_updates", 0)) <= 0
            or not math.isfinite(float(result.get("final_perplexity", math.nan)))
        ):
            raise RuntimeError(f"formal baseline is not matched and complete: {optimizer}/{run_label}")


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


def render_qwen_candidate_comparison(
    root: Path, *, run_label: str, candidate: str
) -> tuple[Path, Path]:
    """Render one proposal against the three required matched baselines."""
    if candidate not in TRIAL_DISPLAY_NAMES or candidate in BASELINE_DISPLAY_NAMES:
        raise ValueError("candidate renderer requires a non-baseline Qwen trial optimizer")
    traces = {
        BASELINE_DISPLAY_NAMES[optimizer]: _metric_records(qwen_trial_paths(root, optimizer, run_label).metric)
        for optimizer in BASELINE_DISPLAY_NAMES
    }
    traces[TRIAL_DISPLAY_NAMES[candidate]] = _metric_records(qwen_trial_paths(root, candidate, run_label).metric)
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"qwen3_0p6b_{run_label}_{candidate}_metric_steps.png",
        output_root / f"qwen3_0p6b_{run_label}_{candidate}_metric_time.png",
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
            title=f"Qwen3-0.6B {TRIAL_DISPLAY_NAMES[candidate]} versus matched baselines",
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


def _write_qwen_checkpoint(
    *,
    paths: QwenTrialPaths,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    completed_epochs: int,
    active_epoch: int,
    completed_batches_in_active_epoch: int,
    active_epoch_generator_state: torch.Tensor,
    completed_updates: int,
    elapsed_seconds: float,
    final_perplexity: float | None,
    manifest_digest: str,
    peak_memory_mib: float | None,
    config: QwenTrialConfig,
) -> None:
    """Atomically preserve a progress checkpoint below the cache root."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "completed_epochs": completed_epochs,
        "active_epoch": active_epoch,
        "completed_batches_in_active_epoch": completed_batches_in_active_epoch,
        "active_epoch_generator_state": active_epoch_generator_state.cpu(),
        "completed_updates": completed_updates,
        "elapsed_seconds": elapsed_seconds,
        "final_perplexity": final_perplexity,
        "data_manifest_sha256": manifest_digest,
        "peak_memory_mib": peak_memory_mib,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": asdict(config),
    }
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = paths.checkpoint.with_suffix(paths.checkpoint.suffix + ".partial")
    torch.save(checkpoint, partial)
    partial.replace(paths.checkpoint)


def _resume_checkpoint(
    *,
    paths: QwenTrialPaths,
    config: QwenTrialConfig,
    manifest_digest: str,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    device: torch.device,
) -> dict:
    """Load a compatible periodic checkpoint before continuing a trial."""
    if not paths.checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint for Qwen resume: {paths.checkpoint}")
    checkpoint = torch.load(paths.checkpoint, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config")
    expected_config = asdict(config)
    expected_config.pop("resume")
    if not isinstance(saved_config, dict):
        raise RuntimeError("Qwen checkpoint lacks a resolved configuration")
    saved_config = dict(saved_config)
    saved_config.pop("resume", None)
    required = {
        "model",
        "optimizers",
        "completed_epochs",
        "active_epoch",
        "completed_batches_in_active_epoch",
        "active_epoch_generator_state",
        "completed_updates",
        "elapsed_seconds",
        "data_manifest_sha256",
        "torch_rng_state",
    }
    if not required <= checkpoint.keys():
        raise RuntimeError("Qwen checkpoint predates resumable progress state")
    if saved_config != expected_config or checkpoint["data_manifest_sha256"] != manifest_digest:
        raise RuntimeError("Qwen checkpoint is incompatible with this requested trial")
    model.load_state_dict(checkpoint["model"])
    for name, optimizer in optimizers.items():
        if name not in checkpoint["optimizers"]:
            raise RuntimeError(f"Qwen checkpoint lacks optimizer state: {name}")
        optimizer.load_state_dict(checkpoint["optimizers"][name])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    cuda_states = checkpoint.get("cuda_rng_states")
    if device.type == "cuda" and cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    return checkpoint


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


def evaluate_qwen_checkpoint(
    root: Path,
    *,
    optimizer: str,
    run_label: str,
    device: str = "cuda",
) -> dict:
    """Evaluate every held-out token block from a completed trial checkpoint."""
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    paths = qwen_trial_paths(root, optimizer, run_label)
    output = qwen_full_evaluation_path(root, optimizer, run_label)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Qwen full validation: {output}")
    if not paths.checkpoint.is_file():
        raise FileNotFoundError(f"missing Qwen checkpoint: {paths.checkpoint}")
    cache = load_qwen_token_cache(root)
    manifest_digest = _manifest_sha256(cache.manifest)
    checkpoint = torch.load(paths.checkpoint, map_location=selected_device, weights_only=False)
    if checkpoint.get("data_manifest_sha256") != manifest_digest:
        raise RuntimeError("Qwen checkpoint is bound to a different token-cache manifest")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("Qwen checkpoint lacks its resolved trial configuration")
    micro_batch_size = int(checkpoint_config["micro_batch_size"])
    seed = int(checkpoint_config["seed"])
    _, validation_loader = qwen_block_loaders(
        cache,
        micro_batch_size=micro_batch_size,
        workers=0,
        seed=seed,
    )
    model = load_qwen3_model(root).to(selected_device)
    model.load_state_dict(checkpoint["model"])
    validation_batches = len(validation_loader)
    perplexity = _validation_perplexity(model, validation_loader, selected_device, validation_batches)
    result = {
        "optimizer": optimizer,
        "run_label": run_label,
        "full_validation": True,
        "validation_batches": validation_batches,
        "validation_tokens": len(validation_loader.dataset) * cache.sequence_length,
        "perplexity": perplexity,
        "data_manifest_sha256": manifest_digest,
        "checkpoint": str(paths.checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


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
    if config.resume and paths.result.exists():
        raise FileExistsError(f"refusing to resume completed Qwen trial {config.run_label}/{config.optimizer}")
    if not config.resume and any(path.exists() for path in (paths.metric, paths.result, paths.checkpoint)):
        raise FileExistsError(f"refusing to overwrite Qwen trial {config.run_label}/{config.optimizer}")
    if config.optimizer not in BASELINE_DISPLAY_NAMES:
        _require_completed_qwen_baselines(
            config.root,
            run_label=config.run_label,
            manifest_digest=manifest_digest,
            expected_epochs=config.maximum_epochs,
        )
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
    elapsed_offset = 0.0
    completed_updates = 0
    completed_epochs = 0
    active_epoch = 0
    completed_batches_in_active_epoch = 0
    active_epoch_generator_state: torch.Tensor | None = None
    checkpoint_written_at = -1
    stopped_early = False
    final_perplexity: float | None = None
    if config.resume:
        checkpoint = _resume_checkpoint(
            paths=paths,
            config=config,
            manifest_digest=manifest_digest,
            model=model,
            optimizers=optimizers,
            device=device,
        )
        completed_updates = int(checkpoint["completed_updates"])
        completed_epochs = int(checkpoint["completed_epochs"])
        active_epoch = int(checkpoint["active_epoch"])
        completed_batches_in_active_epoch = int(checkpoint["completed_batches_in_active_epoch"])
        active_epoch_generator_state = checkpoint["active_epoch_generator_state"].cpu()
        elapsed_offset = float(checkpoint["elapsed_seconds"])
        final_perplexity = checkpoint.get("final_perplexity")
        checkpoint_written_at = completed_updates
        stopped_early = (
            config.maximum_updates is not None and completed_updates >= config.maximum_updates
        )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    first_epoch = active_epoch if config.resume else 1
    for epoch in range(first_epoch, config.maximum_epochs + 1):
        if stopped_early:
            break
        active_epoch = epoch
        model.train()
        if config.resume and epoch == first_epoch:
            if active_epoch_generator_state is None:
                raise RuntimeError("Qwen checkpoint lacks the active epoch sampler state")
            train_loader.generator.set_state(active_epoch_generator_state)
            skip_batches = completed_batches_in_active_epoch
        else:
            skip_batches = 0
            completed_batches_in_active_epoch = 0
            active_epoch_generator_state = train_loader.generator.get_state().clone()
        for batch_index, (input_ids, labels) in enumerate(train_loader):
            if batch_index < skip_batches:
                continue
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
            completed_batches_in_active_epoch = batch_index + 1
            if completed_updates % config.evaluation_interval_updates == 0:
                elapsed_seconds = elapsed_offset + time.perf_counter() - started
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
                _write_qwen_checkpoint(
                    paths=paths,
                    model=model,
                    optimizers=optimizers,
                    completed_epochs=completed_epochs,
                    active_epoch=active_epoch,
                    completed_batches_in_active_epoch=completed_batches_in_active_epoch,
                    active_epoch_generator_state=active_epoch_generator_state,
                    completed_updates=completed_updates,
                    elapsed_seconds=elapsed_seconds,
                    final_perplexity=final_perplexity,
                    manifest_digest=manifest_digest,
                    peak_memory_mib=peak_memory_mib,
                    config=config,
                )
                checkpoint_written_at = completed_updates
                model.train()
            if config.maximum_updates is not None and completed_updates >= config.maximum_updates:
                stopped_early = True
                break
        if stopped_early:
            break
        completed_epochs = epoch
    elapsed_seconds = elapsed_offset + time.perf_counter() - started
    if final_perplexity is None or completed_updates % config.evaluation_interval_updates:
        final_perplexity = _validation_perplexity(
            model, validation_loader, device, config.validation_batches
        )
        record = {
            "epoch": completed_epochs,
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
    if checkpoint_written_at != completed_updates:
        _write_qwen_checkpoint(
            paths=paths,
            model=model,
            optimizers=optimizers,
            completed_epochs=completed_epochs,
            active_epoch=active_epoch,
            completed_batches_in_active_epoch=completed_batches_in_active_epoch,
            active_epoch_generator_state=active_epoch_generator_state,
            completed_updates=completed_updates,
            elapsed_seconds=elapsed_seconds,
            final_perplexity=final_perplexity,
            manifest_digest=manifest_digest,
            peak_memory_mib=peak_memory_mib,
            config=config,
        )
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
