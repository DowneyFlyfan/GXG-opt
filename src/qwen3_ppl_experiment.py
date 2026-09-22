"""Metric paths and rendering for matched Qwen3 validation perplexity trials."""

from __future__ import annotations

import json
import hashlib
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plot
import torch
import torch.nn.functional as functional
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from qwen3_data import load_qwen_token_cache, qwen_block_loaders
from qwen3_model import build_qwen_optimizers, load_qwen3_model, qwen_chunked_loss


BASELINE_DISPLAY_NAMES = {"adamw": "AdamW", "muon": "Muon", "muown": "Muown"}
TRIAL_DISPLAY_NAMES = {
    **BASELINE_DISPLAY_NAMES,
    "proposal_notch_v1": "Proposal-notch Muon",
    "routing_resistance_v1": "Routing-resistance Muon",
    "tied_path_curvature_v1": "Tied-path curvature",
    "feature_remap_cohort_v1": "Feature-remap cohort Muon",
    "feature_scalar_cohort_v1": "Feature-scalar cohort Muon",
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
    baseline_run_label: str | None = None
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
    maximum_epochs: int = 5
    maximum_updates: int | None = None
    validation_batches: int = 1
    evaluation_interval_updates: int = 1_000
    checkpoint_interval_updates: int | None = None
    workers: int = 0
    seed: int = 1337
    device: str = "cuda"
    activation_checkpointing: bool = False
    resume: bool = False
    initialization: str = "pretrained"
    warmup_updates: int = 0
    schedule_updates: int = 0
    minimum_lr_ratio: float = 0.1
    gradient_clip: float = 0.0
    data_directory: str | None = None
    train_tokens_per_epoch: int | None = None


@dataclass(frozen=True)
class _DistributedRuntime:
    """The process-group details derived by ``torchrun`` for one trial."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def _distributed_runtime(requested_device: str) -> _DistributedRuntime:
    """Initialize a CUDA process group only when ``torchrun`` requested one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1:
        return _DistributedRuntime(rank=0, world_size=1, local_rank=0, device=torch.device(requested_device))
    if not requested_device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("multi-process Qwen training requires CUDA")
    torch.cuda.set_device(local_rank)
    distributed.init_process_group(backend="nccl")
    return _DistributedRuntime(rank=rank, world_size=world_size, local_rank=local_rank, device=torch.device("cuda", local_rank))


def _broadcast_float(value: float | None, runtime: _DistributedRuntime) -> float:
    """Return rank zero's scalar validation result to every data-parallel rank."""
    if runtime.world_size == 1:
        if value is None:
            raise RuntimeError("single-process validation did not produce a value")
        return value
    payload = torch.tensor(
        [math.nan if value is None else value], dtype=torch.float64, device=runtime.device
    )
    distributed.broadcast(payload, src=0)
    return float(payload.item())


def _distributed_validation_perplexity(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    maximum_batches: int,
    runtime: _DistributedRuntime,
) -> float:
    """Evaluate once on rank zero, then publish the matched metric to all ranks."""
    module = model.module if isinstance(model, DistributedDataParallel) else model
    value = _validation_perplexity(module, loader, device, maximum_batches) if runtime.primary else None
    return _broadcast_float(value, runtime)


def qwen_learning_rate_scale(step: int, *, warmup: int, total: int, minimum: float) -> float:
    """Linear warmup followed by cosine decay, indexed by committed update."""
    if warmup and step <= warmup:
        return step / warmup
    if total <= 0:
        return 1.0
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


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
    from_scratch = all(records[0].get("initialization") == "scratch" for records in traces.values())
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
            ylabel=("Validation perplexity (log scale; lower is better)" if from_scratch
                    else "Validation perplexity (lower is better)"),
            yscale="log" if from_scratch else "linear",
            title="Qwen3-0.6B matched optimizer baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def render_qwen_candidate_comparison(
    root: Path, *, run_label: str, candidate: str, baseline_run_label: str | None = None
) -> tuple[Path, Path]:
    """Render one proposal against the three required matched baselines."""
    if candidate not in TRIAL_DISPLAY_NAMES or candidate in BASELINE_DISPLAY_NAMES:
        raise ValueError("candidate renderer requires a non-baseline Qwen trial optimizer")
    baseline_label = baseline_run_label or run_label
    traces = {
        BASELINE_DISPLAY_NAMES[optimizer]: _metric_records(qwen_trial_paths(root, optimizer, baseline_label).metric)
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


def _restore_cuda_rng_states(states: list[torch.Tensor]) -> None:
    """Restore serialized CUDA RNG bytes on the CPU, as required by PyTorch."""
    torch.cuda.set_rng_state_all([state.cpu() for state in states])


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
    checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    saved_config = checkpoint.get("config")
    expected_config = asdict(config)
    expected_config.pop("resume")
    expected_config.pop("maximum_updates")
    if not isinstance(saved_config, dict):
        raise RuntimeError("Qwen checkpoint lacks a resolved configuration")
    saved_config = dict(saved_config)
    saved_config.pop("resume", None)
    saved_config.pop("maximum_updates", None)
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
        _restore_cuda_rng_states(cuda_states)
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
            if getattr(model, "_qwen_chunked_loss", False):
                loss = qwen_chunked_loss(model, input_ids.to(device), labels.to(device)) * labels.numel()
            else:
                logits = _logits(model(input_ids=input_ids.to(device), use_cache=False))
                loss = functional.cross_entropy(
                    logits.float().reshape(-1, logits.size(-1)),
                    labels.to(device).reshape(-1), reduction="sum",
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
    checkpoint = torch.load(paths.checkpoint, map_location="cpu", weights_only=False)
    cache = load_qwen_token_cache(root, directory=checkpoint.get("config", {}).get("data_directory"))
    manifest_digest = _manifest_sha256(cache.manifest)
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
    model = (load_qwen3_model(root, initialization="scratch")
             if checkpoint_config.get("initialization") == "scratch" else load_qwen3_model(root)).to(selected_device)
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
    if config.initialization not in {"scratch", "pretrained"}:
        raise ValueError("invalid initialization")
    if config.warmup_updates < 0 or config.schedule_updates < 0 or config.gradient_clip < 0 or not 0 <= config.minimum_lr_ratio <= 1:
        raise ValueError("invalid learning-rate schedule or gradient clipping")
    runtime = _distributed_runtime(config.device)
    device = runtime.device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config.seed)
    cache = load_qwen_token_cache(config.root, directory=config.data_directory)
    manifest_digest = _manifest_sha256(cache.manifest)
    paths = qwen_trial_paths(config.root, config.optimizer, config.run_label)
    if runtime.world_size > 1 and config.resume:
        raise RuntimeError("distributed Qwen trials do not support checkpoint resume")
    if config.resume and paths.result.exists():
        completed_result = json.loads(paths.result.read_text())
        completed_updates = int(completed_result.get("completed_updates", -1))
        if config.maximum_updates is None or completed_updates >= config.maximum_updates:
            raise FileExistsError(
                f"refusing to resume completed Qwen trial {config.run_label}/{config.optimizer}"
            )
    if not config.resume and any(path.exists() for path in (paths.metric, paths.result, paths.checkpoint)):
        raise FileExistsError(f"refusing to overwrite Qwen trial {config.run_label}/{config.optimizer}")
    if config.optimizer not in BASELINE_DISPLAY_NAMES:
        _require_completed_qwen_baselines(
            config.root,
            run_label=config.baseline_run_label or config.run_label,
            manifest_digest=manifest_digest,
            expected_epochs=config.maximum_epochs,
        )
    train_loader, validation_loader = qwen_block_loaders(
        cache,
        micro_batch_size=config.micro_batch_size,
        workers=config.workers,
        seed=config.seed,
        train_tokens_per_epoch=config.train_tokens_per_epoch,
        world_size=runtime.world_size,
        rank=runtime.rank,
    )
    model = (load_qwen3_model(config.root, initialization="scratch")
             if config.initialization == "scratch" else load_qwen3_model(config.root)).to(device)
    initial_digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        initial_digest.update(name.encode())
        initial_digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    initial_sha256 = initial_digest.hexdigest()

    if config.activation_checkpointing:
        enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_checkpointing):
            raise TypeError("Qwen activation checkpointing is unavailable on this model")
        enable_checkpointing()
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
    base_rates = [(group, {key: group[key] for key in ("lr", "direction_lr", "gain_lr") if key in group})
                  for optimizer in optimizers.values() for group in getattr(optimizer, "param_groups", [])]
    feature_optimizer = optimizers.get("feature_remap_cohort_v1") or optimizers.get("feature_scalar_cohort_v1")
    if feature_optimizer is not None:
        fit_ids, _ = train_loader.dataset[0]
        check_ids, _ = train_loader.dataset[1]
        target_modules = sorted(
            name.removesuffix(".weight")
            for name in feature_optimizer.adapter.matrix_names
            if name.endswith("mlp.down_proj.weight")
        )
        feature_optimizer.configure_anchors(
            model, fit_ids.unsqueeze(0), check_ids.unsqueeze(0), target_modules, interval=8,
            mode="scalar" if config.optimizer == "feature_scalar_cohort_v1" else "map",
        )
    training_model: torch.nn.Module = (
        DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            broadcast_buffers=False,
        )
        if runtime.world_size > 1
        else model
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
    if config.initialization == "scratch" and not config.resume:
        initial_perplexity = _distributed_validation_perplexity(
            training_model, validation_loader, device, config.validation_batches, runtime
        )
        if runtime.primary:
            with paths.metric.open("a") as handle:
                handle.write(json.dumps({"epoch": 0, "step": 0, "elapsed_seconds": time.perf_counter() - started,
                                         "perplexity": initial_perplexity, "token_exposure": 0,
                                         "initialization": config.initialization, "initial_model_sha256": initial_sha256,
                                         "data_manifest_sha256": manifest_digest,
                                         "world_size": runtime.world_size}) + "\n")
    first_epoch = active_epoch if config.resume else 1
    for epoch in range(first_epoch, config.maximum_epochs + 1):
        if stopped_early:
            if completed_batches_in_active_epoch >= full_microbatches:
                completed_epochs = epoch
            break
        active_epoch = epoch
        training_model.train()
        sampler = getattr(train_loader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        if config.resume and epoch == first_epoch:
            if active_epoch_generator_state is None:
                raise RuntimeError("Qwen checkpoint lacks the active epoch sampler state")
            train_loader.generator.set_state(active_epoch_generator_state)
            skip_batches = completed_batches_in_active_epoch
        else:
            skip_batches = 0
            completed_batches_in_active_epoch = 0
            active_epoch_generator_state = train_loader.generator.get_state().clone()
        full_microbatches = (len(train_loader) // config.gradient_accumulation) * config.gradient_accumulation
        for batch_index, (input_ids, labels) in enumerate(train_loader):
            # Scratch comparisons commit only full effective batches, including epoch tails.
            if config.initialization == "scratch" and batch_index >= full_microbatches:
                break
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
                if config.initialization == "scratch" and runtime.world_size == 1:
                    loss = qwen_chunked_loss(training_model, input_ids.to(device), labels.to(device))
                else:
                    logits = _logits(training_model(input_ids=input_ids.to(device), use_cache=False))
                    loss = functional.cross_entropy(
                        logits.float().reshape(-1, logits.size(-1)), labels.to(device).reshape(-1)
                    )
            (loss / config.gradient_accumulation).backward()
            if (batch_index + 1) % config.gradient_accumulation:
                continue
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            scale = qwen_learning_rate_scale(completed_updates + 1, warmup=config.warmup_updates,
                                             total=config.schedule_updates, minimum=config.minimum_lr_ratio)
            for group, rates in base_rates:
                for key, value in rates.items():
                    group[key] = value * scale
            for optimizer in optimizers.values():
                optimizer.step()
            completed_updates += 1
            completed_batches_in_active_epoch = batch_index + 1
            if completed_updates % config.evaluation_interval_updates == 0:
                elapsed_seconds = elapsed_offset + time.perf_counter() - started
                final_perplexity = _distributed_validation_perplexity(
                    training_model, validation_loader, device, config.validation_batches, runtime
                )
                peak_memory_mib = (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.type == "cuda"
                    else None
                )
                record = {
                    "epoch": epoch,
                    "last_microbatch_loss": float(loss.detach()),
                    "learning_rate_scale": scale,
                    "step": completed_updates,
                    "elapsed_seconds": elapsed_seconds,
                    "token_exposure": (
                        completed_updates
                        * config.micro_batch_size
                        * config.gradient_accumulation
                        * cache.sequence_length
                        * runtime.world_size
                    ),
                    "perplexity": final_perplexity,
                    "data_manifest_sha256": manifest_digest,
                    "peak_memory_mib": peak_memory_mib,
                }
                if runtime.primary:
                    with paths.metric.open("a") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                checkpoint_interval = (
                    config.evaluation_interval_updates
                    if config.checkpoint_interval_updates is None
                    else config.checkpoint_interval_updates
                )
                if (
                    runtime.primary
                    and checkpoint_interval > 0
                    and completed_updates % checkpoint_interval == 0
                ):
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
                training_model.train()
            if config.maximum_updates is not None and completed_updates >= config.maximum_updates:
                stopped_early = True
                break
        if stopped_early:
            break
        completed_epochs = epoch
    elapsed_seconds = elapsed_offset + time.perf_counter() - started
    if final_perplexity is None or completed_updates % config.evaluation_interval_updates:
        final_perplexity = _distributed_validation_perplexity(
            training_model, validation_loader, device, config.validation_batches, runtime
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
                * runtime.world_size
            ),
            "perplexity": final_perplexity,
            "data_manifest_sha256": manifest_digest,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            ),
        }
        if runtime.primary:
            with paths.metric.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    token_exposure = (completed_updates * config.micro_batch_size * config.gradient_accumulation
                      * cache.sequence_length * runtime.world_size)
    if (
        runtime.primary
        and checkpoint_interval > 0
        and checkpoint_written_at != completed_updates
    ):
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
        "initialization": config.initialization,
        "initial_model_sha256": initial_sha256,
        "config": {**asdict(config), "root": str(config.root)},
        "completed_epochs": completed_epochs,
        "completed_updates": completed_updates,
        "final_perplexity": final_perplexity,
        "elapsed_seconds": elapsed_seconds,
        "data_manifest_sha256": manifest_digest,
        "peak_memory_mib": peak_memory_mib,
    }
    result["world_size"] = runtime.world_size
    if runtime.primary:
        paths.result.parent.mkdir(parents=True, exist_ok=True)
        paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if runtime.world_size > 1:
        distributed.barrier()
        distributed.destroy_process_group()
    return result
