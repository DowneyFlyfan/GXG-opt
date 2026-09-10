#!/usr/bin/env python3
"""Run the paired GPT-2 comparison for AdamW, Muon, and spectral geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multi_step_spectral_geometry import (  # noqa: E402
    MultiStepSpectralOptimizer,
    SpectralPolicyConfig,
)
from optimizers import Muon, muon_parameter_names  # noqa: E402


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "gpt2_wikitext103_spectral.yaml"
)
REQUIRED_METHODS = ("adamw", "muon", "multi_step_spectral_policy")


class OptimizerBundle:
    """Expose one training interface for Muon's matrix and auxiliary optimizers."""

    def __init__(self, optimizers: dict[str, torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [
            group
            for optimizer in self.optimizers.values()
            for group in optimizer.param_groups
        ]

    def zero_grad(self, *, set_to_none: bool) -> None:
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self) -> dict[str, dict]:
        return {
            name: optimizer.state_dict()
            for name, optimizer in self.optimizers.items()
        }

    def load_state_dict(self, state: dict[str, dict]) -> None:
        if set(state) != set(self.optimizers):
            raise ValueError("checkpoint optimizer set does not match this run")
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state[name])


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    method_ids = [str(method["id"]) for method in config.get("methods", [])]
    if tuple(method_ids) != REQUIRED_METHODS:
        raise ValueError(
            f"GPT-2 comparison methods must be {list(REQUIRED_METHODS)}, "
            f"got {method_ids}"
        )
    roles = {str(method["id"]): str(method["role"]) for method in config["methods"]}
    if roles != {
        "adamw": "baseline",
        "muon": "baseline",
        "multi_step_spectral_policy": "proposal",
    }:
        raise ValueError("AdamW and Muon must be baselines and spectral policy a proposal")
    training = config["training"]
    positive = (
        "steps",
        "micro_batch_size",
        "gradient_accumulation",
        "evaluation_batch_size",
        "evaluation_batches",
        "evaluation_interval",
        "checkpoint_interval",
    )
    if any(int(training[name]) < 1 for name in positive):
        raise ValueError("training counts must be positive")
    time_limit = float(training["time_limit_hours"])
    if not 0.0 < time_limit <= 4.0:
        raise ValueError("time_limit_hours must be in (0, 4]")
    if str(training["precision"]) != "bfloat16":
        raise ValueError("the RTX 5090 comparison requires bfloat16")
    if int(config["model"]["sequence_length"]) < 2:
        raise ValueError("sequence length must be at least two")
    for method in config["methods"]:
        if not _resolve(method["optimizer_config"]).is_file():
            raise FileNotFoundError(method["optimizer_config"])
    return config


def comparison_plan(config: dict[str, object]) -> dict[str, object]:
    training = config["training"]
    methods = [str(method["id"]) for method in config["methods"]]
    seeds = [int(seed) for seed in training["seeds"]]
    steps = int(training["steps"])
    sequence_length = int(config["model"]["sequence_length"])
    micro_batch_size = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation"])
    effective_batch_size = micro_batch_size * accumulation
    return {
        "model": config["model"]["name"],
        "dataset": f"{config['dataset']['name']}/{config['dataset']['config']}",
        "methods": methods,
        "seeds": seeds,
        "runs": len(methods) * len(seeds),
        "steps_per_run": steps,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation": accumulation,
        "effective_batch_size": effective_batch_size,
        "input_tokens_per_run": steps * effective_batch_size * sequence_length,
        "time_limit_hours_per_run": float(training["time_limit_hours"]),
        "gpu_count": 1,
        "target_gpu": "NVIDIA GeForce RTX 5090 (32 GB)",
    }


def _dataclass_kwargs(cls, values: dict[str, object]) -> dict[str, object]:
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported {cls.__name__} options: {sorted(unknown)}")
    return values


def _matrix_parameters(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], list[str]]:
    selected_names = muon_parameter_names(model)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name in selected_names and parameter.requires_grad
    ]
    matrices = [parameter for _, parameter in selected]
    matrix_ids = {id(parameter) for parameter in matrices}
    auxiliary = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in matrix_ids
    ]
    return matrices, auxiliary, sorted(name for name, _ in selected)


def optimizer_config(method: dict[str, object]) -> dict[str, object]:
    return yaml.safe_load(
        _resolve(method["optimizer_config"]).read_text(encoding="utf-8")
    )


def build_optimizer(
    model: torch.nn.Module, config: dict[str, object]
) -> tuple[OptimizerBundle, dict[str, object]]:
    values = dict(config["optimizer"])
    name = str(values.pop("name"))
    matrices, auxiliary, matrix_names = _matrix_parameters(model)
    if not matrices:
        raise ValueError("no GPT-2 matrices qualified for Muon/spectral routing")
    if name == "adamw":
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        optimizers = {"adamw": torch.optim.AdamW(model.parameters(), **values)}
    elif name == "muon":
        auxiliary_lr = float(values.pop("auxiliary_lr"))
        learning_rate = float(values.pop("lr"))
        weight_decay = float(values.pop("weight_decay"))
        optimizers = {
            "muon": Muon(
                matrices,
                lr=learning_rate,
                weight_decay=weight_decay,
                **values,
            ),
            "adamw_aux": torch.optim.AdamW(
                auxiliary,
                lr=auxiliary_lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            ),
        }
    elif name == "multi_step_spectral_policy":
        policy = dict(config["policy"])
        dynamics = dict(config["dynamics"])
        values["candidates"] = tuple(float(value) for value in values["candidates"])
        values["policy_interval"] = policy.pop("interval")
        values.update(policy)
        values.update(dynamics)
        values["horizon_weights"] = tuple(values["horizon_weights"])
        spectral_config = SpectralPolicyConfig(
            **_dataclass_kwargs(SpectralPolicyConfig, values)
        )
        groups = [
            {"params": matrices, "use_spectral": True},
            {"params": auxiliary, "use_spectral": False},
        ]
        optimizers = {
            "multi_step_spectral_policy": MultiStepSpectralOptimizer(
                groups, spectral_config
            )
        }
    else:
        raise ValueError(f"unsupported optimizer: {name}")
    bundle = OptimizerBundle(optimizers)
    for group in bundle.param_groups:
        group["peak_lr"] = float(group["lr"])
    return bundle, {"name": name, "matrix_names": matrix_names}


def _learning_rate_factor(
    step: int, total_steps: int, warmup_fraction: float
) -> float:
    warmup = max(round(total_steps * warmup_fraction), 1)
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
    )


def deterministic_batch(
    dataset, *, seed: int, step: int, batch_size: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + step + 10_000)
    indices = torch.randint(len(dataset), (batch_size,), generator=generator)
    rows = dataset[indices.tolist()]["input_ids"]
    return torch.as_tensor(rows, dtype=torch.long)


def evaluate(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch_index in range(batches):
            start = batch_index * batch_size
            indices = [
                index % len(dataset) for index in range(start, start + batch_size)
            ]
            input_ids = torch.as_tensor(
                dataset[indices]["input_ids"], dtype=torch.long
            ).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = causal_lm_loss(model(input_ids=input_ids).logits, input_ids)
            losses.append(float(loss))
    model.train()
    mean_loss = statistics.fmean(losses)
    return {
        "validation_nll": mean_loss,
        "validation_perplexity": math.exp(mean_loss),
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _diagnostics(bundle: OptimizerBundle) -> list[dict[str, object]]:
    diagnostics = []
    for optimizer in bundle.optimizers.values():
        diagnostics.extend(getattr(optimizer, "last_diagnostics", []))
    return _jsonable(diagnostics)


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: OptimizerBundle,
    step_times: list[float],
    best_validation_nll: float,
    elapsed_wall_seconds: float,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step_times": step_times,
            "best_validation_nll": best_validation_nll,
            "elapsed_wall_seconds": elapsed_wall_seconds,
            "torch_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: OptimizerBundle
) -> tuple[int, list[float], float, float]:
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.random.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return (
        int(checkpoint["step"]),
        [float(value) for value in checkpoint["step_times"]],
        float(checkpoint["best_validation_nll"]),
        float(checkpoint["elapsed_wall_seconds"]),
    )


def _archive_incomplete(path: Path) -> None:
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path.replace(
            path.with_name(f"{path.stem}.incomplete-{timestamp}{path.suffix}")
        )


def run_one(
    method: dict[str, object],
    seed: int,
    config: dict[str, object],
    train_dataset,
    validation_dataset,
    output: Path,
    *,
    steps_override: int | None = None,
    checkpoint_enabled: bool = True,
) -> dict[str, object]:
    training = config["training"]
    total_steps = int(
        training["steps"] if steps_override is None else steps_override
    )
    micro_batch_size = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation"])
    effective_batch_size = micro_batch_size * accumulation
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = bool(training["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(training["tf32"])

    model = AutoModelForCausalLM.from_pretrained(
        _resolve(config["model"]["local_path"]),
        local_files_only=True,
        attn_implementation="eager",
    )
    model.float()
    model.config.use_cache = False
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 10_000_000 <= parameter_count <= 200_000_000:
        raise ValueError(f"model has {parameter_count:,} parameters; expected 10M-200M")
    initialization_hash = _model_state_hash(model)
    model.to("cuda")
    selected_config = optimizer_config(method)
    optimizer, runtime = build_optimizer(model, selected_config)

    output.mkdir(parents=True, exist_ok=True)
    (output / "optimizer_config.yaml").write_text(
        yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8"
    )
    checkpoint_path = output / "checkpoint.pt"
    metrics_path = output / "metrics.jsonl"
    diagnostics_path = output / "optimizer_diagnostics.jsonl"
    start_step = 0
    step_times: list[float] = []
    best_validation_nll = math.inf
    prior_elapsed_seconds = 0.0
    if checkpoint_enabled and checkpoint_path.exists():
        (
            start_step,
            step_times,
            best_validation_nll,
            prior_elapsed_seconds,
        ) = _load_checkpoint(checkpoint_path, model, optimizer)
        metrics_mode = diagnostics_mode = "a"
    else:
        _archive_incomplete(metrics_path)
        _archive_incomplete(diagnostics_path)
        metrics_mode = diagnostics_mode = "w"

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    session_started = time.perf_counter()
    completed_step = start_step
    time_limit_reached = False
    last_train_loss = math.nan
    with (
        metrics_path.open(metrics_mode, encoding="utf-8") as metrics_handle,
        diagnostics_path.open(diagnostics_mode, encoding="utf-8") as diagnostics_handle,
    ):
        if start_step == 0:
            initial = evaluate(
                model,
                validation_dataset,
                batch_size=int(training["evaluation_batch_size"]),
                batches=int(training["evaluation_batches"]),
                device=device,
            )
            elapsed = prior_elapsed_seconds + time.perf_counter() - session_started
            initial_metric = {
                "step": 0,
                "train_loss": None,
                **initial,
                "mean_step_seconds": None,
                "elapsed_wall_seconds": elapsed,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
            metrics_handle.write(json.dumps(initial_metric, sort_keys=True) + "\n")
            metrics_handle.flush()
            best_validation_nll = initial["validation_nll"]
            print(json.dumps(initial_metric, sort_keys=True), flush=True)

        for step in range(start_step + 1, total_steps + 1):
            factor = _learning_rate_factor(
                step, total_steps, float(training["warmup_fraction"])
            )
            for group in optimizer.param_groups:
                group["lr"] = float(group["peak_lr"]) * factor
            optimizer.zero_grad(set_to_none=True)
            full_batch = deterministic_batch(
                train_dataset,
                seed=seed,
                step=step,
                batch_size=effective_batch_size,
            )
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            micro_losses = []
            for input_ids in full_batch.split(micro_batch_size):
                input_ids = input_ids.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = causal_lm_loss(model(input_ids=input_ids).logits, input_ids)
                micro_losses.append(float(loss.detach()))
                (loss / accumulation).backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            step_times.append(step_seconds)
            last_train_loss = statistics.fmean(micro_losses)
            completed_step = step
            elapsed = prior_elapsed_seconds + time.perf_counter() - session_started
            time_limit_reached = elapsed >= float(training["time_limit_hours"]) * 3600
            should_evaluate = (
                step % int(training["evaluation_interval"]) == 0
                or step == total_steps
                or time_limit_reached
            )
            if should_evaluate:
                evaluation = evaluate(
                    model,
                    validation_dataset,
                    batch_size=int(training["evaluation_batch_size"]),
                    batches=int(training["evaluation_batches"]),
                    device=device,
                )
                elapsed = prior_elapsed_seconds + time.perf_counter() - session_started
                best_validation_nll = min(
                    best_validation_nll, evaluation["validation_nll"]
                )
                metric = {
                    "step": step,
                    "train_loss": last_train_loss,
                    **evaluation,
                    "mean_step_seconds": statistics.fmean(step_times),
                    "elapsed_wall_seconds": elapsed,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
                metrics_handle.write(json.dumps(metric, sort_keys=True) + "\n")
                metrics_handle.flush()
                diagnostics_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "optimizer": runtime["name"],
                            "total_step_seconds": step_seconds,
                            "tensors": _diagnostics(optimizer),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                diagnostics_handle.flush()
                print(json.dumps(metric, sort_keys=True), flush=True)
            if checkpoint_enabled and (
                step % int(training["checkpoint_interval"]) == 0
                or step == total_steps
                or time_limit_reached
            ):
                _save_checkpoint(
                    checkpoint_path,
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    step_times=step_times,
                    best_validation_nll=best_validation_nll,
                    elapsed_wall_seconds=elapsed,
                )
            if time_limit_reached:
                break

    final_metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line
    ][-1]
    result = {
        "method": method["id"],
        "label": method["label"],
        "role": method["role"],
        "seed": seed,
        "requested_steps": total_steps,
        "completed_steps": completed_step,
        "stop_reason": "four_hour_limit" if time_limit_reached else "step_budget",
        "initialization_hash": initialization_hash,
        "parameter_count": parameter_count,
        "matrix_parameter_count": len(runtime["matrix_names"]),
        "final_validation_nll": float(final_metrics["validation_nll"]),
        "final_validation_perplexity": float(final_metrics["validation_perplexity"]),
        "best_validation_nll": best_validation_nll,
        "best_validation_perplexity": math.exp(best_validation_nll),
        "mean_step_seconds": float(final_metrics["mean_step_seconds"]),
        "elapsed_wall_seconds": float(final_metrics["elapsed_wall_seconds"]),
        "peak_memory_gib": float(final_metrics["peak_memory_gib"]),
    }
    _atomic_json(output / "run_summary.json", result)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    by_method = {}
    for method in REQUIRED_METHODS:
        selected = [run for run in runs if run["method"] == method]
        if not selected:
            continue
        by_method[method] = {
            "label": selected[0]["label"],
            "role": selected[0]["role"],
            "completed_runs": len(selected),
            "mean_final_validation_nll": statistics.fmean(
                float(run["final_validation_nll"]) for run in selected
            ),
            "mean_final_validation_perplexity": statistics.fmean(
                float(run["final_validation_perplexity"]) for run in selected
            ),
            "mean_step_seconds": statistics.fmean(
                float(run["mean_step_seconds"]) for run in selected
            ),
            "mean_peak_memory_gib": statistics.fmean(
                float(run["peak_memory_gib"]) for run in selected
            ),
        }
    return {
        "scope": "fixed-step GPT-2/WikiText-103 optimizer comparison",
        "lower_is_better": [
            "mean_final_validation_nll",
            "mean_final_validation_perplexity",
            "mean_step_seconds",
        ],
        "methods": by_method,
        "runs": runs,
    }


def _write_summary(output: Path, runs: list[dict[str, object]]) -> None:
    _atomic_json(output / "summary.json", summarize(runs))
    rows = sorted(runs, key=lambda run: (str(run["method"]), int(run["seed"])))
    if rows:
        with (output / "final_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _render_if_paired(output: Path) -> tuple[Path, Path] | None:
    from render_gpt2_spectral_comparison import render

    step_graph = output / "gpt2_wikitext103_validation_nll_steps.png"
    time_graph = output / "gpt2_wikitext103_validation_nll_time.png"
    try:
        return (
            render(output, step_graph),
            render(output, time_graph, x_key="elapsed_wall_seconds"),
        )
    except ValueError:
        return None


def run(
    config_path: Path,
    output: Path,
    *,
    methods_override: list[str] | None = None,
    seeds_override: list[int] | None = None,
    steps_override: int | None = None,
    checkpoint_enabled: bool = True,
) -> Path:
    from datasets import load_from_disk

    config = load_config(config_path)
    methods = [
        method
        for method in config["methods"]
        if methods_override is None or method["id"] in methods_override
    ]
    seeds = seeds_override or [int(seed) for seed in config["training"]["seeds"]]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(config, sort_keys=False)
    resolved_path = output / "config_resolved.yaml"
    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != config_text:
        raise ValueError("output directory contains a different resolved configuration")
    resolved_path.write_text(config_text, encoding="utf-8")
    dataset_path = _resolve(config["dataset"]["local_path"])
    manifest_path = dataset_path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing prepared assets manifest: {manifest_path}; "
            "run scripts/prepare_gpt2_wikitext.py first"
        )
    _atomic_json(
        output / "comparison_manifest.json",
        {
            **comparison_plan(config),
            "selection_source": config["selection_source"],
            "assets": json.loads(manifest_path.read_text(encoding="utf-8")),
            "system": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        },
    )
    datasets = load_from_disk(dataset_path)
    state_path = output / "comparison_state.json"
    requested_jobs = []
    for seed in seeds:
        for method in methods:
            run_output = output / "runs" / str(method["id"]) / f"seed_{seed}"
            requested_jobs.append(
                {
                    "method": str(method["id"]),
                    "seed": seed,
                    "status": "pending",
                    "output_directory": str(run_output),
                }
            )
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected = [(job["method"], int(job["seed"])) for job in requested_jobs]
        actual = [(job["method"], int(job["seed"])) for job in state["jobs"]]
        if expected != actual:
            raise ValueError("resume job matrix does not match requested methods and seeds")
    else:
        state = {"jobs": requested_jobs}
        _atomic_json(state_path, state)
    completed = []
    for job in state["jobs"]:
        run_output = Path(job["output_directory"])
        summary_path = run_output / "run_summary.json"
        if job["status"] == "completed" and summary_path.exists():
            completed.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        method = next(method for method in methods if method["id"] == job["method"])
        job["status"] = "running"
        job.pop("error", None)
        _atomic_json(state_path, state)
        print(
            f"[{len(completed) + 1}/{len(state['jobs'])}] "
            f"{method['label']} seed={job['seed']}",
            flush=True,
        )
        try:
            result = run_one(
                method,
                int(job["seed"]),
                config,
                datasets["train"],
                datasets["validation"],
                run_output,
                steps_override=steps_override,
                checkpoint_enabled=checkpoint_enabled,
            )
            completed.append(result)
            job["status"] = "completed"
        except Exception as error:
            job["status"] = "failed"
            job["error"] = f"{type(error).__name__}: {error}"
            _atomic_json(state_path, state)
            _write_summary(output, completed)
            raise
        _atomic_json(state_path, state)
        _write_summary(output, completed)
        _render_if_paired(output)
    _render_if_paired(output)
    print(output, flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpt2_wikitext103_spectral",
    )
    parser.add_argument("--methods", nargs="+", choices=REQUIRED_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    config = load_config(args.config)
    if args.dry_run:
        plan = comparison_plan(config)
        if args.methods:
            plan["methods"] = args.methods
        if args.seeds:
            plan["seeds"] = args.seeds
        if args.steps:
            plan["steps_per_run"] = args.steps
            plan["input_tokens_per_run"] = (
                args.steps
                * int(plan["effective_batch_size"])
                * int(plan["sequence_length"])
            )
        plan["runs"] = len(plan["methods"]) * len(plan["seeds"])
        print(json.dumps(plan, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("the GPT-2 comparison requires CUDA")
    run(
        args.config,
        args.output,
        methods_override=args.methods,
        seeds_override=args.seeds,
        steps_override=args.steps,
        checkpoint_enabled=not args.no_checkpoint,
    )


if __name__ == "__main__":
    main()
