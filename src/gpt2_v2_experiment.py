from __future__ import annotations

import copy
import csv
import fcntl
import hashlib
import json
import math
import platform
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

from optimizer_v2.adapter import muon_provenance
from optimizer_v2.optimizer import METHODS, OptimizerV2
from optimizer_v2.probes import lm_loss


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def checkpoint_path(output: Path) -> Path:
    """Keep resumable v2 state in the cache while results retain metrics and plots."""
    resolved_output = Path(output).resolve()
    try:
        relative_output = resolved_output.relative_to(ROOT)
    except ValueError:
        relative_output = Path("external") / hashlib.sha256(
            str(resolved_output).encode("utf-8")
        ).hexdigest()
    return ROOT / ".cache" / "gpt2-v2" / "checkpoints" / relative_output / "checkpoint.pt"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_config(config):
    if tuple(method["id"] for method in config["methods"]) != METHODS:
        raise ValueError("the local optimizer 2.0 profile requires the registered eight methods")
    if [method["role"] for method in config["methods"]] != ["baseline"] * 2 + ["proposal"] * 6:
        raise ValueError("the local profile requires two baselines and six proposals")
    if config["model"]["initialization"] != "random":
        raise ValueError("optimizer 2.0 uses random initialization")
    training = config["training"]
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "evaluation_batch_size",
                "evaluation_interval", "checkpoint_interval", "probe_sequence_length"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if training["precision"] != "bfloat16" or config["model"]["sequence_length"] < 2:
        raise ValueError("local training requires BF16 and at least two tokens")
    if not training["seeds"] or len(set(training["seeds"])) != len(training["seeds"]):
        raise ValueError("seeds must be nonempty and unique")
    hardware = config.setdefault(
        "hardware", {"gpu": "NVIDIA GeForce RTX 5090", "minimum_gpu_memory_gib": 32}
    )
    if not isinstance(hardware.get("gpu"), str) or not hardware["gpu"]:
        raise ValueError("hardware.gpu must be a nonempty string")
    if not isinstance(hardware.get("minimum_gpu_memory_gib"), int) or hardware[
        "minimum_gpu_memory_gib"
    ] < 1:
        raise ValueError("hardware.minimum_gpu_memory_gib must be a positive integer")
    return config


def resolved_config(config, methods=None, seeds=None, steps=None):
    config = copy.deepcopy(config)
    if methods is not None:
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= set(METHODS):
            raise ValueError("invalid local method selection")
        config["methods"] = [method for method in config["methods"] if method["id"] in methods]
    if seeds is not None:
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be nonempty and unique")
        config["training"]["seeds"] = seeds
    if steps is not None:
        if steps < 1:
            raise ValueError("smoke steps must be positive")
        config["training"]["max_steps"] = steps
    for method in config["methods"]:
        options = {}
        if "optimizer_config" in method:
            options = yaml.safe_load(resolve(method["optimizer_config"]).read_text())["optimizer"]
            options = dict(options)
            if options.pop("name") != method["id"]:
                raise ValueError("optimizer configuration ID does not match the registered method")
        method["options"] = options
    return config


def comparison_plan(config, blocks=None):
    training = config["training"]
    batch = training["batch_size"] * training["gradient_accumulation_steps"]
    if blocks is None:
        manifest = resolve(config["dataset"]["local_path"]).parent / "manifest.json"
        if manifest.exists():
            blocks = json.loads(manifest.read_text())["splits"]["train"]["blocks"]
    steps = math.ceil(blocks / batch) * training["epochs"] if blocks else None
    if training.get("max_steps"):
        steps = min(steps, training["max_steps"]) if steps else training["max_steps"]
    return {
        "profile": "optimizer_v2_local", "scope": "single-seed screen" if len(training["seeds"]) == 1 else "paired-seed screen",
        "model": config["model"]["name"], "initialization": "random", "epochs": training["epochs"],
        "dataset": f"{config['dataset']['name']}/{config['dataset']['config']}",
        "train_blocks": blocks, "steps_per_run": steps, "methods": [m["id"] for m in config["methods"]],
        "seeds": training["seeds"], "runs": len(config["methods"]) * len(training["seeds"]),
        "physical_batch_size": training["batch_size"], "effective_batch_size": batch,
        "sequence_length": config["model"]["sequence_length"], "gpu_count": 1,
        "gpu": config["hardware"]["gpu"],
        "minimum_gpu_memory_gib": config["hardware"]["minimum_gpu_memory_gib"],
        "smoke_only": bool(training.get("max_steps")),
        "full_budget_input_tokens_per_run": blocks * training["epochs"] * config["model"]["sequence_length"] if blocks else None,
    }


def require_configured_gpu(config):
    actual_gpu = torch.cuda.get_device_name(0)
    expected_gpu = config["hardware"]["gpu"]
    if actual_gpu != expected_gpu:
        raise RuntimeError(f"this profile requires {expected_gpu}, found {actual_gpu}")


def epoch_order(length, seed, epoch):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed * 1_000_003 + epoch + 10_000))


def epoch_batches(length, seed, epoch, batch_size, accumulation, offset=0):
    order = epoch_order(length, seed, epoch)
    for start in range(offset, length, batch_size * accumulation):
        indices = order[start:start + batch_size * accumulation]
        yield start + len(indices), [chunk.tolist() for chunk in indices.split(batch_size)]


def token_batch(dataset, indices, device):
    return torch.as_tensor(dataset[indices]["input_ids"], dtype=torch.long, device=device)


def evaluate(model, dataset, batch_size, device):
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        with torch.inference_mode():
            for start in range(0, len(dataset), batch_size):
                ids = token_batch(dataset, list(range(start, min(start + batch_size, len(dataset)))), device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss = lm_loss(model(input_ids=ids, use_cache=False).logits, ids)
                tokens = ids.shape[0] * (ids.shape[1] - 1)
                total += float(loss) * tokens
                count += tokens
    finally:
        model.train(was_training)
    mean = total / count
    if not math.isfinite(mean):
        raise FloatingPointError("nonfinite validation loss")
    return {"validation_nll": mean, "validation_perplexity": math.exp(mean) if mean < 700 else None,
            "validation_prediction_tokens": count}


def tree_to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to(item, device) for item in value)
    return value


def save_checkpoint(path, model, optimizer, progress):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "progress": progress,
                "torch_rng": torch.random.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    checkpoint["optimizer"]["state"] = tree_to(checkpoint["optimizer"]["state"], device)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.random.set_rng_state(checkpoint["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    return checkpoint["progress"]


def trim_log(path, step):
    if path.exists():
        lines = []
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item["step"] <= step:
                lines.append(line)
        path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _learning_rate_factor(step: int, total_steps: int, warmup_fraction: float) -> float:
    warmup = max(round(total_steps * warmup_fraction), 1)
    if step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_one(config, method, seed, train, validation, output, stop):
    device = torch.device("cuda")
    training = config["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = training["tf32"]
    torch.backends.cudnn.allow_tf32 = training["tf32"]
    model_config = AutoConfig.from_pretrained(resolve(config["model"]["local_path"]), local_files_only=True)
    model_config.use_cache = False
    model_config.tie_word_embeddings = True
    model = AutoModelForCausalLM.from_config(model_config, attn_implementation="eager").float()
    initial_hash = _model_state_hash(model)
    model.to(device)
    optimizer = OptimizerV2(model, method["id"], method["options"], seed=seed, baseline=config["baseline"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(output)
    metrics_path, diagnostics_path = output / "metrics.jsonl", output / "optimizer_diagnostics.jsonl"
    progress = {"step": 0, "epoch": 0, "offset": 0, "input_tokens": 0, "prediction_tokens": 0,
                "elapsed_wall_seconds": 0.0, "training_step_seconds": 0.0,
                "training_forward_calls": 0, "training_backward_calls": 0,
                "validation_input_tokens": 0, "validation_forward_calls": 0,
                "peak_memory_gib": 0.0, "peak_reserved_memory_gib": 0.0,
                "initialization_hash": initial_hash, "best_validation_nll": None,
                "last_train_loss": None, "last_evaluation": None, "last_evaluation_step": -1}
    if checkpoint.exists():
        progress = load_checkpoint(checkpoint, model, optimizer, device)
        if progress["initialization_hash"] != initial_hash:
            raise ValueError("random model initialization changed on resume")
        trim_log(metrics_path, progress["step"])
        trim_log(diagnostics_path, progress["step"])
    elif metrics_path.exists() or diagnostics_path.exists():
        raise ValueError("run logs exist without a resumable checkpoint")
    anchors = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + 71))[:2].tolist()
    if len(anchors) < 2:
        raise ValueError("training data must contain separate fitting/checking sequences")
    length = min(training["probe_sequence_length"], config["model"]["sequence_length"])
    fit_ids = token_batch(train, [anchors[0]], device)[:, :length]
    check_ids = token_batch(train, [anchors[1]], device)[:, :length]
    audit_order = [index for index in epoch_order(len(train), seed + 99, 0).tolist() if index not in anchors]
    atomic_json(output / "run_manifest.json", {"method": method, "seed": seed, "initialization_hash": initial_hash,
                "calibration_block_indices": anchors, "probe_sequence_length": length,
                "epoch_order_hashes": [hashlib.sha256(epoch_order(len(train), seed, epoch).numpy().tobytes()).hexdigest()
                                       for epoch in range(training["epochs"])],
                "model_config": model.config.to_dict()})
    total_steps = math.ceil(len(train) / (training["batch_size"] * training["gradient_accumulation_steps"])) * training["epochs"]
    step_limit = min(total_steps, training.get("max_steps", total_steps))
    elapsed_before = progress["elapsed_wall_seconds"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.train()

    def checkpoint_now():
        progress["elapsed_wall_seconds"] = elapsed_before + time.perf_counter() - started
        progress["peak_memory_gib"] = max(progress["peak_memory_gib"], torch.cuda.max_memory_allocated() / 1024 ** 3)
        progress["peak_reserved_memory_gib"] = max(progress["peak_reserved_memory_gib"], torch.cuda.max_memory_reserved() / 1024 ** 3)
        save_checkpoint(checkpoint, model, optimizer, progress)

    with metrics_path.open("a") as metrics, diagnostics_path.open("a") as diagnostics:
        def record_evaluation():
            evaluation = evaluate(model, validation, training["evaluation_batch_size"], device)
            progress["validation_input_tokens"] += len(validation) * config["model"]["sequence_length"]
            progress["validation_forward_calls"] += math.ceil(len(validation) / training["evaluation_batch_size"])
            progress["last_evaluation"] = evaluation
            progress["last_evaluation_step"] = progress["step"]
            previous = progress["best_validation_nll"]
            progress["best_validation_nll"] = min(previous, evaluation["validation_nll"]) if previous is not None else evaluation["validation_nll"]
            record = {**progress, **evaluation, "last_evaluation": None,
                      "elapsed_wall_seconds": elapsed_before + time.perf_counter() - started,
                      "peak_memory_gib": max(progress["peak_memory_gib"], torch.cuda.max_memory_allocated() / 1024 ** 3),
                      "peak_reserved_memory_gib": max(progress["peak_reserved_memory_gib"], torch.cuda.max_memory_reserved() / 1024 ** 3),
                      "total_model_token_passes": 2 * progress["input_tokens"] + progress["validation_input_tokens"] + optimizer.cost["auxiliary_input_tokens"],
                      **optimizer.cost, **optimizer.persistent_state_bytes()}
            metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()
            print(json.dumps({"method": method["id"], **record}, allow_nan=False), flush=True)

        if progress["last_evaluation"] is None:
            record_evaluation()
            checkpoint_now()
        uncommitted_rng = None
        try:
            while progress["epoch"] < training["epochs"] and progress["step"] < step_limit and not stop["requested"]:
                epoch = progress["epoch"]
                batches = epoch_batches(len(train), seed, epoch, training["batch_size"],
                                        training["gradient_accumulation_steps"], progress["offset"])
                for next_offset, chunks in batches:
                    if stop["requested"]:
                        break
                    step = progress["step"] + 1
                    uncommitted_rng = {
                        "step": progress["step"], "torch": torch.random.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(), "numpy": np.random.get_state(),
                        "python": random.getstate(), "lrs": [group["lr"] for group in optimizer.param_groups],
                    }
                    factor = _learning_rate_factor(step, total_steps, training["warmup_fraction"])
                    for group in optimizer.param_groups:
                        group["lr"] = group["peak_lr"] * factor
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    step_started = time.perf_counter()
                    rows = sum(len(chunk) for chunk in chunks)
                    train_loss = 0.0
                    for chunk in chunks:
                        ids = token_batch(train, chunk, device)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(input_ids=ids, use_cache=False).logits
                            loss = lm_loss(logits, ids)
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError("nonfinite training loss; update not committed")
                        weight = len(chunk) / rows
                        train_loss += float(loss.detach()) * weight
                        (loss * weight).backward()
                        del logits, loss, ids
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
                    if not bool(torch.isfinite(gradient_norm)):
                        raise FloatingPointError("nonfinite accumulated gradients; update not committed")
                    audit_ids = None
                    if method["id"].startswith("feature_remap") and audit_order:
                        audit_index = audit_order[((step - 1) // method["options"].get("interval", 8)) % len(audit_order)]
                        audit_ids = token_batch(train, [audit_index], device)[:, :length]
                    optimizer.step(fit_ids, check_ids, audit_ids)
                    torch.cuda.synchronize()
                    progress["training_step_seconds"] += time.perf_counter() - step_started
                    progress.update(step=step, offset=next_offset, last_train_loss=train_loss)
                    progress["input_tokens"] += rows * config["model"]["sequence_length"]
                    progress["prediction_tokens"] += rows * (config["model"]["sequence_length"] - 1)
                    progress["training_forward_calls"] += len(chunks)
                    progress["training_backward_calls"] += len(chunks)
                    epoch_end = next_offset == len(train)
                    if epoch_end:
                        progress.update(epoch=epoch + 1, offset=0)
                    diagnostics.write(json.dumps({**optimizer.last_diagnostics, "gradient_norm": float(gradient_norm),
                                                   **optimizer.cost}, allow_nan=False) + "\n")
                    diagnostics.flush()
                    if step % training["evaluation_interval"] == 0 or epoch_end or step == step_limit:
                        record_evaluation()
                    if step % training["checkpoint_interval"] == 0 or epoch_end or stop["requested"] or step == step_limit:
                        checkpoint_now()
                    if step % 10 == 0:
                        print(f"{method['id']} step {step}/{step_limit} epoch {progress['epoch'] + progress['offset'] / len(train):.3f} loss {train_loss:.5f}", flush=True)
                    if step >= step_limit or stop["requested"]:
                        break
            if progress["step"] >= step_limit and progress["last_evaluation_step"] != progress["step"]:
                record_evaluation()
            checkpoint_now()
        except FloatingPointError:
            if uncommitted_rng is not None and progress["step"] == uncommitted_rng["step"]:
                torch.random.set_rng_state(uncommitted_rng["torch"])
                torch.cuda.set_rng_state_all(uncommitted_rng["cuda"])
                np.random.set_state(uncommitted_rng["numpy"])
                random.setstate(uncommitted_rng["python"])
                for group, lr in zip(optimizer.param_groups, uncommitted_rng["lrs"]):
                    group["lr"] = lr
            checkpoint_now()
            raise
    if stop["requested"] and progress["step"] < step_limit:
        return None
    evaluation = progress["last_evaluation"]
    result = {"method": method["id"], "label": method["label"], "role": method["role"], "seed": seed,
              "steps": progress["step"], "epochs_completed": progress["epoch"],
              "initialization_hash": initial_hash, "input_tokens": progress["input_tokens"],
              "prediction_tokens": progress["prediction_tokens"], "smoke_only": bool(training.get("max_steps")),
              "final_validation_nll": evaluation["validation_nll"], "final_validation_perplexity": evaluation["validation_perplexity"],
              "best_validation_nll": progress["best_validation_nll"],
              "elapsed_wall_seconds": progress["elapsed_wall_seconds"],
              "mean_step_seconds": progress["training_step_seconds"] / max(progress["step"], 1),
              "training_tokens_per_second": progress["input_tokens"] / max(progress["training_step_seconds"], 1e-12),
              "peak_memory_gib": progress["peak_memory_gib"],
              "peak_reserved_memory_gib": progress["peak_reserved_memory_gib"],
              "validation_input_tokens": progress["validation_input_tokens"],
              "training_forward_calls": progress["training_forward_calls"],
              "training_backward_calls": progress["training_backward_calls"],
              "validation_forward_calls": progress["validation_forward_calls"],
              "total_model_token_passes": 2 * progress["input_tokens"] + progress["validation_input_tokens"] + optimizer.cost["auxiliary_input_tokens"],
              **optimizer.cost, **optimizer.persistent_state_bytes()}
    with checkpoint.open("rb") as handle:
        result["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    atomic_json(output / "run_summary.json", result)
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def write_summary(output, completed):
    atomic_json(output / "summary.json", {"scope": "optimizer 2.0 screen; fixed hyperparameters; no superiority claim",
                                          "runs": completed, "completed_runs": len(completed)})
    if completed:
        with (output / "final_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(completed[0]))
            writer.writeheader()
            writer.writerows(completed)
    lines = ["# GPT-2 optimizer 2.0 comparison", "",
             "Fixed hyperparameter screen; these results do not establish superiority or novelty.", "",
             "| Method | Seed | Epochs completed | Updates | Validation PPL | Peak GiB |", "|---|---:|---:|---:|---:|---:|"]
    for result in completed:
        lines.append(f"| {result['method']} | {result['seed']} | {result['epochs_completed']} | {result['steps']} | {result['final_validation_perplexity']:.6f} | {result['peak_memory_gib']:.3f} |")
    if any(result["smoke_only"] for result in completed):
        lines.extend(["", "This directory contains bounded smoke runs, not the five-epoch comparison."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _validation_perplexity(record):
    value = record.get("validation_perplexity")
    if value is not None:
        return float(value)
    return math.exp(float(record["validation_nll"]))


def render_recorded_perplexity(output):
    """Render PPL directly from recorded GPT-2 metric traces.

    This is deliberately unavailable for historical accuracy-only artifacts:
    perplexity cannot be recovered from argmax accuracy.  NLL traces, by
    contrast, have an exact pointwise PPL conversion.
    """
    from matplotlib.figure import Figure

    traces = []
    for path in sorted(output.glob("runs/*/seed_*/metrics.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not records:
            continue
        summary_path = path.with_name("run_summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        method = path.parents[1].name
        seed = path.parent.name.removeprefix("seed_")
        traces.append((summary.get("label", method) + f" (seed {seed})", records))
    if not traces:
        raise ValueError("no recorded GPT-2 NLL/PPL traces were found")
    outputs = []
    for key, suffix, xlabel in (("step", "steps", "Optimizer update"),
                                ("elapsed_wall_seconds", "time", "Cumulative wall-clock time (seconds)")):
        figure = Figure(figsize=(10, 6))
        axis = figure.subplots()
        for index, (label, records) in enumerate(traces):
            axis.plot([record[key] for record in records],
                      [_validation_perplexity(record) for record in records],
                      label=label, color=("#475569", "#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706")[index % 6])
        axis.set(title="GPT-2 / WikiText-103 optimizer comparison",
                 xlabel=xlabel, ylabel="Validation perplexity (lower is better)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        output_path = output / f"gpt2_wikitext103_validation_perplexity_{suffix}.png"
        figure.savefig(output_path, dpi=160)
        outputs.append(output_path)
    return tuple(outputs)


def render_comparison(output, config, completed):
    from matplotlib.figure import Figure

    methods = config["methods"]
    paired = set(config["training"]["seeds"])
    for method in methods:
        paired &= {result["seed"] for result in completed if result["method"] == method["id"]}
    if not paired:
        return
    colors = ("#475569", "#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2", "#db2777")
    for key, suffix, xlabel in (("step", "steps", "Optimizer update"),
                                ("elapsed_wall_seconds", "time", "Cumulative wall-clock time (seconds)")):
        figure = Figure(figsize=(10, 6))
        axis = figure.subplots()
        for method, color in zip(methods, colors):
            for seed in sorted(paired):
                path = output / "runs" / method["id"] / f"seed_{seed}" / "metrics.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines() if line]
                label = method["label"] if len(paired) == 1 else f"{method['label']} (seed {seed})"
                axis.plot([record[key] for record in records],
                          [_validation_perplexity(record) for record in records], label=label, color=color)
        scope = "single-seed screen" if len(paired) == 1 else "paired-seed screen"
        if config["training"].get("max_steps"):
            scope += "; bounded smoke"
        axis.set(title=f"GPT-2 / WikiText-103 optimizer 2.0 ({scope})",
                 xlabel=xlabel, ylabel="Validation perplexity (lower is better)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / f"gpt2_wikitext103_validation_perplexity_{suffix}.png", dpi=160)


def run(config, output):
    from datasets import load_from_disk
    if not torch.cuda.is_available():
        raise RuntimeError("local GPT-2 training requires CUDA")
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("this comparison directory is already in use") from error
        return _run_locked(config, output, load_from_disk)


def _run_locked(config, output, load_from_disk):
    provenance = muon_provenance()
    require_configured_gpu(config)
    dataset_path = resolve(config["dataset"]["local_path"])
    assets = json.loads((dataset_path.parent / "manifest.json").read_text())
    if assets["sequence_length"] != config["model"]["sequence_length"] or assets.get("model_init") != "random":
        raise ValueError("prepared assets do not match the random-init experiment")
    datasets = load_from_disk(dataset_path)
    digests = assets.get("data_sha256", {})
    actual_files = {str(path.relative_to(dataset_path)) for path in dataset_path.rglob("*.arrow")}
    if not digests or set(digests) != actual_files:
        raise ValueError("dataset file inventory does not match its manifest")
    for relative, expected in digests.items():
        with (dataset_path / relative).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError(f"dataset checksum mismatch: {relative}")
    for split in ("train", "validation"):
        if not len(datasets[split]) or len(datasets[split]) != assets["splits"][split]["blocks"]:
            raise ValueError("dataset split is empty or does not match its manifest")
        if datasets[split]._fingerprint != assets["splits"][split]["fingerprint"]:
            raise ValueError("dataset fingerprint does not match its manifest")
    source_paths = [Path(__file__), ROOT / "scripts/run_gpt2_v2_comparison.py",
                    ROOT / "scripts/run_gpt2_v2_5090.sh", ROOT / "scripts/prepare_gpt2_wikitext.py",
                    *sorted((ROOT / "src/optimizer_v2").glob("*.py"))]
    identity = {"config": config, "assets": assets, "baseline": provenance,
                "runtime": {"transformers": __import__("transformers").__version__,
                            "datasets": __import__("datasets").__version__, "numpy": np.__version__,
                            "python": platform.python_version()},
                "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
                "model_config_sha256": hashlib.sha256((resolve(config["model"]["local_path"]) / "config.json").read_bytes()).hexdigest()}
    config_text = yaml.safe_dump(identity, sort_keys=False)
    resolved_path = output / "config_resolved.yaml"
    if resolved_path.exists() and resolved_path.read_text() != config_text:
        raise ValueError("resume configuration, data, source, or optimizer implementation changed; choose a new output")
    resolved_path.write_text(config_text)
    for path in source_paths:
        snapshot = output / "source_snapshot" / path.relative_to(ROOT)
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(path.read_bytes())
    atomic_json(output / "comparison_manifest.json", {**comparison_plan(config, len(datasets["train"])),
                "assets": assets, "baseline": provenance, "system": {"platform": platform.platform(),
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "transformers": __import__("transformers").__version__, "datasets": __import__("datasets").__version__},
                "probe_policy": "training-only; separate fit/check; dropout disabled; independent RNG",
                "approximations": ["sampled attention rows/edges", "stale paired embedding metric", "block-diagonal feature maps",
                                   "periodic age-separated momentum", "local QK output surrogate", "reduced LN GGN"],
                "prediction_gate_policy": "diagnostic only; all six candidates remain included"})
    jobs = [{"method": method["id"], "seed": seed, "status": "pending"}
            for seed in config["training"]["seeds"] for method in config["methods"]]
    state_path = output / "comparison_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": jobs}
    if [(j["method"], j["seed"]) for j in state["jobs"]] != [(j["method"], j["seed"]) for j in jobs]:
        raise ValueError("resume job matrix changed")
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    completed = []
    try:
        for job in state["jobs"]:
            run_output = output / "runs" / job["method"] / f"seed_{job['seed']}"
            summary = run_output / "run_summary.json"
            checkpoint = checkpoint_path(run_output)
            if job["status"] == "completed" and summary.exists() and checkpoint.exists():
                result = json.loads(summary.read_text())
                expected_steps = comparison_plan(config, len(datasets["train"]))["steps_per_run"]
                if result["method"] != job["method"] or result["seed"] != job["seed"] or result["steps"] != expected_steps:
                    raise ValueError("completed run summary does not match the requested job")
                with checkpoint.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != result.get("checkpoint_sha256"):
                        raise ValueError("completed checkpoint checksum mismatch")
                completed.append(result)
                continue
            if stop["requested"]:
                break
            job["status"] = "running"
            job.pop("error", None)
            atomic_json(state_path, state)
            method = next(method for method in config["methods"] if method["id"] == job["method"])
            print(f"[{len(completed) + 1}/{len(jobs)}] {method['label']} seed={job['seed']}", flush=True)
            try:
                result = run_one(config, method, job["seed"], datasets["train"], datasets["validation"], run_output, stop)
                if result is None:
                    job["status"] = "paused"
                else:
                    completed.append(result)
                    job["status"] = "completed"
            except Exception as error:
                job.update(status="failed", error=f"{type(error).__name__}: {error}")
                atomic_json(state_path, state)
                write_summary(output, completed)
                raise
            atomic_json(state_path, state)
            write_summary(output, completed)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    render_comparison(output, config, completed)
    print(f"{'Paused' if stop['requested'] else 'Completed'}: {output}", flush=True)
    return 130 if stop["requested"] else 0
