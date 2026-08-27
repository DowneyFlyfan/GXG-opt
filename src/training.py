from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from artifacts import write_metric, write_metric_plot
from config import FormalTask
from data import (
    cifar100_loaders,
    dinov3_cifar100_loaders,
    librispeech_loaders,
    owsm_decode_ctc_ids,
    owsm_librispeech_loaders,
    smollm2_wikitext_loaders,
    wikitext_loaders,
)
from models import create_audio_model, create_cv_model, create_nlp_model, parameter_count
from optimizers import build_optimizers


def configure_reproducibility(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def _model(task: FormalTask) -> nn.Module:
    if task.domain == "nlp":
        return create_nlp_model(task.model)
    if task.domain == "cv":
        return create_cv_model(task.model)
    if task.domain == "audio":
        return create_audio_model(task.model)
    raise ValueError(task.domain)


def _loaders(task: FormalTask, root: Path, workers: int, seed: int = 1337):
    if task.domain == "nlp":
        if task.model == "smollm2_135m":
            return smollm2_wikitext_loaders(root, task.micro_batch_size, workers, seed)
        return wikitext_loaders(root, task.micro_batch_size, workers, seed)
    if task.domain == "cv":
        if task.model == "dinov3_vitb16":
            return dinov3_cifar100_loaders(root, task.micro_batch_size, workers, seed)
        return cifar100_loaders(root, task.micro_batch_size, workers, seed)
    if task.domain == "audio":
        if task.model == "owsm_v3.1_base":
            return owsm_librispeech_loaders(root, task.micro_batch_size, workers, seed)
        return librispeech_loaders(root, task.micro_batch_size, workers, seed)
    raise ValueError(task.domain)


def _model_logits(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    output = model(token_ids)
    return output.logits if hasattr(output, "logits") else output


def _audio_error(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor) -> tuple[int, int]:
    prediction = logits.argmax(dim=-1)
    target_offset = 0
    errors = characters = 0
    for row, length in zip(prediction, lengths, strict=True):
        collapsed = torch.unique_consecutive(row).tolist()
        decoded = [token for token in collapsed if token != 0]
        expected = targets[target_offset : target_offset + length].tolist()
        target_offset += length
        previous = list(range(len(expected) + 1))
        for token in decoded:
            current = [previous[0] + 1]
            for column, expected_token in enumerate(expected, start=1):
                current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (token != expected_token)))
            previous = current
        errors += previous[-1]
        characters += len(expected)
    return errors, characters


def _character_error(prediction: str, target: str) -> tuple[int, int]:
    previous = list(range(len(target) + 1))
    for predicted_character in prediction:
        current = [previous[0] + 1]
        for column, target_character in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (predicted_character != target_character),
                )
            )
        previous = current
    return previous[-1], len(target)


@torch.no_grad()
def _evaluate(task: FormalTask, model: nn.Module, loader, device: torch.device, maximum_batches: int = 64) -> float:
    model.eval()
    correct = total = errors = characters = 0
    for index, batch in enumerate(loader):
        if index >= maximum_batches:
            break
        if task.domain == "nlp":
            token_ids, targets = (value.to(device, non_blocking=True) for value in batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = _model_logits(model, token_ids)
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.numel()
        elif task.domain == "cv":
            images, targets = (value.to(device, non_blocking=True) for value in batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            total += targets.numel()
        elif task.model == "owsm_v3.1_base":
            speech = batch["speech"].to(device, non_blocking=True)
            speech_lengths = batch["speech_lengths"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoder_output, encoder_lengths = model.encode(speech, speech_lengths)
                ctc_tokens = model.ctc.argmax(encoder_output)
            for tokens, length, transcript in zip(ctc_tokens, encoder_lengths, batch["transcripts"], strict=True):
                collapsed = torch.unique_consecutive(tokens[: length.item()])
                decoded = owsm_decode_ctc_ids(
                    Path(__file__).resolve().parents[1],
                    collapsed[collapsed != model.blank_id],
                )
                batch_errors, batch_characters = _character_error(decoded, transcript)
                errors += batch_errors
                characters += batch_characters
        else:
            features, targets, lengths = batch
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(features)
            batch_errors, batch_characters = _audio_error(logits, targets, lengths)
            errors += batch_errors
            characters += batch_characters
    return errors / max(characters, 1) if task.domain == "audio" else correct / max(total, 1)


def _loss(task: FormalTask, model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    if task.domain == "nlp":
        token_ids, targets = (value.to(device, non_blocking=True) for value in batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _model_logits(model, token_ids)
            return functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    if task.domain == "cv":
        images, targets = (value.to(device, non_blocking=True) for value in batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return functional.cross_entropy(model(images), targets)
    if task.model == "owsm_v3.1_base":
        model_batch = {
            name: value.to(device, non_blocking=True)
            for name, value in batch.items()
            if name != "transcripts"
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _ = model(**model_batch)
        return loss
    features, targets, target_lengths = batch
    features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(features).float().log_softmax(dim=-1).transpose(0, 1)
    input_lengths = torch.full((features.size(0),), logits.size(0), dtype=torch.long)
    return functional.ctc_loss(logits, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)


def _metric_label(domain: str) -> str:
    return {"nlp": "Validation next-token accuracy", "cv": "Validation top-1 accuracy", "audio": "Validation character error rate"}[domain]


def _save_trial_checkpoint(
    path: Path,
    model: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
    completed_epoch: int,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
            "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
            "completed_epoch": completed_epoch,
            "elapsed_seconds": elapsed_seconds,
        },
        path,
    )


def _load_trial_checkpoint(
    path: Path,
    model: nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler],
) -> tuple[int, float]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(state["schedulers"][name])
    return int(state["completed_epoch"]), float(state["elapsed_seconds"])


def run_trial(task: FormalTask, optimizer_name: str, root: Path, workers: int = 4, seed: int = 1337) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    configure_reproducibility(seed)
    device = torch.device("cuda")
    train_loader, validation_loader = _loaders(task, root, workers, seed)
    model = _model(task).to(device)
    learning_rate = task.adamw_lr if optimizer_name == "adamw" else task.muon_lr
    optimizers = build_optimizers(model, optimizer_name, learning_rate, task.weight_decay, task.muon_aux_lr)
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=task.estimated_epochs) for name, optimizer in optimizers.items()}
    metric_path = root / "metrics" / task.domain / f"{task.identifier}__{optimizer_name}.jsonl"
    checkpoint_path = root / "results" / task.domain / f"{task.identifier}__{optimizer_name}.checkpoint.pt"
    completed_epoch = 0
    elapsed_seconds = 0.0
    if checkpoint_path.exists():
        completed_epoch, elapsed_seconds = _load_trial_checkpoint(checkpoint_path, model, optimizers, schedulers)
    elif metric_path.exists():
        metric_path.unlink()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, task.estimated_epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            if batch_index % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.zero_grad(set_to_none=True)
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
            if (batch_index + 1) % task.gradient_accumulation == 0:
                for optimizer in optimizers.values():
                    optimizer.step()
        for scheduler in schedulers.values():
            scheduler.step()
        metric = _evaluate(task, model, validation_loader, device)
        write_metric(metric_path, {"epoch": epoch, "metric": metric})
        elapsed_seconds += time.perf_counter() - started
        _save_trial_checkpoint(checkpoint_path, model, optimizers, schedulers, epoch, elapsed_seconds)
        started = time.perf_counter()
        print(f"task={task.identifier} optimizer={optimizer_name} epoch={epoch}/{task.estimated_epochs} metric={metric:.6f}", flush=True)
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": optimizer_name,
        "parameters": parameter_count(model),
        "epochs": task.estimated_epochs,
        "micro_batch_size": task.micro_batch_size,
        "gradient_accumulation": task.gradient_accumulation,
        "seconds": elapsed_seconds + time.perf_counter() - started,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": "completed",
    }
    result_path = root / "results" / task.domain / f"{task.identifier}__{optimizer_name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    checkpoint_path.unlink(missing_ok=True)
    paired = root / "metrics" / task.domain / f"{task.identifier}__{'muon' if optimizer_name == 'adamw' else 'adamw'}.jsonl"
    if paired.exists():
        adamw = metric_path if optimizer_name == "adamw" else paired
        muon = metric_path if optimizer_name == "muon" else paired
        paired_result = root / "results" / task.domain / f"{task.identifier}__{'muon' if optimizer_name == 'adamw' else 'adamw'}.json"
        runtimes = None
        if paired_result.exists():
            other_seconds = json.loads(paired_result.read_text())["seconds"]
            runtimes = {"AdamW": result["seconds"] if optimizer_name == "adamw" else other_seconds, "Muon": result["seconds"] if optimizer_name == "muon" else other_seconds}
        write_metric_plot(adamw, muon, root / "results" / task.domain / f"{task.identifier}_metric_steps.png", _metric_label(task.domain), runtimes)
    return result
