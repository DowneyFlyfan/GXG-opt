from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path

import torch

from config import FORMAL_TASKS, FormalTask
from optimizers import build_optimizers, qualify_ratio
from training import _loaders, _loss, _model, configure_reproducibility


def _profile_batches(task: FormalTask, root: Path, workers: int):
    train, _ = _loaders(task, root, workers=workers)
    return train


def profile(task: FormalTask, optimizer_name: str, root: Path, workers: int = 0, measured_updates: int = 10) -> dict:
    configure_reproducibility(1337)
    device = torch.device("cuda")
    model = _model(task).to(device)
    learning_rate = task.adamw_lr if optimizer_name == "adamw" else task.muon_lr
    optimizers = build_optimizers(model, optimizer_name, learning_rate, task.weight_decay, task.muon_aux_lr)
    batches = list(islice(_profile_batches(task, root, workers), task.gradient_accumulation))

    def update() -> float:
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        for batch in batches:
            (_loss(task, model, batch, device) / task.gradient_accumulation).backward()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for optimizer in optimizers.values():
            optimizer.step()
        torch.cuda.synchronize()
        return time.perf_counter() - started

    for _ in range(4):
        update()
    total_times, update_times = [], []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(measured_updates):
        started = time.perf_counter()
        update_times.append(update())
        torch.cuda.synchronize()
        total_times.append(time.perf_counter() - started)
    return {
        "task": task.identifier,
        "optimizer": optimizer_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "total_update_seconds": sorted(total_times)[len(total_times) // 2],
        "optimizer_seconds": sorted(update_times)[len(update_times) // 2],
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify formal baseline candidates")
    parser.add_argument("--task", choices=[task.identifier for task in FORMAL_TASKS])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = [task for task in FORMAL_TASKS if args.task is None or task.identifier == args.task]
    reports = []
    for task in tasks:
        adamw, muon = profile(task, "adamw", root), profile(task, "muon", root)
        qualification = qualify_ratio(adamw["total_update_seconds"], muon["total_update_seconds"])
        report = {"task": task.identifier, "adamw": adamw, "muon": muon, "ratio": qualification.ratio, "qualified": qualification.qualified}
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    path = root / "records" / "2026-08-25-formal-code-qualification.json"
    path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    if not all(report["qualified"] for report in reports):
        raise SystemExit("One or more candidates exceed the 20% Muon timing limit")


if __name__ == "__main__":
    main()
