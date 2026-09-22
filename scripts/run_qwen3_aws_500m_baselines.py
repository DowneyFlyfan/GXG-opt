"""Tune and run the three scratch Qwen3 baselines on one eight-GPU AWS node."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


RUN_LABEL = os.environ.get("QWEN_RUN_LABEL", "scratch_500m_5ep_batch64_a100_v1")
WORLD_SIZE = int(os.environ.get("QWEN_WORLD_SIZE", "8"))
SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 64
MICRO_BATCH_SIZE = int(os.environ.get("QWEN_MICRO_BATCH_SIZE", "8"))
if GLOBAL_BATCH_SIZE % (WORLD_SIZE * MICRO_BATCH_SIZE):
    raise ValueError("global batch size must be divisible by world size times micro batch size")
GRADIENT_ACCUMULATION = GLOBAL_BATCH_SIZE // (WORLD_SIZE * MICRO_BATCH_SIZE)
DATA_DIRECTORY = os.environ.get("QWEN_DATA_DIRECTORY", ".cache/Fineweb_Edu_2B")
SCREEN_UPDATES = 123
SCREEN_TOKENS = SCREEN_UPDATES * GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH + 1
UPDATES_PER_EPOCH = 763
TOKENS_PER_EPOCH = UPDATES_PER_EPOCH * GLOBAL_BATCH_SIZE * SEQUENCE_LENGTH + 1
FORMAL_UPDATES = UPDATES_PER_EPOCH * 5
SCREEN_CONFIGS = {
    "adamw": [
        {"learning_rate": 0.001},
        {"learning_rate": 0.003},
        {"learning_rate": 0.006},
    ],
    "muon": [
        {"learning_rate": 0.003, "auxiliary_lr": 0.001},
        {"learning_rate": 0.003, "auxiliary_lr": 0.003},
    ],
    "muown": [
        {"direction_lr": 0.01, "gain_lr": 0.0003, "auxiliary_lr": 0.001},
        {"direction_lr": 0.01, "gain_lr": 0.0003, "auxiliary_lr": 0.003},
        {"direction_lr": 0.01, "gain_lr": 0.0003, "auxiliary_lr": 0.01},
    ],
}


def _trial_label(optimizer: str, index: int) -> str:
    return f"{RUN_LABEL}_{optimizer}_tune16m_{index}"


def _result_path(root: Path, optimizer: str, label: str) -> Path:
    return root / "results/nlp" / f"qwen3_0p6b__{label}__{optimizer}.ppl.json"


def _checkpoint_path(root: Path, optimizer: str, label: str) -> Path:
    return (
        root / ".cache/qwen3_0p6b/checkpoints"
        / f"qwen3_0p6b__{label}__{optimizer}.checkpoint.pt"
    )


def _torchrun(root: Path, arguments: list[str]) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(WORLD_SIZE),
            str(root / "src/run_qwen3_ppl.py"),
            "--root",
            str(root),
            *arguments,
        ],
        cwd=root,
        check=True,
    )


def _common_run_arguments(optimizer: str, label: str, *, maximum_epochs: int, maximum_updates: int,
                          train_tokens: int, rate_config: dict[str, float]) -> list[str]:
    command = [
        "run",
        "--initialization",
        "scratch",
        "--optimizer",
        optimizer,
        "--run-label",
        label,
        "--data-directory",
        DATA_DIRECTORY,
        "--micro-batch-size",
        str(MICRO_BATCH_SIZE),
        "--gradient-accumulation",
        str(GRADIENT_ACCUMULATION),
        "--maximum-epochs",
        str(maximum_epochs),
        "--maximum-updates",
        str(maximum_updates),
        "--train-tokens-per-epoch",
        str(train_tokens),
        "--validation-batches",
        "12",
        "--evaluation-interval-updates",
        str(maximum_updates if maximum_epochs == 1 else UPDATES_PER_EPOCH),
        "--checkpoint-interval-updates",
        str(0 if maximum_epochs == 1 else UPDATES_PER_EPOCH),
        "--warmup-updates",
        str(max(25, maximum_updates // 10)),
        "--schedule-updates",
        str(maximum_updates),
        "--minimum-lr-ratio",
        "0.1",
        "--gradient-clip",
        "1",
        "--weight-decay",
        "0.1",
        "--workers",
        "8",
        "--seed",
        "1337",
    ]
    for name, value in rate_config.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    return command


def _winner(root: Path, optimizer: str) -> tuple[str, dict[str, float]]:
    candidates: list[tuple[float, str, dict[str, float]]] = []
    for index, rate_config in enumerate(SCREEN_CONFIGS[optimizer], start=1):
        label = _trial_label(optimizer, index)
        result = json.loads(_result_path(root, optimizer, label).read_text())
        if result.get("optimizer") != optimizer or result.get("completed_updates") != SCREEN_UPDATES:
            raise RuntimeError(f"incomplete screening trial: {optimizer}/{label}")
        candidates.append((float(result["final_perplexity"]), label, rate_config))
    _, label, rate_config = min(candidates)
    return label, rate_config


def _screen_completed(root: Path, optimizer: str, label: str) -> bool:
    """Return whether an existing screen has the requested terminal result."""
    path = _result_path(root, optimizer, label)
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        result.get("optimizer") == optimizer
        and result.get("completed_updates") == SCREEN_UPDATES
    )


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.cwd()).resolve()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices.count(",") + 1 != WORLD_SIZE:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to all eight GPUs or leave it unset")
    record = root / "records/2026-09-16_qwen3_500m_baselines.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({
        "status": "screening_started",
        "run_label": RUN_LABEL,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "world_size": WORLD_SIZE,
        "micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "screen_tokens": SCREEN_TOKENS - 1,
        "formal_tokens_per_epoch": TOKENS_PER_EPOCH - 1,
        "formal_total_tokens": (TOKENS_PER_EPOCH - 1) * 5,
    }, indent=2, sort_keys=True) + "\n")
    for optimizer, candidates in SCREEN_CONFIGS.items():
        for index, rate_config in enumerate(candidates, start=1):
            label = _trial_label(optimizer, index)
            if _screen_completed(root, optimizer, label):
                continue
            _torchrun(root, _common_run_arguments(
                optimizer,
                label,
                maximum_epochs=1,
                maximum_updates=SCREEN_UPDATES,
                train_tokens=SCREEN_TOKENS,
                rate_config=rate_config,
            ))
    winners = {optimizer: _winner(root, optimizer) for optimizer in SCREEN_CONFIGS}
    record.write_text(json.dumps({
        "status": "formal_runs_started",
        "run_label": RUN_LABEL,
        "winners": {name: {"screen_label": label, "rate_config": rates}
                    for name, (label, rates) in winners.items()},
    }, indent=2, sort_keys=True) + "\n")
    for optimizer, (_, rate_config) in winners.items():
        _torchrun(root, _common_run_arguments(
            optimizer,
            RUN_LABEL,
            maximum_epochs=5,
            maximum_updates=FORMAL_UPDATES,
            train_tokens=TOKENS_PER_EPOCH,
            rate_config=rate_config,
        ))
        _checkpoint_path(root, optimizer, RUN_LABEL).unlink(missing_ok=True)
    subprocess.run([sys.executable, str(root / "src/run_qwen3_ppl.py"), "--root", str(root),
                    "render", "--run-label", RUN_LABEL], cwd=root, check=True)
    record.write_text(json.dumps({
        "status": "formal_runs_completed",
        "run_label": RUN_LABEL,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "world_size": WORLD_SIZE,
        "micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "formal_updates": FORMAL_UPDATES,
        "formal_tokens_per_epoch": TOKENS_PER_EPOCH - 1,
        "formal_total_tokens": (TOKENS_PER_EPOCH - 1) * 5,
        "winners": {name: {"screen_label": label, "rate_config": rates}
                    for name, (label, rates) in winners.items()},
        "results": {
            optimizer: json.loads(_result_path(root, optimizer, RUN_LABEL).read_text())
            for optimizer in SCREEN_CONFIGS
        },
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
