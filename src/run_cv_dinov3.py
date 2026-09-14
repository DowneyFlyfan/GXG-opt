"""Run one DINOv3 ImageNet-100 optimizer screen or final trial."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from config import DINOV3_IMAGENET100_TASK
from cv_dinov3_experiment import write_cv_dinov3_baseline_plots
from training import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DINOv3 ImageNet-100 classification")
    parser.add_argument("--optimizer", choices=("adamw", "muon", "muown"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--auxiliary-learning-rate", type=float, default=1e-4)
    parser.add_argument("--direction-learning-rate", type=float)
    parser.add_argument("--gain-learning-rate", type=float)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.optimizer == "muown" and (
        arguments.direction_learning_rate is None or arguments.gain_learning_rate is None
    ):
        parser.error("Muown requires --direction-learning-rate and --gain-learning-rate")
    if min(arguments.learning_rate, arguments.auxiliary_learning_rate) <= 0:
        parser.error("learning rates must be positive")
    task = replace(
        DINOV3_IMAGENET100_TASK,
        micro_batch_size=arguments.micro_batch_size,
        gradient_accumulation=arguments.gradient_accumulation,
    )
    root = Path(__file__).resolve().parents[1]
    result = run_trial(
        task,
        arguments.optimizer,
        root,
        workers=arguments.workers,
        run_label=arguments.label,
        maximum_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=0.0,
        auxiliary_learning_rate=arguments.auxiliary_learning_rate,
        muown_direction_lr=arguments.direction_learning_rate,
        muown_gain_lr=arguments.gain_learning_rate,
    )
    if arguments.epochs == task.estimated_epochs:
        try:
            write_cv_dinov3_baseline_plots(root, arguments.label)
        except RuntimeError:
            pass
    print(result)


if __name__ == "__main__":
    main()
