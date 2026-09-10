"""Run matched GPT2-12x512 Muown rate screens and final trials."""

from __future__ import annotations

import argparse
from pathlib import Path

from gn_experiment import language_model_task
from training import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched GPT2-12x512 Muown")
    parser.add_argument("--label", required=True)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=6)
    parser.add_argument("--maximum-epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_trial(
        language_model_task(
            micro_batch_size=arguments.micro_batch_size,
            gradient_accumulation=arguments.gradient_accumulation,
        ),
        "muown",
        root,
        workers=arguments.workers,
        run_label=arguments.label,
        maximum_epochs=arguments.maximum_epochs,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    print(result)


if __name__ == "__main__":
    main()
