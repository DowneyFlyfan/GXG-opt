"""Tune or run matched GPT2-12x512 PPL comparisons on one CUDA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gn_experiment import language_model_task
from gpt2_ppl_experiment import (
    DISPLAY_NAMES,
    FORMAL_PPL_HYPERPARAMETERS,
    run_ppl_trial,
    write_ppl_comparison_plots,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--methods", nargs="+", choices=tuple(DISPLAY_NAMES), required=True)
    parser.add_argument("--adamw-learning-rate", type=float, default=FORMAL_PPL_HYPERPARAMETERS["adamw"]["learning_rate"])
    parser.add_argument("--muon-learning-rate", type=float, default=FORMAL_PPL_HYPERPARAMETERS["muon"]["learning_rate"])
    parser.add_argument("--muown-learning-rate", type=float, default=FORMAL_PPL_HYPERPARAMETERS["muown"]["learning_rate"])
    parser.add_argument("--muown-momentum", type=float, default=0.95)
    parser.add_argument("--effective-rank-learning-rate", type=float, default=FORMAL_PPL_HYPERPARAMETERS["effective_rank_linear"]["learning_rate"])
    parser.add_argument("--effective-rank-momentum", type=float, default=0.95)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=6)
    parser.add_argument("--maximum-epochs", type=int, default=5)
    parser.add_argument("--maximum-updates", type=int)
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, help="override the method-specific formal weight decay")
    parser.add_argument("--skip-plots", action="store_true", help="defer rendering until distributed method records are consolidated")
    arguments = parser.parse_args()
    rates = {
        "adamw": arguments.adamw_learning_rate,
        "muon": arguments.muon_learning_rate,
        "muown": arguments.muown_learning_rate,
        "effective_rank_half": arguments.effective_rank_learning_rate,
        "effective_rank_joint_newton": arguments.effective_rank_learning_rate,
        "effective_rank_linear": arguments.effective_rank_learning_rate,
        "effective_rank_linear_joint_newton": arguments.effective_rank_learning_rate,
    }
    root = Path(__file__).resolve().parents[1]
    task = language_model_task(
        micro_batch_size=arguments.micro_batch_size,
        gradient_accumulation=arguments.gradient_accumulation,
    )
    results = []
    for method in arguments.methods:
        result = run_ppl_trial(
            task,
            method,
            root,
            run_label=arguments.label,
            learning_rate=rates[method],
            workers=arguments.workers,
            maximum_epochs=arguments.maximum_epochs,
            maximum_updates=arguments.maximum_updates,
            validation_batches=arguments.validation_batches,
            weight_decay=(
                FORMAL_PPL_HYPERPARAMETERS[method]["weight_decay"]
                if arguments.weight_decay is None
                else arguments.weight_decay
            ),
            auxiliary_learning_rate=FORMAL_PPL_HYPERPARAMETERS[method]["auxiliary_lr"],
            effective_rank_momentum=(
                arguments.effective_rank_momentum
                if method in {"effective_rank_half", "effective_rank_joint_newton"}
                else None
            ),
            muown_momentum=(
                arguments.muown_momentum if method == "muown" else None
            ),
        )
        results.append(result)
    if arguments.maximum_updates is None and not arguments.skip_plots:
        plots = write_ppl_comparison_plots(root, label=arguments.label, methods=arguments.methods)
        print(json.dumps({"results": results, "plots": [str(path) for path in plots]}, sort_keys=True))
    else:
        print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
