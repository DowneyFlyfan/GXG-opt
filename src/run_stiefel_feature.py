from __future__ import annotations

import argparse
import json
from pathlib import Path

from stiefel_feature_experiment import run_stiefel_feature_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stiefel Feature-Steepest Descent on matched GPT2-12x512"
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--auxiliary-learning-rate", type=float)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--maximum-epochs", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = run_stiefel_feature_trial(
        Path(__file__).resolve().parents[1],
        label=args.label,
        alpha=args.alpha,
        auxiliary_learning_rate=args.auxiliary_learning_rate,
        workers=args.workers,
        gradient_accumulation=args.gradient_accumulation,
        maximum_seconds=args.maximum_seconds,
        maximum_epochs=args.maximum_epochs,
        seed=args.seed,
        fresh=args.fresh,
        write_plots=args.final,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
