from __future__ import annotations

import argparse
import json
from pathlib import Path

from new_curvature_experiment import run_mkor_trial, run_nystrom_ggn_trial, run_racs_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled GPT curvature approximation trials")
    parser.add_argument("--candidate", choices=("mkor", "racs", "nystrom_ggn"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--matrix-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--auxiliary-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--racs-scale", type=float, default=0.05)
    parser.add_argument("--maximum-epochs", type=int)
    parser.add_argument("--maximum-outer-steps", type=int, default=8)
    parser.add_argument("--initial-step-scale", type=float, default=1.0)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.candidate == "nystrom_ggn":
        result = run_nystrom_ggn_trial(root, label=args.label, maximum_seconds=args.maximum_seconds, maximum_outer_steps=args.maximum_outer_steps, initial_step_scale=args.initial_step_scale, workers=args.workers, seed=args.seed, fresh=args.fresh)
    else:
        runner = run_mkor_trial if args.candidate == "mkor" else run_racs_trial
        result = runner(root, label=args.label, matrix_learning_rate=args.matrix_learning_rate, auxiliary_learning_rate=args.auxiliary_learning_rate, maximum_seconds=args.maximum_seconds, workers=args.workers, seed=args.seed, fresh=args.fresh, racs_scale=args.racs_scale, maximum_epochs=args.maximum_epochs, write_plots=args.final)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
