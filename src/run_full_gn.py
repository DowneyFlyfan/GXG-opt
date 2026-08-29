from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_gn_experiment import run_full_gn_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the matrix-free full generalized Gauss--Newton language-model trial"
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--initial-damping", type=float, required=True)
    parser.add_argument("--maximum-cg-iterations", type=int, required=True)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--evaluation-interval-steps", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_full_gn_trial(
        Path(__file__).resolve().parents[1],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        initial_damping=args.initial_damping,
        maximum_cg_iterations=args.maximum_cg_iterations,
        maximum_seconds=args.maximum_seconds,
        evaluation_interval_steps=args.evaluation_interval_steps,
        workers=args.workers,
        fresh=args.fresh,
        label=args.label,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
