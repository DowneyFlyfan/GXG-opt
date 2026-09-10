from __future__ import annotations

import argparse
import json
from pathlib import Path

from stiefel_muon_experiment import run_stiefel_muon_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stiefel-Muon on GPT-12x512")
    parser.add_argument("--label", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--evaluation-interval-steps", type=int, default=1_024)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_stiefel_muon_trial(
        Path(__file__).resolve().parents[1],
        label=args.label,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        ns_steps=args.ns_steps,
        workers=args.workers,
        maximum_seconds=args.maximum_seconds,
        maximum_steps=args.maximum_steps,
        evaluation_interval_steps=args.evaluation_interval_steps,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        seed=args.seed,
        fresh=args.fresh,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
