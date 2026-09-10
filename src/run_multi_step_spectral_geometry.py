from __future__ import annotations

import argparse
import json
from pathlib import Path

from multi_step_spectral_geometry_experiment import run_multi_step_spectral_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Multi-Step Spectral Geometry on matched GPT2-12x512 NLP"
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--auxiliary-learning-rate", type=float)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--initial-p", type=float, default=float("inf"))
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--policy-interval", type=int, default=100)
    parser.add_argument("--switch-margin", type=float, default=0.005)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--maximum-epochs", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = run_multi_step_spectral_trial(
        Path(__file__).resolve().parents[1],
        label=args.label,
        learning_rate=args.learning_rate,
        auxiliary_learning_rate=args.auxiliary_learning_rate,
        momentum=args.momentum,
        initial_p=args.initial_p,
        warmup_steps=args.warmup_steps,
        policy_interval=args.policy_interval,
        switch_margin=args.switch_margin,
        workers=args.workers,
        micro_batch_size=args.micro_batch_size,
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
