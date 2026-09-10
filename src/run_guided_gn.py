from __future__ import annotations

import argparse
import json
from pathlib import Path

from guided_gn_experiment import DEFAULT_GUIDED_BLOCK_PATTERNS, run_guided_gn_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled generalized Gauss--Newton-guided AdamW baseline"
    )
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation", type=int, required=True)
    parser.add_argument("--curvature-accumulation", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--guided-block-pattern", action="append")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--refresh-interval", type=int, default=200)
    parser.add_argument("--initial-damping", type=float, default=1.0e-2)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--trust-radius", type=float, default=1.0)
    parser.add_argument("--max-relative-block-update", type=float, default=1.0e-3)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--max-basis-age", type=int)
    parser.add_argument("--max-parameter-drift", type=float, default=1.0e-2)
    parser.add_argument("--rho-min", type=float, default=0.0)
    parser.add_argument("--acceptance-margin", type=float, default=0.0)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--evaluation-interval-steps", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_guided_gn_trial(
        Path(__file__).resolve().parents[1],
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        curvature_accumulation=args.curvature_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        guided_block_patterns=tuple(
            args.guided_block_pattern or DEFAULT_GUIDED_BLOCK_PATTERNS
        ),
        rank=args.rank,
        refresh_interval=args.refresh_interval,
        initial_damping=args.initial_damping,
        warmup_steps=args.warmup_steps,
        trust_radius=args.trust_radius,
        max_relative_block_update=args.max_relative_block_update,
        alpha_max=args.alpha_max,
        max_basis_age=args.max_basis_age,
        max_parameter_drift=args.max_parameter_drift,
        rho_min=args.rho_min,
        acceptance_margin=args.acceptance_margin,
        maximum_seconds=args.maximum_seconds,
        maximum_steps=args.maximum_steps,
        evaluation_interval_steps=args.evaluation_interval_steps,
        workers=args.workers,
        seed=args.seed,
        fresh=args.fresh,
        label=args.label,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
