from __future__ import annotations

import argparse
import json
from pathlib import Path

from spectral_unit_ball_gn_experiment import run_spectral_unit_ball_screen


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen spectral-unit-ball full GGN on GPT2-12x512")
    parser.add_argument("--label", required=True)
    parser.add_argument("--maximum-steps", type=int, default=1)
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--initial-beta", type=float, default=1.0)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--auxiliary-learning-rate", type=float, default=0.0)
    parser.add_argument("--auxiliary-weight-decay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_spectral_unit_ball_screen(
        Path(__file__).resolve().parents[1],
        label=args.label,
        maximum_steps=args.maximum_steps,
        inner_iterations=args.inner_iterations,
        initial_beta=args.initial_beta,
        micro_batch_size=args.micro_batch_size,
        validation_batches=args.validation_batches,
        auxiliary_learning_rate=args.auxiliary_learning_rate,
        auxiliary_weight_decay=args.auxiliary_weight_decay,
        workers=args.workers,
        fresh=args.fresh,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
