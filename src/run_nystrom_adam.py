from __future__ import annotations

import argparse
import json
from pathlib import Path

from nystrom_adam_experiment import run_nystrom_adam_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Nyström-GGN first stage followed by fresh AdamW on GPT-2 12x512"
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--nystrom-initial-step-scale", type=float, default=1.5e-5)
    parser.add_argument("--nystrom-outer-steps", type=int, default=1)
    parser.add_argument("--adamw-learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_nystrom_adam_trial(
        Path(__file__).resolve().parents[1],
        label=args.label,
        nystrom_initial_step_scale=args.nystrom_initial_step_scale,
        nystrom_outer_steps=args.nystrom_outer_steps,
        adamw_learning_rate=args.adamw_learning_rate,
        maximum_seconds=args.maximum_seconds,
        workers=args.workers,
        seed=args.seed,
        fresh=args.fresh,
        write_plots=args.final,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
