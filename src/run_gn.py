from __future__ import annotations

import argparse
import json
from pathlib import Path

from gn_experiment import run_gn_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the cached GPT-12x512 Kronecker GN language-model trial"
    )
    parser.add_argument(
        "--optimizer",
        choices=("kronecker_ggn", "low_rank_corrected_kronecker_ggn"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--maximum-seconds", type=float, default=14_400.0)
    parser.add_argument("--evaluation-interval-steps", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--gn-learning-rate", type=float)
    parser.add_argument("--gn-damping", type=float)
    parser.add_argument("--gn-trust-clip", type=float)
    parser.add_argument("--gn-global-trust-clip", type=float)
    parser.add_argument("--factor-update-interval", type=int)
    parser.add_argument("--spectral-update-interval", type=int)
    parser.add_argument("--adaptive-damping", action="store_true")
    parser.add_argument("--damping-adaptation-interval", type=int)
    parser.add_argument("--label", help="Separate artifact suffix for a tuning trial.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an incompatible checkpoint and metric trace before running.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_overrides = {
        key: value
        for key, value in {
            "learning_rate": args.gn_learning_rate,
            "damping": args.gn_damping,
            "trust_clip": args.gn_trust_clip,
            "global_trust_clip": args.gn_global_trust_clip,
            "factor_update_interval": args.factor_update_interval,
            "spectral_update_interval": args.spectral_update_interval,
            "adaptive_damping": True if args.adaptive_damping else None,
            "damping_adaptation_interval": args.damping_adaptation_interval,
        }.items()
        if value is not None
    }
    if args.gn_global_trust_clip is not None:
        config_overrides["trust_clip"] = None
    result = run_gn_trial(
        root,
        args.optimizer,
        workers=args.workers,
        maximum_seconds=args.maximum_seconds,
        evaluation_interval_steps=args.evaluation_interval_steps,
        fresh=args.fresh,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation=args.gradient_accumulation,
        config_overrides=config_overrides,
        run_label=args.label,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
