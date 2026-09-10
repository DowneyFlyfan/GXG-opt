from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_full_gn_experiment import (
    prepare_common_adamw_warmup,
    run_paper_full_gn_trial,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the Muon-inner Full Gauss-Newton paper algorithm"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    warmup = subparsers.add_parser("warmup")
    warmup.add_argument("--maximum-seconds", type=float, default=14_400.0)
    warmup.add_argument("--workers", type=int, default=4)
    warmup.add_argument("--fresh", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--micro-batch-size", type=int, default=1)
    train.add_argument("--sequence-length", type=int, default=1024)
    train.add_argument("--inner-steps", type=int, default=122)
    train.add_argument("--inner-gradient-accumulation", type=int, default=1)
    train.add_argument("--inner-learning-rate", type=float, default=0.01)
    train.add_argument("--optimizer-weight-decay", type=float, default=0.001)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--line-search-range", type=int, default=5)
    train.add_argument("--maximum-outer-steps", type=int, default=10_000)
    train.add_argument("--maximum-seconds", type=float, default=14_400.0)
    train.add_argument("--evaluation-interval-steps", type=int, default=1)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--label", required=True)
    train.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.command == "warmup":
        result = prepare_common_adamw_warmup(
            root,
            maximum_seconds=args.maximum_seconds,
            workers=args.workers,
            fresh=args.fresh,
        )
    else:
        result = run_paper_full_gn_trial(
            root,
            micro_batch_size=args.micro_batch_size,
            sequence_length=args.sequence_length,
            inner_steps=args.inner_steps,
            inner_gradient_accumulation=args.inner_gradient_accumulation,
            inner_learning_rate=args.inner_learning_rate,
            optimizer_weight_decay=args.optimizer_weight_decay,
            gradient_clip=args.gradient_clip,
            line_search_range=args.line_search_range,
            maximum_outer_steps=args.maximum_outer_steps,
            maximum_seconds=args.maximum_seconds,
            evaluation_interval_steps=args.evaluation_interval_steps,
            workers=args.workers,
            label=args.label,
            fresh=args.fresh,
        )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
