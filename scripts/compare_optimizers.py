#!/usr/bin/env python3
"""Create the required metric-vs-step PNG with AdamW and Muon baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_metrics(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamw", type=Path, required=True)
    parser.add_argument("--muon", type=Path, required=True)
    parser.add_argument("--kronecker-ggn", type=Path, required=True)
    parser.add_argument("--low-rank-corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ylabel", default="Metric")
    args = parser.parse_args()
    import matplotlib.pyplot as plot

    paths = {
        "AdamW": args.adamw,
        "Muon": args.muon,
        "Kronecker GGN": args.kronecker_ggn,
        "Low-rank-corrected Kronecker GGN": args.low_rank_corrected,
    }
    figure, axis = plot.subplots(figsize=(9, 5))
    for label, path in paths.items():
        records = read_metrics(path)
        if not records:
            raise ValueError(f"No metrics in {path}")
        step_key = "step" if "step" in records[0] else "epoch"
        axis.plot(
            [record[step_key] for record in records],
            [record["metric"] for record in records],
            label=label,
        )
    axis.set(xlabel="Optimizer step", ylabel=args.ylabel)
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plot.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
