#!/usr/bin/env python3
"""Render GPT-2 validation NLL against optimizer steps for paired runs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = {
    "adamw": ("AdamW", "#475569"),
    "muon": ("Muon", "#2563eb"),
    "multi_step_spectral_policy": (
        "Multi-step spectral geometry policy",
        "#dc2626",
    ),
}


def read_paired_curves(
    comparison_directory: Path,
) -> tuple[dict[str, dict[int, list[dict]]], list[int]]:
    state = json.loads(
        (comparison_directory / "comparison_state.json").read_text(encoding="utf-8")
    )
    completed: dict[str, dict[int, list[dict]]] = defaultdict(dict)
    for job in state["jobs"]:
        method = str(job["method"])
        if method not in METHODS or job["status"] != "completed":
            continue
        path = Path(job["output_directory"]) / "metrics.jsonl"
        completed[method][int(job["seed"])] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    missing = sorted(set(METHODS) - set(completed))
    if missing:
        raise ValueError(f"no completed runs for methods: {missing}")
    paired_seeds = sorted(
        set.intersection(*(set(runs) for runs in completed.values()))
    )
    if not paired_seeds:
        raise ValueError("the methods have no commonly completed seed")
    return dict(completed), paired_seeds


def aggregate_curves(
    curves: dict[str, dict[int, list[dict]]],
    paired_seeds: list[int],
    *,
    x_key: str = "step",
) -> dict[str, list[tuple[float, float, float, float]]]:
    aggregates = {}
    for method, runs in curves.items():
        by_step: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for seed in paired_seeds:
            for metric in runs[seed]:
                by_step[int(metric["step"])].append(
                    (float(metric[x_key]), float(metric["validation_nll"]))
                )
        aggregates[method] = [
            (
                float(step) if x_key == "step" else statistics.fmean(x for x, _ in values),
                statistics.fmean(value for _, value in values),
                min(value for _, value in values),
                max(value for _, value in values),
            )
            for step, values in sorted(by_step.items())
            if len(values) == len(paired_seeds)
        ]
    return aggregates


def render(
    comparison_directory: Path, output: Path, *, x_key: str = "step"
) -> Path:
    import matplotlib.pyplot as plot

    if x_key not in {"step", "elapsed_wall_seconds"}:
        raise ValueError(f"unsupported x-axis metric: {x_key}")
    curves, paired_seeds = read_paired_curves(comparison_directory)
    aggregates = aggregate_curves(curves, paired_seeds, x_key=x_key)
    figure, axis = plot.subplots(figsize=(9, 5.5))
    for method, (label, color) in METHODS.items():
        for seed in paired_seeds:
            metrics = curves[method][seed]
            axis.plot(
                [metric[x_key] for metric in metrics],
                [metric["validation_nll"] for metric in metrics],
                color=color,
                alpha=0.18,
                linewidth=1,
            )
        points = aggregates[method]
        x_values = [point[0] for point in points]
        means = [point[1] for point in points]
        lows = [point[2] for point in points]
        highs = [point[3] for point in points]
        axis.fill_between(x_values, lows, highs, color=color, alpha=0.1)
        axis.plot(x_values, means, color=color, linewidth=2.5, label=label)
    axis.set(
        title=(
            "GPT-2 / WikiText-103 optimizer comparison "
            f"(paired seeds: {', '.join(map(str, paired_seeds))})"
        ),
        xlabel=(
            "Optimizer step"
            if x_key == "step"
            else "Cumulative wall-clock time (seconds)"
        ),
        ylabel="Validation negative log-likelihood (lower is better)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plot.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_directory", type=Path)
    parser.add_argument("--steps-output", type=Path)
    parser.add_argument("--time-output", type=Path)
    args = parser.parse_args()
    steps_output = args.steps_output or (
        args.comparison_directory / "gpt2_wikitext103_validation_nll_steps.png"
    )
    time_output = args.time_output or (
        args.comparison_directory / "gpt2_wikitext103_validation_nll_time.png"
    )
    print(render(args.comparison_directory, steps_output))
    print(
        render(
            args.comparison_directory,
            time_output,
            x_key="elapsed_wall_seconds",
        )
    )


if __name__ == "__main__":
    main()
