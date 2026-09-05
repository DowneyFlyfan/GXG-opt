"""Write a reproducible summary after the labelled GPT learning-rate sweep."""

from __future__ import annotations

import json
from pathlib import Path

from lr_tuning_plot import LITERATURE_TUNING_RUNS, _paths, select_best_tuning_runs, write_literature_tuning_plots


def write_literature_tuning_summary(root: Path) -> Path:
    """Record every candidate that completed, including its final accuracy."""
    rows: list[tuple[str, str, int, float, float, float]] = []
    incomplete: list[str] = []
    for run in LITERATURE_TUNING_RUNS:
        metric_path, result_path = _paths(root, run)
        if not metric_path.is_file() or not result_path.is_file():
            incomplete.append(run.label)
            continue
        metrics = [json.loads(line) for line in metric_path.read_text().splitlines() if line]
        if not metrics:
            incomplete.append(run.label)
            continue
        result = json.loads(result_path.read_text())
        rows.append(
            (
                run.label,
                run.optimizer,
                int(result["epochs"]),
                float(metrics[-1]["metric"]),
                float(result["seconds"]),
                float(result["peak_memory_mb"]),
            )
        )
    winners = select_best_tuning_runs(root)
    plots = write_literature_tuning_plots(root)
    lines = [
        "# GPT-12x512 learning-rate tuning",
        "",
        "## Fixed protocol",
        "",
        "- Model: GPT-12x512 (54.68M parameters); WikiText byte stream cached under `.cache`.",
        "- Full candidates use five epochs; high-end boundary probes may stop after one epoch. All use micro-batch 12, gradient accumulation 4, effective batch 48, and weight decay 0.01.",
        "- Metric: validation next-token accuracy. No loss curves are used.",
        "- Each listed candidate ran alone on the local GPU.",
        "",
        "## Completed candidates",
        "",
        "| Optimizer | Learning rate | Epochs | Final validation accuracy | Seconds | Peak MiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, optimizer, epochs, metric, seconds, memory in rows:
        lines.append(f"| {optimizer} | {label.rsplit('=', 1)[1].strip()} | {epochs} | {metric:.6f} | {seconds:.1f} | {memory:.1f} |")
    lines.extend(["", "## Selected settings", ""])
    for optimizer in ("muon", "adamw"):
        winner = winners.get(optimizer)
        if winner is not None:
            lines.append(f"- {optimizer}: {winner.label}, final validation accuracy {winner.records[-1]['metric']:.6f}.")
    if incomplete:
        lines.extend(["", "## Incomplete candidates", "", *[f"- {label}" for label in incomplete]])
    if plots is not None:
        lines.extend(["", "## Figures", "", *[f"- `{path.relative_to(root)}`" for path in plots]])
    output = root / "records" / "2026-09-04-literature-lr-tuning.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


if __name__ == "__main__":
    print(write_literature_tuning_summary(Path(__file__).resolve().parents[1]))
