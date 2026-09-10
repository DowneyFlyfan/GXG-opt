"""Comparison plots for the selected Muown GPT2-12x512 baseline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plot

from gpt_baseline_selection import selected_baseline_label
from low_spectral_variance_experiment import _baseline_records


def write_muown_comparison_plots(root: Path, *, label: str) -> tuple[Path, Path]:
    """Write the final Muown curves against the selected GPT baselines."""
    traces = [
        (selected_baseline_label("adamw"), _baseline_records(root, "adamw")),
        (selected_baseline_label("muon"), _baseline_records(root, "muon")),
        (selected_baseline_label("muown"), _baseline_records(root, "muown")),
    ]
    if not all(records for _, records in traces):
        raise RuntimeError("Muown comparison requires all selected GPT baselines")
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"muown_{label}_metric_steps.png",
        output_root / f"muown_{label}_metric_time.png",
    )
    for output, key, xlabel in (
        (outputs[0], "step", "Completed optimizer step"),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)"),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for name, records in traces:
            axis.plot(
                [record[key] / 3600 if key == "elapsed_seconds" else record[key] for record in records],
                [record["metric"] for record in records],
                label=name,
            )
        axis.set(
            xlabel=xlabel,
            ylabel="Validation next-token accuracy",
            title="Muown versus selected GPT-2 baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs
