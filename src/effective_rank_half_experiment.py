"""Comparison plots for the Effective-Rank-Half GPT2-12x512 optimizer."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plot

from gn_experiment import _read_metrics
from gpt_baseline_selection import selected_baseline_label
from low_spectral_variance_experiment import _baseline_records


def _candidate_records(root: Path, label: str) -> list[dict]:
    stem = f"nlp_gpt_12x512__{label}__effective_rank_half"
    metric_path = root / "metrics" / "nlp" / f"{stem}.jsonl"
    result_path = root / "results" / "nlp" / f"{stem}.json"
    records = _read_metrics(metric_path) if metric_path.exists() else []
    if not records or {"step", "elapsed_seconds"} <= records[0].keys():
        return records
    if not result_path.exists():
        return []
    result = json.loads(result_path.read_text())
    epochs = int(result["epochs"])
    steps_per_epoch = math.ceil(12_207 / int(result["gradient_accumulation"]))
    return [
        {
            **record,
            "step": int(record["epoch"]) * steps_per_epoch,
            "elapsed_seconds": float(result["seconds"]) * int(record["epoch"]) / epochs,
        }
        for record in records
    ]


def write_effective_rank_half_comparison_plots(
    root: Path, *, label: str
) -> tuple[Path, Path] | None:
    """Write the final candidate against the selected GPT baseline traces."""
    traces = [
        (selected_baseline_label("adamw"), _baseline_records(root, "adamw")),
        (selected_baseline_label("muon"), _baseline_records(root, "muon")),
        (selected_baseline_label("muown"), _baseline_records(root, "muown")),
        ("Effective-Rank-Half", _candidate_records(root, label)),
    ]
    if not all(records for _, records in traces):
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"effective_rank_half_{label}_metric_steps.png",
        output_root / f"effective_rank_half_{label}_metric_time.png",
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
            title="Effective-Rank-Half versus tuned GPT-2 baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs
