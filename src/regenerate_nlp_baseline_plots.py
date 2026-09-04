"""Regenerate historical GPT comparison plots after an AdamW/Muon retune.

The candidate metrics are immutable experiment records.  This utility only
refreshes the two common baseline traces from their current canonical files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import plot


TASK_IDENTIFIER = "nlp_gpt_12x512"


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _baseline_records(root: Path, optimizer: str) -> list[dict]:
    metric_path = root / "metrics" / "nlp" / f"{TASK_IDENTIFIER}__{optimizer}.jsonl"
    result_path = root / "results" / "nlp" / f"{TASK_IDENTIFIER}__{optimizer}.json"
    records = _records(metric_path)
    result = json.loads(result_path.read_text())
    # WikiText-103 train stream size used by the retained task.  Metrics are
    # collected at epoch end, so their historical comparison coordinates are
    # epoch proportion × measured total duration.
    steps_per_epoch = (12_207 + int(result["gradient_accumulation"]) - 1) // int(
        result["gradient_accumulation"]
    )
    return [
        {
            **record,
            "step": int(record["epoch"]) * steps_per_epoch,
            "elapsed_seconds": float(result["seconds"])
            * int(record["epoch"])
            / int(result["epochs"]),
        }
        for record in records
    ]


def _candidate_paths(root: Path) -> list[tuple[str, Path]]:
    results = root / "results" / "nlp"
    metrics = root / "metrics" / "nlp"
    candidates = []
    for output in sorted(results.glob("*_metric_steps.png")):
        stem = output.name.removesuffix("_metric_steps.png")
        metric = metrics / f"{TASK_IDENTIFIER}__{stem}.jsonl"
        if metric.exists():
            candidates.append((stem, metric))
    return candidates


def _display_name(stem: str) -> str:
    return stem.replace("_", " ").replace("rpcg", "RPCG").replace("gn", "GN")


def _write_plot(
    output: Path,
    *,
    candidate_name: str,
    candidate_records: list[dict],
    adamw_records: list[dict],
    muon_records: list[dict],
    coordinate: str,
) -> None:
    x_label = "Optimizer step" if coordinate == "step" else "Wall-clock time (hours)"
    figure, axis = plot.subplots(figsize=(9, 5))
    for label, records in (
        ("AdamW (6e-4)", adamw_records),
        ("Muon (6e-4)", muon_records),
        (_display_name(candidate_name), candidate_records),
    ):
        x_values = [
            float(record[coordinate])
            if coordinate == "step"
            else float(record[coordinate]) / 3600
            for record in records
        ]
        axis.plot(x_values, [float(record["metric"]) for record in records], label=label)
    if coordinate == "step" and candidate_name.startswith(("kron_rpcg", "lowrank")):
        axis.set_xscale("log")
    axis.set(
        xlabel=x_label,
        ylabel="Validation next-token accuracy",
        title=f"{_display_name(candidate_name)} versus retuned AdamW and Muon baselines",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plot.close(figure)


def regenerate(root: Path, *, dry_run: bool = False) -> list[Path]:
    """Rewrite each historical NLP candidate plot with current baseline traces."""
    adamw_records = _baseline_records(root, "adamw")
    muon_records = _baseline_records(root, "muon")
    outputs: list[Path] = []
    for stem, candidate_path in _candidate_paths(root):
        candidate_records = _records(candidate_path)
        if not candidate_records or not {"step", "elapsed_seconds"} <= candidate_records[0].keys():
            continue
        for suffix, coordinate in (("metric_steps.png", "step"), ("metric_time.png", "elapsed_seconds")):
            output = root / "results" / "nlp" / f"{stem}_{suffix}"
            if not output.exists():
                continue
            if not dry_run:
                _write_plot(
                    output,
                    candidate_name=stem,
                    candidate_records=candidate_records,
                    adamw_records=adamw_records,
                    muon_records=muon_records,
                    coordinate=coordinate,
                )
            outputs.append(output)
    aggregate_outputs = (
        root / "results" / "nlp" / f"{TASK_IDENTIFIER}_gn_metric_steps.png",
        root / "results" / "nlp" / f"{TASK_IDENTIFIER}_gn_metric_time.png",
    )
    if all(output.exists() for output in aggregate_outputs):
        if not dry_run:
            from gn_experiment import write_gn_comparison_plots

            written = write_gn_comparison_plots(root)
            if written is None:
                raise RuntimeError("Missing traces for aggregate Gauss--Newton plot")
        outputs.extend(aggregate_outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    outputs = regenerate(arguments.root, dry_run=arguments.dry_run)
    print(f"{'would regenerate' if arguments.dry_run else 'regenerated'} {len(outputs)} plots")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
