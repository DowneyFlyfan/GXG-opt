"""Aggregate labelled learning-rate sweeps into metric comparison figures."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plot


TASK = "nlp_gpt_12x512"


@dataclass(frozen=True)
class TuningRun:
    label: str
    run_label: str
    optimizer: str


@dataclass(frozen=True)
class TuningTrace:
    label: str
    records: tuple[dict, ...]
    seconds: float


LITERATURE_TUNING_RUNS = (
    TuningRun("Muon lr=0.00125", "literature_mu00125_adamw0005_b12_a4", "muon"),
    TuningRun("Muon lr=0.0025", "literature_mu0025_adamw0005_b12_a4", "muon"),
    TuningRun("Muon lr=0.0035", "literature_mu0035_adamw0005_b12_a4", "muon"),
    TuningRun("Muon lr=0.005", "literature_mu005_adamw0005_b12_a4", "muon"),
    TuningRun("Muon lr=0.01", "literature_mu01_adamw0005_b12_a4", "muon"),
    TuningRun("Muon lr=0.02", "literature_mu02_adamw0005_b12_a4", "muon"),
    TuningRun("AdamW lr=0.0003", "literature_adamw0003_b12_a4", "adamw"),
    TuningRun("AdamW lr=0.0003 (full)", "literature_adamw0003_full_b12_a4", "adamw"),
    TuningRun("AdamW lr=0.00015", "literature_adamw00015_b12_a4", "adamw"),
    TuningRun("AdamW lr=0.00015 (full)", "literature_adamw00015_full_b12_a4", "adamw"),
    TuningRun("AdamW lr=0.000075", "literature_adamw000075_b12_a4", "adamw"),
    TuningRun("AdamW lr=0.0005", "literature_adamw0005_b12_a4", "adamw"),
    TuningRun("AdamW lr=0.0008", "literature_adamw0008_b12_a4", "adamw"),
)


def _paths(root: Path, run: TuningRun) -> tuple[Path, Path]:
    stem = f"{TASK}__{run.run_label}__{run.optimizer}"
    return (
        root / "metrics" / "nlp" / f"{stem}.jsonl",
        root / "results" / "nlp" / f"{stem}.json",
    )


def collect_tuning_traces(root: Path, runs: tuple[TuningRun, ...]) -> tuple[TuningTrace, ...]:
    """Return only completed, labelled tuning traces with their measured runtime."""
    traces = []
    for run in runs:
        metric_path, result_path = _paths(root, run)
        if not metric_path.is_file() or not result_path.is_file():
            continue
        records = tuple(
            json.loads(line) for line in metric_path.read_text().splitlines() if line
        )
        if not records:
            continue
        seconds = float(json.loads(result_path.read_text())["seconds"])
        traces.append(TuningTrace(run.label, records, seconds))
    return tuple(traces)


def select_best_tuning_runs(
    root: Path, runs: tuple[TuningRun, ...] = LITERATURE_TUNING_RUNS
) -> dict[str, TuningTrace]:
    """Select the largest final validation accuracy separately per optimizer."""
    traces_by_key = {
        (run.label, run.optimizer): trace
        for run, trace in zip(
            (run for run in runs if _paths(root, run)[0].is_file() and _paths(root, run)[1].is_file()),
            collect_tuning_traces(root, runs),
            strict=True,
        )
    }
    winners: dict[str, TuningTrace] = {}
    for run in runs:
        trace = traces_by_key.get((run.label, run.optimizer))
        if trace is None:
            continue
        previous = winners.get(run.optimizer)
        if previous is None or trace.records[-1]["metric"] > previous.records[-1]["metric"]:
            winners[run.optimizer] = trace
    return winners


def write_literature_tuning_plots(root: Path) -> tuple[Path, Path] | None:
    traces = collect_tuning_traces(root, LITERATURE_TUNING_RUNS)
    if not traces:
        return None
    output_root = root / "results" / "nlp"
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (
        output_root / f"{TASK}_literature_lr_tuning_metric_steps.png",
        output_root / f"{TASK}_literature_lr_tuning_metric_time.png",
    )
    for output, x_label, x_values in (
        (
            outputs[0],
            "Epoch",
            lambda trace: [float(record["epoch"]) for record in trace.records],
        ),
        (
            outputs[1],
            "Wall-clock time (hours)",
            lambda trace: [
                trace.seconds * float(record["epoch"]) / len(trace.records) / 3600
                for record in trace.records
            ],
        ),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for trace in traces:
            axis.plot(x_values(trace), [record["metric"] for record in trace.records], label=trace.label)
        axis.set(xlabel=x_label, ylabel="Validation next-token accuracy")
        axis.set_title("GPT-12x512 learning-rate tuning")
        axis.grid(alpha=0.2)
        axis.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    print(write_literature_tuning_plots(project_root))
