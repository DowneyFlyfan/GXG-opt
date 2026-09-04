from __future__ import annotations

import json
import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(
    Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
)

import matplotlib.pyplot as plot


def write_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _format_runtime(seconds: float) -> str:
    hours, remainder = divmod(round(seconds), 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {remaining_seconds:02d}s" if hours else f"{minutes}m {remaining_seconds:02d}s"


def write_metric_plot(
    adamw_path: Path,
    muon_path: Path,
    output: Path,
    ylabel: str,
    runtimes: dict[str, float] | None = None,
) -> Path:
    adamw, muon = _read_metrics(adamw_path), _read_metrics(muon_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plot.subplots(figsize=(8, 5))
    axis.plot([item["epoch"] for item in adamw], [item["metric"] for item in adamw], label="AdamW")
    axis.plot([item["epoch"] for item in muon], [item["metric"] for item in muon], label="Muon")
    axis.set(xlabel="Epoch", ylabel=ylabel)
    if runtimes is not None:
        axis.set_title(
            f"Wall-clock time\nAdamW: {_format_runtime(runtimes['AdamW'])} | "
            f"Muon: {_format_runtime(runtimes['Muon'])}"
        )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plot.close(figure)
    return output


def write_metric_time_plot(
    adamw_path: Path,
    muon_path: Path,
    output: Path,
    ylabel: str,
    runtimes: dict[str, float],
) -> Path:
    """Plot epoch-end metrics against each run's measured wall-clock duration.

    Metrics are evaluated at epoch boundaries, so their time coordinates are
    linearly placed over the measured total duration of the corresponding run.
    """
    adamw, muon = _read_metrics(adamw_path), _read_metrics(muon_path)

    def metric_times(records: list[dict], seconds: float) -> list[float]:
        total_epochs = max(record["epoch"] for record in records)
        return [seconds * record["epoch"] / total_epochs / 60 for record in records]

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plot.subplots(figsize=(8, 5))
    axis.plot(metric_times(adamw, runtimes["AdamW"]), [item["metric"] for item in adamw], label="AdamW")
    axis.plot(metric_times(muon, runtimes["Muon"]), [item["metric"] for item in muon], label="Muon")
    axis.set(
        xlabel="Wall-clock time (minutes)",
        ylabel=ylabel,
        title=f"AdamW: {_format_runtime(runtimes['AdamW'])} | Muon: {_format_runtime(runtimes['Muon'])}",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plot.close(figure)
    return output
