from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plot


def write_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_metrics(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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
        def format_runtime(seconds: float) -> str:
            hours, remainder = divmod(round(seconds), 3600)
            minutes, remaining_seconds = divmod(remainder, 60)
            return f"{hours}h {minutes}m {remaining_seconds:02d}s" if hours else f"{minutes}m {remaining_seconds:02d}s"

        axis.set_title(f"Wall-clock time\nAdamW: {format_runtime(runtimes['AdamW'])} | Muon: {format_runtime(runtimes['Muon'])}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plot.close(figure)
    return output
