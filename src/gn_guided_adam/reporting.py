from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plot


def _records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def plot_metric_comparison(
    adamw_jsonl: str | Path,
    guided_jsonl: str | Path,
    output: str | Path,
    *,
    metric_key: str = "validation_metric",
    x_key: str = "wall_time_seconds",
    x_label: str | None = None,
) -> Path:
    adamw, guided = _records(adamw_jsonl), _records(guided_jsonl)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plot.subplots(figsize=(8, 5))
    axis.plot([row[x_key] for row in adamw], [row[metric_key] for row in adamw], label="AdamW")
    axis.plot([row[x_key] for row in guided], [row[metric_key] for row in guided], label="GN-guided AdamW")
    axis.set(xlabel=x_label or x_key, ylabel=metric_key)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plot.close(figure)
    return destination


def plot_metric_time_comparison(
    adamw_jsonl: str | Path,
    guided_jsonl: str | Path,
    output: str | Path,
    *,
    metric_key: str = "validation_metric",
) -> Path:
    return plot_metric_comparison(
        adamw_jsonl,
        guided_jsonl,
        output,
        metric_key=metric_key,
        x_key="wall_time_seconds",
        x_label="Wall time (seconds)",
    )


def plot_metric_steps_comparison(
    adamw_jsonl: str | Path,
    guided_jsonl: str | Path,
    output: str | Path,
    *,
    metric_key: str = "validation_metric",
) -> Path:
    return plot_metric_comparison(
        adamw_jsonl,
        guided_jsonl,
        output,
        metric_key=metric_key,
        x_key="step",
        x_label="Optimizer step",
    )


def plot_metric_tokens_comparison(
    adamw_jsonl: str | Path,
    guided_jsonl: str | Path,
    output: str | Path,
    *,
    metric_key: str = "validation_metric",
) -> Path:
    return plot_metric_comparison(
        adamw_jsonl,
        guided_jsonl,
        output,
        metric_key=metric_key,
        x_key="tokens",
        x_label="Tokens",
    )
