"""Render selected DINOv3 ImageNet-100 baseline comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from artifacts import write_metric_plot, write_metric_time_plot
from config import DINOV3_IMAGENET100_TASK
from training import trial_artifact_paths


def write_cv_dinov3_baseline_plots(root: Path, label: str) -> tuple[Path, Path]:
    paths = {
        optimizer: trial_artifact_paths(root, DINOV3_IMAGENET100_TASK, optimizer, run_label=label)
        for optimizer in ("adamw", "muon", "muown")
    }
    if not all(path.metric.exists() and path.result.exists() for path in paths.values()):
        raise RuntimeError("DINOv3 comparison requires completed AdamW, Muon, and Muown trials")
    runtimes = {
        "AdamW": json.loads(paths["adamw"].result.read_text())["seconds"],
        "Muon": json.loads(paths["muon"].result.read_text())["seconds"],
        "Muown": json.loads(paths["muown"].result.read_text())["seconds"],
    }
    output_root = root / "results" / "cv"
    return (
        write_metric_plot(
            paths["adamw"].metric,
            paths["muon"].metric,
            output_root / "cv_dinov3_vitb16_imagenet100_baselines_metric_steps.png",
            "Validation top-1 accuracy",
            runtimes,
            paths["muown"].metric,
        ),
        write_metric_time_plot(
            paths["adamw"].metric,
            paths["muon"].metric,
            output_root / "cv_dinov3_vitb16_imagenet100_baselines_metric_time.png",
            "Validation top-1 accuracy",
            runtimes,
            paths["muown"].metric,
        ),
    )
