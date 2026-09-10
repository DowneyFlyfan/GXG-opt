"""Regression coverage for Muown on every dedicated GPT comparison plot."""

from __future__ import annotations

import importlib
import json


def test_each_dedicated_gpt_renderer_requests_muown(tmp_path, monkeypatch):
    renderers = (
        (
            "low_spectral_variance_experiment",
            "low_spectral_variance_paths",
            "write_low_spectral_variance_comparison_plots",
        ),
        (
            "multi_step_spectral_geometry_experiment",
            "multi_step_spectral_paths",
            "write_multi_step_spectral_comparison_plots",
        ),
        ("srip_band_experiment", "srip_band_paths", "write_srip_band_comparison_plots"),
        (
            "stiefel_feature_experiment",
            "stiefel_feature_paths",
            "write_stiefel_feature_comparison_plots",
        ),
    )
    baseline = [{"epoch": 1, "step": 1, "elapsed_seconds": 1.0, "metric": 0.7}]
    for module_name, paths_name, writer_name in renderers:
        module = importlib.import_module(module_name)
        label = f"muown_{module_name}"
        paths = getattr(module, paths_name)(tmp_path, label)
        paths.metric.parent.mkdir(parents=True, exist_ok=True)
        paths.metric.write_text(json.dumps(baseline[0]) + "\n")
        requested: list[str] = []

        def baselines(_root, optimizer):
            requested.append(optimizer)
            return baseline

        monkeypatch.setattr(module, "_baseline_records", baselines)
        monkeypatch.setattr(module, "selected_baseline_label", lambda optimizer: optimizer)

        outputs = getattr(module, writer_name)(tmp_path, label=label)

        assert requested == ["adamw", "muon", "muown"]
        assert outputs is not None
        assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_muown_writer_creates_step_and_time_comparisons(tmp_path):
    from gpt_baseline_selection import SELECTED_GPT_BASELINES
    from muown_experiment import write_muown_comparison_plots

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    for optimizer, (_, label) in SELECTED_GPT_BASELINES.items():
        stem = f"nlp_gpt_12x512__{label}__{optimizer}"
        (metrics / f"{stem}.jsonl").write_text(
            json.dumps({"epoch": 1, "metric": 0.7}) + "\n"
        )
        (results / f"{stem}.json").write_text(
            json.dumps({"gradient_accumulation": 6, "seconds": 60.0, "epochs": 5})
        )

    outputs = write_muown_comparison_plots(tmp_path, label="lr0005_b8_a6_final")

    assert [output.name for output in outputs] == [
        "muown_lr0005_b8_a6_final_metric_steps.png",
        "muown_lr0005_b8_a6_final_metric_time.png",
    ]
    assert all(output.exists() and output.stat().st_size > 0 for output in outputs)
