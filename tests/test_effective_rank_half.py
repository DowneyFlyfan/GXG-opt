from __future__ import annotations

import math
import json

import pytest
import torch


def test_certified_step_preserves_effective_rank_and_spectral_bound():
    from effective_rank_half import certified_effective_rank_step, effective_rank

    weight = torch.eye(3, dtype=torch.float64)
    gradient = torch.tensor(
        [[0.4, -0.2, 0.1], [-0.3, 0.2, 0.5], [0.2, -0.1, 0.3]], dtype=torch.float64
    )

    update = certified_effective_rank_step(weight, gradient, step_size=0.5)

    assert update.accepted
    assert update.direction.shape == weight.shape
    assert torch.linalg.matrix_norm(update.direction, ord=2) <= 1.0 + 1.0e-10
    assert effective_rank(update.weight) >= 0.5 - 1.0e-10


def test_certified_step_backtracks_at_the_effective_rank_boundary():
    from effective_rank_half import certified_effective_rank_step, effective_rank

    small_singular_value_squared = 2.0 - math.sqrt(3.0)
    weight = torch.diag(torch.tensor([1.0, math.sqrt(small_singular_value_squared), 0.0], dtype=torch.float64))
    gradient = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))

    update = certified_effective_rank_step(weight, gradient, step_size=0.55)

    assert effective_rank(weight) == pytest.approx(0.5, abs=1.0e-10)
    assert update.accepted
    assert 0.0 < update.scale < 1.0
    assert effective_rank(update.weight) >= 0.5 - 1.0e-10


def test_float32_boundary_weight_decay_keeps_certified_optimizer_running():
    """Float32 scale-only decay must not reject a feasible boundary matrix."""
    from effective_rank_half import EffectiveRankHalf, effective_rank

    boundary_singular_value = math.sqrt(2.0 - math.sqrt(3.0))
    parameter = torch.nn.Parameter(
        torch.diag(torch.tensor([1.0, boundary_singular_value, 0.0], dtype=torch.float32))
    )
    parameter.grad = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32))
    optimizer = EffectiveRankHalf([parameter], lr=0.00125, weight_decay=0.01)

    optimizer.step()

    assert optimizer.state[parameter]["accepted_steps"] == 1
    assert effective_rank(parameter) >= 0.5 - 1.0e-6


def test_optimizer_and_builder_keep_matrix_updates_certified():
    from effective_rank_half import EffectiveRankHalf, effective_rank
    from optimizers import build_optimizers

    parameter = torch.nn.Parameter(torch.eye(3))
    parameter.grad = torch.tensor([[0.2, -0.3, 0.1], [0.1, 0.4, -0.2], [0.0, 0.3, 0.2]])
    optimizer = EffectiveRankHalf([parameter], lr=0.1, weight_decay=0.0)
    optimizer.step()

    assert effective_rank(parameter) >= 0.5 - 1.0e-6
    assert optimizer.state[parameter]["accepted_steps"] == 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.eye(3))
            self.bias = torch.nn.Parameter(torch.zeros(3))

    optimizers = build_optimizers(
        Model(), "effective_rank_half", lr=0.01, weight_decay=0.0, auxiliary_lr=0.001
    )

    assert isinstance(optimizers["effective_rank_half"], EffectiveRankHalf)
    assert isinstance(optimizers["adamw_aux"], torch.optim.AdamW)


def test_gpt_runner_exposes_matched_screen_controls():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "src/run_effective_rank_half.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )

    assert "--learning-rate" in completed.stdout
    assert "--maximum-epochs" in completed.stdout


def test_final_renderer_writes_step_and_time_comparisons(tmp_path, monkeypatch):
    import effective_rank_half_experiment as experiment

    metric = (
        tmp_path
        / "metrics"
        / "nlp"
        / "nlp_gpt_12x512__lr000125_b8_a6_final__effective_rank_half.jsonl"
    )
    metric.parent.mkdir(parents=True)
    metric.write_text(
        json.dumps({"epoch": 1, "step": 2034, "metric": 0.75, "elapsed_seconds": 12.0})
        + "\n"
    )
    baseline = [{"epoch": 1, "step": 2034, "metric": 0.70, "elapsed_seconds": 10.0}]
    monkeypatch.setattr(experiment, "_baseline_records", lambda _root, _optimizer: baseline)
    monkeypatch.setattr(experiment, "selected_baseline_label", lambda optimizer: optimizer)

    outputs = experiment.write_effective_rank_half_comparison_plots(
        tmp_path, label="lr000125_b8_a6_final"
    )

    assert [path.name for path in outputs] == [
        "effective_rank_half_lr000125_b8_a6_final_metric_steps.png",
        "effective_rank_half_lr000125_b8_a6_final_metric_time.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
