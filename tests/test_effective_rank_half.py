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


def test_joint_newton_step_returns_a_certified_active_constraint_update():
    """A smooth active case converges without falling back to bisection."""
    from effective_rank_half import (
        effective_rank,
        joint_newton_effective_rank_step,
    )

    weight = torch.tensor(
        [
            [0.0667522043586033, -0.6291102570899281, -0.2722289854045090],
            [-0.5852130630390915, 0.1897551684060146, 0.0467166644690034],
            [0.1524841993186392, 0.2608752343528235, -0.9306904363832025],
        ],
        dtype=torch.float64,
    )
    gradient = torch.tensor(
        [
            [-1.0973077042250232, 0.5250752320866116, 0.7969625170665243],
            [0.1443745847855657, -0.7399100063415882, -0.3533482254419504],
            [-1.5493114901666452, 0.8235255343228248, 0.1169950046842461],
        ],
        dtype=torch.float64,
    )

    update = joint_newton_effective_rank_step(
        weight, gradient, step_size=0.7384175083695312, minimum_effective_rank=0.5
    )

    assert update.accepted
    assert update.solver == "joint_newton"
    assert update.multiplier > 0.0
    assert torch.linalg.matrix_norm(update.direction, ord=2) <= 1.0 + 1.0e-10
    assert effective_rank(update.weight) >= 0.5 - 1.0e-10


def test_joint_newton_float32_active_constraint_does_not_fail_the_polar_preflight():
    """The float32 GPT optimizer path must not reject a converged active solve."""
    from effective_rank_half import (
        effective_rank,
        joint_newton_effective_rank_step,
    )

    weight = torch.tensor(
        [
            [0.0667522043586033, -0.6291102570899281, -0.2722289854045090],
            [-0.5852130630390915, 0.1897551684060146, 0.0467166644690034],
            [0.1524841993186392, 0.2608752343528235, -0.9306904363832025],
        ],
        dtype=torch.float32,
    )
    gradient = torch.tensor(
        [
            [-1.0973077042250232, 0.5250752320866116, 0.7969625170665243],
            [0.1443745847855657, -0.7399100063415882, -0.3533482254419504],
            [-1.5493114901666452, 0.8235255343228248, 0.1169950046842461],
        ],
        dtype=torch.float32,
    )

    update = joint_newton_effective_rank_step(
        weight, gradient, step_size=0.7384175083695312, minimum_effective_rank=0.5
    )

    assert update.solver == "joint_newton"
    assert effective_rank(update.weight) >= 0.5 - 1.0e-6


def test_joint_newton_falls_back_for_a_rank_deficient_polar_derivative():
    """A nonsmooth gradient must use the existing finite-step certificate."""
    from effective_rank_half import joint_newton_effective_rank_step

    small_singular_value_squared = 2.0 - math.sqrt(3.0)
    weight = torch.diag(
        torch.tensor([1.0, math.sqrt(small_singular_value_squared), 0.0], dtype=torch.float64)
    )
    gradient = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))

    update = joint_newton_effective_rank_step(
        weight, gradient, step_size=0.55, minimum_effective_rank=0.5
    )

    assert update.accepted
    assert update.solver == "certified_fallback"
    assert update.effective_rank >= 0.5 - 1.0e-10


def test_joint_newton_solves_the_equivalent_transposed_wide_problem():
    from effective_rank_half import joint_newton_effective_rank_step

    weight = torch.tensor(
        [
            [0.0667522043586033, -0.6291102570899281, -0.2722289854045090],
            [-0.5852130630390915, 0.1897551684060146, 0.0467166644690034],
            [0.1524841993186392, 0.2608752343528235, -0.9306904363832025],
        ],
        dtype=torch.float64,
    ).transpose(0, 1)
    gradient = torch.tensor(
        [
            [-1.0973077042250232, 0.5250752320866116, 0.7969625170665243],
            [0.1443745847855657, -0.7399100063415882, -0.3533482254419504],
            [-1.5493114901666452, 0.8235255343228248, 0.1169950046842461],
        ],
        dtype=torch.float64,
    ).transpose(0, 1)

    update = joint_newton_effective_rank_step(
        weight, gradient, step_size=0.7384175083695312, minimum_effective_rank=0.5
    )

    assert update.solver == "joint_newton"
    assert update.weight.shape == weight.shape


def test_certified_step_admits_and_preserves_the_one_third_constraint():
    from effective_rank_half import certified_effective_rank_step, effective_rank

    squared_small_singular_value = 5.0 - math.sqrt(24.0)
    weight = torch.diag(
        torch.tensor([1.0, math.sqrt(squared_small_singular_value), 0.0], dtype=torch.float64)
    )
    gradient = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))

    with pytest.raises(ValueError, match="effective-rank-half"):
        certified_effective_rank_step(weight, gradient, step_size=0.1)
    update = certified_effective_rank_step(
        weight, gradient, step_size=0.1, minimum_effective_rank=1.0 / 3.0
    )

    assert effective_rank(weight) == pytest.approx(0.4, abs=1.0e-10)
    assert update.accepted
    assert effective_rank(update.weight) >= 1.0 / 3.0 - 1.0e-10


def test_one_third_optimizer_and_builder_keep_matrix_updates_certified():
    from effective_rank_half import EffectiveRankThird, effective_rank
    from optimizers import build_optimizers

    parameter = torch.nn.Parameter(torch.eye(3))
    parameter.grad = torch.tensor([[0.2, -0.3, 0.1], [0.1, 0.4, -0.2], [0.0, 0.3, 0.2]])
    optimizer = EffectiveRankThird([parameter], lr=0.1, weight_decay=0.0)
    optimizer.step()

    assert effective_rank(parameter) >= 1.0 / 3.0 - 1.0e-6
    assert optimizer.state[parameter]["accepted_steps"] == 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.eye(3))
            self.bias = torch.nn.Parameter(torch.zeros(3))

    optimizers = build_optimizers(
        Model(), "effective_rank_third", lr=0.01, weight_decay=0.0, auxiliary_lr=0.001
    )

    assert isinstance(optimizers["effective_rank_third"], EffectiveRankThird)
    assert isinstance(optimizers["adamw_aux"], torch.optim.AdamW)


def test_linear_optimizer_advances_and_restores_its_effective_rank_schedule():
    from effective_rank_half import EffectiveRankLinear, effective_rank
    from optimizers import build_optimizers

    parameter = torch.nn.Parameter(torch.eye(3))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = EffectiveRankLinear(
        [parameter], lr=0.1, weight_decay=0.0, schedule_steps=4
    )
    optimizer.step()

    assert optimizer.minimum_effective_rank == pytest.approx(0.4)
    assert effective_rank(parameter) >= 0.2 - 1.0e-6
    restored_parameter = torch.nn.Parameter(torch.eye(3))
    restored = EffectiveRankLinear(
        [restored_parameter], lr=0.1, weight_decay=0.0, schedule_steps=4
    )
    restored.load_state_dict(optimizer.state_dict())
    assert restored.schedule_step == 1
    assert restored.minimum_effective_rank == pytest.approx(0.4)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.eye(3))
            self.bias = torch.nn.Parameter(torch.zeros(3))

    optimizers = build_optimizers(
        Model(), "effective_rank_linear", lr=0.01, weight_decay=0.0, auxiliary_lr=0.001
    )
    assert isinstance(optimizers["effective_rank_linear"], EffectiveRankLinear)


def test_linear_joint_newton_optimizer_preserves_schedule_and_reports_solver_path():
    from effective_rank_half import EffectiveRankLinearJointNewton, effective_rank
    from optimizers import build_optimizers

    parameter = torch.nn.Parameter(torch.eye(3, dtype=torch.float64))
    parameter.grad = torch.tensor(
        [[0.2, -0.3, 0.1], [0.1, 0.4, -0.2], [0.3, 0.3, 0.2]], dtype=torch.float64
    )
    optimizer = EffectiveRankLinearJointNewton(
        [parameter], lr=0.1, weight_decay=0.0, schedule_steps=4
    )
    optimizer.step()

    state = optimizer.state[parameter]
    assert optimizer.schedule_step == 1
    assert effective_rank(parameter) >= 0.2 - 1.0e-10
    assert sum(
        int(state.get(key, 0))
        for key in ("joint_newton_steps", "unconstrained_steps", "certified_fallback_steps")
    ) == 1

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.eye(3))
            self.bias = torch.nn.Parameter(torch.zeros(3))

    optimizers = build_optimizers(
        Model(), "effective_rank_linear_joint_newton", lr=0.01, weight_decay=0.0,
        auxiliary_lr=0.001,
    )
    assert isinstance(
        optimizers["effective_rank_linear_joint_newton"], EffectiveRankLinearJointNewton
    )


def test_linear_optimizer_projects_an_infeasible_matrix_before_certifying_update():
    from effective_rank_half import EffectiveRankLinear, effective_rank

    parameter = torch.nn.Parameter(torch.diag(torch.tensor([1.0, 0.01, 0.0])))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = EffectiveRankLinear(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        schedule_steps=2,
        start_effective_rank=0.8,
        end_effective_rank=0.8,
    )
    optimizer.step()

    assert effective_rank(parameter) >= 0.8 - 1.0e-6
    assert optimizer.state[parameter]["projected_steps"] == 1


def test_linear_optimizer_uses_polar_fast_projection_for_full_rank_matrix():
    """A full-rank scheduled recovery must avoid the singular-value fallback.

    Removing the Newton--Schulz recovery path (or routing every recovery through
    the exact singular-value decomposition) would leave the rank floor intact,
    but would make the GPT2 experiment unnecessarily slow.  The state counters
    make that performance-critical branch observable without mocking linalg.
    """
    from effective_rank_half import EffectiveRankLinear, effective_rank

    parameter = torch.nn.Parameter(torch.diag(torch.tensor([1.0, 0.5, 0.2])))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = EffectiveRankLinear(
        [parameter],
        lr=0.01,
        weight_decay=0.0,
        schedule_steps=2,
        start_effective_rank=0.8,
        end_effective_rank=0.8,
    )
    initial_frobenius = torch.linalg.vector_norm(parameter).item()
    optimizer.step()

    state = optimizer.state[parameter]
    assert effective_rank(parameter) >= 0.8 - 1.0e-6
    assert torch.linalg.vector_norm(parameter).item() == pytest.approx(initial_frobenius)
    assert state["fast_projection_steps"] == 1
    assert state.get("projection_fallback_steps", 0) == 0

    from gpt2_ppl_experiment import _optimizer_diagnostics

    diagnostics = _optimizer_diagnostics(
        "effective_rank_linear", {"effective_rank_linear": optimizer}
    )
    assert diagnostics["effective_rank_fast_projection_steps"] == 1
    assert diagnostics["effective_rank_projection_fallback_steps"] == 0


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


def test_one_third_gpt_runner_exposes_matched_screen_controls():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "src/run_effective_rank_third.py"), "--help"],
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


def test_one_third_final_renderer_writes_step_and_time_comparisons(tmp_path, monkeypatch):
    import effective_rank_third_experiment as experiment

    metric = (
        tmp_path
        / "metrics"
        / "nlp"
        / "nlp_gpt_12x512__lr000125_b8_a6_screen__effective_rank_third.jsonl"
    )
    metric.parent.mkdir(parents=True)
    metric.write_text(
        json.dumps({"epoch": 1, "step": 2034, "metric": 0.75, "elapsed_seconds": 12.0})
        + "\n"
    )
    baseline = [{"epoch": 1, "step": 2034, "metric": 0.70, "elapsed_seconds": 10.0}]
    monkeypatch.setattr(experiment, "_baseline_records", lambda _root, _optimizer: baseline)
    monkeypatch.setattr(experiment, "selected_baseline_label", lambda optimizer: optimizer)

    outputs = experiment.write_effective_rank_third_comparison_plots(
        tmp_path, label="lr000125_b8_a6_screen"
    )

    assert [path.name for path in outputs] == [
        "effective_rank_third_lr000125_b8_a6_screen_metric_steps.png",
        "effective_rank_third_lr000125_b8_a6_screen_metric_time.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
