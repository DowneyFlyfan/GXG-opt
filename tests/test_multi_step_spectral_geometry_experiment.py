import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def test_matched_multi_step_task_and_artifact_paths(tmp_path):
    assert importlib.util.find_spec("multi_step_spectral_geometry_experiment") is not None
    experiment = importlib.import_module("multi_step_spectral_geometry_experiment")

    task = experiment.matched_multi_step_task(gradient_accumulation=4)
    paths = experiment.multi_step_spectral_paths(tmp_path, "probe")

    assert (task.micro_batch_size, task.gradient_accumulation, task.estimated_epochs) == (12, 4, 5)
    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__multi_step_spectral_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__multi_step_spectral_probe.json"
    assert paths.checkpoint == tmp_path / ".cache/nlp/checkpoints/nlp_gpt_12x512__multi_step_spectral_probe.checkpoint.pt"


def test_matched_multi_step_task_can_preserve_effective_batch_with_smaller_microbatch():
    experiment = importlib.import_module("multi_step_spectral_geometry_experiment")

    task = experiment.matched_multi_step_task(
        micro_batch_size=8, gradient_accumulation=6
    )

    assert (task.micro_batch_size, task.gradient_accumulation) == (8, 6)
    assert task.micro_batch_size * task.gradient_accumulation == 48


def test_multi_step_optimizer_routes_matrix_and_auxiliary_parameters():
    experiment = importlib.import_module("multi_step_spectral_geometry_experiment")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.eye(3))
            self.bias = torch.nn.Parameter(torch.zeros(3))

    model = Model()
    optimizer, selected = experiment.build_multi_step_spectral_optimizer(
        model, learning_rate=3.0e-3, auxiliary_learning_rate=1.0e-4,
        weight_decay=0.0, momentum=0.95,
    )

    spectral = next(group for group in optimizer.param_groups if group["use_spectral"])
    auxiliary = next(group for group in optimizer.param_groups if not group["use_spectral"])
    assert selected == {"matrix"}
    assert spectral["params"] == [model.matrix]
    assert auxiliary["params"] == [model.bias]
    assert auxiliary["lr"] == 1.0e-4
    assert optimizer.config.matrix_scale == "moonlight"


def test_multi_step_optimizer_exposes_policy_tuning_controls():
    experiment = importlib.import_module("multi_step_spectral_geometry_experiment")

    model = torch.nn.Linear(3, 3, bias=False)
    optimizer, _ = experiment.build_multi_step_spectral_optimizer(
        model,
        learning_rate=3.0e-4,
        auxiliary_learning_rate=1.0e-4,
        weight_decay=0.0,
        momentum=0.95,
        initial_p=4.0,
        warmup_steps=50,
        policy_interval=25,
        switch_margin=0.0,
    )

    assert optimizer.config.initial_p == 4.0
    assert optimizer.config.warmup_steps == 50
    assert optimizer.config.policy_interval == 25
    assert optimizer.config.switch_margin == 0.0


def test_trial_rejects_invalid_hyperparameters_before_cuda(tmp_path):
    experiment = importlib.import_module("multi_step_spectral_geometry_experiment")
    assert hasattr(experiment, "run_multi_step_spectral_trial")
    with pytest.raises(ValueError, match="hyperparameters"):
        experiment.run_multi_step_spectral_trial(
            tmp_path, label="invalid", learning_rate=0.0
        )


def test_multi_step_cli_exposes_matched_protocol_arguments():
    script = Path("src/run_multi_step_spectral_geometry.py")
    assert script.is_file()
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=True
    )
    assert "--gradient-accumulation" in completed.stdout
    assert "--micro-batch-size" in completed.stdout
    assert "--maximum-epochs" in completed.stdout
    assert "--initial-p" in completed.stdout
    assert "--warmup-steps" in completed.stdout
    assert "--policy-interval" in completed.stdout
    assert "--switch-margin" in completed.stdout


def test_renderer_writes_metric_step_and_time_comparisons(tmp_path, monkeypatch):
    experiment = importlib.import_module("multi_step_spectral_geometry_experiment")
    paths = experiment.multi_step_spectral_paths(tmp_path, "matched")
    paths.metric.parent.mkdir(parents=True)
    paths.metric.write_text(json.dumps({"epoch": 1, "step": 2034, "metric": .4, "elapsed_seconds": 12.}) + "\n")
    baseline = [{"epoch": 1, "step": 2034, "metric": .3, "elapsed_seconds": 10.}]
    monkeypatch.setattr(experiment, "_baseline_records", lambda _root, _name: baseline)
    monkeypatch.setattr(experiment, "selected_baseline_label", lambda name: name)

    outputs = experiment.write_multi_step_spectral_comparison_plots(tmp_path, label="matched")

    assert [path.name for path in outputs] == [
        "multi_step_spectral_matched_metric_steps.png",
        "multi_step_spectral_matched_metric_time.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
