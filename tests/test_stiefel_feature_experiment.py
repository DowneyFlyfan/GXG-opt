import importlib.util
from pathlib import Path

import importlib
import pytest


def test_stiefel_feature_matched_paths_and_workload(tmp_path):
    assert importlib.util.find_spec("stiefel_feature_experiment") is not None
    from stiefel_feature_experiment import stiefel_feature_paths, stiefel_feature_task

    task = stiefel_feature_task(gradient_accumulation=4)
    paths = stiefel_feature_paths(tmp_path, "probe")

    assert (task.micro_batch_size, task.gradient_accumulation, task.estimated_epochs) == (12, 4, 5)
    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__stiefel_feature_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__stiefel_feature_probe.json"


def test_stiefel_feature_trial_rejects_invalid_alpha_before_cuda(tmp_path):
    experiment = importlib.import_module("stiefel_feature_experiment")
    assert hasattr(experiment, "run_stiefel_feature_trial")
    with pytest.raises(ValueError, match="alpha"):
        experiment.run_stiefel_feature_trial(tmp_path, label="invalid", alpha=0.0)


def test_stiefel_feature_cli_exists():
    assert Path("src/run_stiefel_feature.py").is_file()
