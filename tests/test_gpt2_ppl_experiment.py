from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_perplexity_is_the_exponential_of_mean_negative_log_likelihood():
    from gpt2_ppl_experiment import perplexity_from_nll

    assert perplexity_from_nll(0.0) == pytest.approx(1.0)
    assert perplexity_from_nll(3.0) == pytest.approx(20.0855369232)


def test_formal_ppl_hyperparameters_match_the_completed_gpt2_comparison():
    from gpt2_ppl_experiment import FORMAL_PPL_HYPERPARAMETERS

    assert FORMAL_PPL_HYPERPARAMETERS == {
        "adamw": {"learning_rate": 1.5e-4, "weight_decay": 0.01, "auxiliary_lr": None},
        "muon": {"learning_rate": 2.5e-3, "weight_decay": 0.01, "auxiliary_lr": 5.0e-4},
        "muown": {"learning_rate": 5.0e-3, "weight_decay": 0.0, "auxiliary_lr": 3.0e-4},
        "effective_rank_linear": {
            "learning_rate": 1.25e-3,
            "weight_decay": 0.0,
            "auxiliary_lr": 3.0e-4,
        },
        "effective_rank_linear_joint_newton": {
            "learning_rate": 1.25e-3,
            "weight_decay": 0.0,
            "auxiliary_lr": 3.0e-4,
        },
    }


def test_ppl_renderer_writes_step_and_time_figures(tmp_path: Path):
    from gpt2_ppl_experiment import ppl_trial_paths, write_ppl_comparison_plots

    methods = ("adamw", "muon", "muown", "effective_rank_linear")
    for index, method in enumerate(methods):
        paths = ppl_trial_paths(tmp_path, method, run_label="test")
        paths.metric.parent.mkdir(parents=True, exist_ok=True)
        paths.metric.write_text(
            "\n".join(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": epoch * 10,
                        "elapsed_seconds": epoch * (index + 1),
                        "validation_nll": 3.0 - epoch * 0.1,
                        "perplexity": 20.0 - epoch,
                    }
                )
                for epoch in (1, 2)
            )
            + "\n"
        )

    outputs = write_ppl_comparison_plots(tmp_path, label="test", methods=methods)

    assert [path.name for path in outputs] == [
        "gpt2_ppl_test_steps.png",
        "gpt2_ppl_test_time.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
