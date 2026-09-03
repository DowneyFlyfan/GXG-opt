from pathlib import Path

import pytest


def test_hybrid_artifacts_stay_in_nlp_metrics_results_and_cache(tmp_path: Path):
    from hybrid_stiefel_muon_experiment import hybrid_stiefel_muon_paths

    paths = hybrid_stiefel_muon_paths(tmp_path, "probe")

    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__hybrid_stiefel_muon_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__hybrid_stiefel_muon_probe.json"
    assert paths.checkpoint == (
        tmp_path
        / ".cache/nlp/checkpoints/nlp_gpt_12x512__hybrid_stiefel_muon_probe.checkpoint.pt"
    )


def test_hybrid_rejects_nonpositive_cosine_schedule_horizon(tmp_path: Path):
    from hybrid_stiefel_muon_experiment import run_hybrid_stiefel_muon_trial

    with pytest.raises(ValueError, match="scheduler_t_max"):
        run_hybrid_stiefel_muon_trial(
            tmp_path,
            label="invalid",
            muon_learning_rate=0.0006,
            stiefel_learning_rate=0.003,
            scheduler_t_max=0,
        )
