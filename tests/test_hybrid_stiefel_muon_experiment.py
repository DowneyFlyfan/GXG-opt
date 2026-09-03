from pathlib import Path


def test_hybrid_artifacts_stay_in_nlp_metrics_results_and_cache(tmp_path: Path):
    from hybrid_stiefel_muon_experiment import hybrid_stiefel_muon_paths

    paths = hybrid_stiefel_muon_paths(tmp_path, "probe")

    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__hybrid_stiefel_muon_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__hybrid_stiefel_muon_probe.json"
    assert paths.checkpoint == (
        tmp_path
        / ".cache/nlp/checkpoints/nlp_gpt_12x512__hybrid_stiefel_muon_probe.checkpoint.pt"
    )
