import json


def test_summary_reports_completed_and_incomplete_candidates(tmp_path):
    from finalize_literature_tuning import write_literature_tuning_summary

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    metric = metrics / "nlp_gpt_12x512__literature_mu0025_adamw0005_b12_a4__muon.jsonl"
    result = results / "nlp_gpt_12x512__literature_mu0025_adamw0005_b12_a4__muon.json"
    metric.write_text(json.dumps({"epoch": 5, "metric": 0.75}) + "\n")
    result.write_text(json.dumps({"epochs": 5, "seconds": 120.0, "peak_memory_mb": 1024.0}))

    output = write_literature_tuning_summary(tmp_path)

    text = output.read_text()
    assert "Muon lr=0.0025" in text
    assert "| muon | 0.0025 | 5 |" in text
    assert "0.750000" in text
    assert "Muon lr=0.02" in text
