from pathlib import Path


def test_selected_baseline_paths_are_the_matched_batch_winners(tmp_path):
    from gpt_baseline_selection import selected_baseline_paths

    metric_root = Path(tmp_path) / "metrics/nlp"
    result_root = Path(tmp_path) / "results/nlp"
    metric_root.mkdir(parents=True)
    result_root.mkdir(parents=True)
    (metric_root / "nlp_gpt_12x512__literature_adamw00015_full_b12_a4__adamw.jsonl").write_text("{}\n")
    (result_root / "nlp_gpt_12x512__literature_adamw00015_full_b12_a4__adamw.json").write_text("{}\n")

    metric, result = selected_baseline_paths(Path(tmp_path), "adamw")

    assert metric.name == "nlp_gpt_12x512__literature_adamw00015_full_b12_a4__adamw.jsonl"
    assert result.name == "nlp_gpt_12x512__literature_adamw00015_full_b12_a4__adamw.json"


def test_selected_muown_baseline_uses_the_final_zero_decay_run(tmp_path):
    from gpt_baseline_selection import selected_baseline_label, selected_baseline_paths

    metric_root = Path(tmp_path) / "metrics/nlp"
    result_root = Path(tmp_path) / "results/nlp"
    metric_root.mkdir(parents=True)
    result_root.mkdir(parents=True)
    stem = "nlp_gpt_12x512__lr0005_b8_a6_final__muown"
    (metric_root / f"{stem}.jsonl").write_text("{}\n")
    (result_root / f"{stem}.json").write_text("{}\n")

    metric, result = selected_baseline_paths(Path(tmp_path), "muown")

    assert metric.name == f"{stem}.jsonl"
    assert result.name == f"{stem}.json"
    assert selected_baseline_label("muown") == "Muown (5e-3, wd=0)"
