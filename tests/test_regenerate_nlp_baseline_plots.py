import json


def test_regenerate_forwards_selected_muown_records_to_each_plot(tmp_path, monkeypatch):
    import regenerate_nlp_baseline_plots as redraw
    from gpt_baseline_selection import SELECTED_GPT_BASELINES

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    for optimizer, (_, label) in SELECTED_GPT_BASELINES.items():
        stem = f"nlp_gpt_12x512__{label}__{optimizer}"
        (metrics / f"{stem}.jsonl").write_text(json.dumps({"epoch": 1, "metric": 0.7}) + "\n")
        (results / f"{stem}.json").write_text(
            json.dumps({"gradient_accumulation": 4, "seconds": 60.0, "epochs": 5})
        )
    for suffix in ("metric_steps.png", "metric_time.png"):
        (results / f"candidate_formal_{suffix}").write_bytes(b"png")
    (metrics / "nlp_gpt_12x512__candidate_formal.jsonl").write_text(
        json.dumps({"step": 1, "elapsed_seconds": 1.0, "metric": 0.7}) + "\n"
    )
    observed = []

    def capture_plot(*args, muown_records, **kwargs):
        observed.append(muown_records)

    monkeypatch.setattr(redraw, "_write_plot", capture_plot)

    redraw.regenerate(tmp_path)

    assert observed == [[{"epoch": 1, "metric": 0.7, "step": 3052, "elapsed_seconds": 12.0}]] * 2


def test_candidate_paths_exclude_tuning_only_figures(tmp_path):
    from regenerate_nlp_baseline_plots import _candidate_paths

    results = tmp_path / "results" / "nlp"
    metrics = tmp_path / "metrics" / "nlp"
    results.mkdir(parents=True)
    metrics.mkdir(parents=True)
    for stem in (
        "stiefel_muon_formal",
        "lowrank_secant_equalbatch3904_b2_s01_formal",
        "spectral_sphere_muon_tune_lr001",
    ):
        (results / f"{stem}_metric_steps.png").write_bytes(b"png")
        (metrics / f"nlp_gpt_12x512__{stem}.jsonl").write_text(
            json.dumps({"step": 1, "elapsed_seconds": 1.0, "metric": 0.7}) + "\n"
        )

    candidates = _candidate_paths(tmp_path)

    assert [stem for stem, _ in candidates] == [
        "lowrank_secant_equalbatch3904_b2_s01_formal",
        "stiefel_muon_formal",
    ]
