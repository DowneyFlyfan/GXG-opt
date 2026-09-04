import json


def test_literature_tuning_plot_collects_labelled_metric_and_runtime_data(tmp_path):
    from lr_tuning_plot import TuningRun, collect_tuning_traces

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    (metrics / "nlp_gpt_12x512__probe__muon.jsonl").write_text(
        json.dumps({"epoch": 1, "metric": 0.70}) + "\n"
    )
    (results / "nlp_gpt_12x512__probe__muon.json").write_text(
        json.dumps({"seconds": 120.0})
    )

    traces = collect_tuning_traces(
        tmp_path,
        (TuningRun("Muon lr=0.005", "probe", "muon"),),
    )

    assert len(traces) == 1
    assert traces[0].label == "Muon lr=0.005"
    assert traces[0].records == ({"epoch": 1, "metric": 0.70},)
    assert traces[0].seconds == 120.0


def test_select_best_tuning_runs_uses_final_validation_metric(tmp_path):
    from lr_tuning_plot import TuningRun, select_best_tuning_runs

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    for label, optimizer, metric in (("mu-low", "muon", 0.72), ("mu-high", "muon", 0.75), ("adam", "adamw", 0.74)):
        (metrics / f"nlp_gpt_12x512__{label}__{optimizer}.jsonl").write_text(
            json.dumps({"epoch": 1, "metric": metric}) + "\n"
        )
        (results / f"nlp_gpt_12x512__{label}__{optimizer}.json").write_text(
            json.dumps({"seconds": 60.0})
        )

    winners = select_best_tuning_runs(
        tmp_path,
        (
            TuningRun("Muon lr=0.001", "mu-low", "muon"),
            TuningRun("Muon lr=0.005", "mu-high", "muon"),
            TuningRun("AdamW lr=0.0005", "adam", "adamw"),
        ),
    )

    assert winners["muon"].label == "Muon lr=0.005"
    assert winners["adamw"].label == "AdamW lr=0.0005"
