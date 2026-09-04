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
