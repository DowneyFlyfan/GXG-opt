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


def test_select_best_final_runs_excludes_one_epoch_boundary_probes(tmp_path):
    from lr_tuning_plot import TuningRun, select_best_final_runs

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    for label, optimizer, metric, epochs in (
        ("mu-low", "muon", 0.72, 5),
        ("mu-high", "muon", 0.75, 5),
        ("mu-probe", "muon", 0.99, 1),
        ("adam", "adamw", 0.74, 5),
    ):
        (metrics / f"nlp_gpt_12x512__{label}__{optimizer}.jsonl").write_text(
            "\n".join(json.dumps({"epoch": epoch, "metric": metric}) for epoch in range(1, epochs + 1)) + "\n"
        )
        (results / f"nlp_gpt_12x512__{label}__{optimizer}.json").write_text(
            json.dumps({"seconds": 60.0, "epochs": epochs})
        )

    winners = select_best_final_runs(
        tmp_path,
        (
            TuningRun("Muon lr=0.001", "mu-low", "muon"),
            TuningRun("Muon lr=0.005", "mu-high", "muon"),
            TuningRun("Muon lr=0.01", "mu-probe", "muon"),
            TuningRun("AdamW lr=0.0005", "adam", "adamw"),
        ),
    )

    assert winners["muon"].label == "Muon lr=0.005"
    assert winners["adamw"].label == "AdamW lr=0.0005"


def test_final_baseline_plots_are_not_tuning_sweep_plots(tmp_path):
    from lr_tuning_plot import TuningRun, write_final_baseline_plots

    metrics = tmp_path / "metrics" / "nlp"
    results = tmp_path / "results" / "nlp"
    metrics.mkdir(parents=True)
    results.mkdir(parents=True)
    runs = (
        TuningRun("Muon lr=0.0025", "mu", "muon"),
        TuningRun("AdamW lr=0.00015", "adam", "adamw"),
    )
    for run in runs:
        (metrics / f"nlp_gpt_12x512__{run.run_label}__{run.optimizer}.jsonl").write_text(
            "\n".join(json.dumps({"epoch": epoch, "metric": 0.7 + epoch / 100}) for epoch in range(1, 6)) + "\n"
        )
        (results / f"nlp_gpt_12x512__{run.run_label}__{run.optimizer}.json").write_text(
            json.dumps({"seconds": 120.0, "epochs": 5})
        )

    outputs = write_final_baseline_plots(tmp_path, runs)

    assert outputs is not None
    assert all(path.is_file() for path in outputs)
    assert all("final_baselines" in path.name for path in outputs)
