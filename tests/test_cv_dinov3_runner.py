import sys


def test_muown_cli_uses_direction_rate_without_a_shared_learning_rate(monkeypatch):
    import run_cv_dinov3

    captured = {}
    monkeypatch.setattr(
        run_cv_dinov3,
        "run_trial",
        lambda *args, **kwargs: captured.update(kwargs) or {"status": "mocked"},
    )
    monkeypatch.setattr(run_cv_dinov3, "write_cv_dinov3_baseline_plots", lambda *args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cv_dinov3.py",
            "--optimizer",
            "muown",
            "--label",
            "screen",
            "--direction-learning-rate",
            "0.002",
            "--gain-learning-rate",
            "0.0002",
        ],
    )

    run_cv_dinov3.main()

    assert captured["learning_rate"] == 0.002
    assert captured["muown_direction_lr"] == 0.002
    assert captured["muown_gain_lr"] == 0.0002
