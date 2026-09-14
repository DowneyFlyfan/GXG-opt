from pathlib import Path

import artifacts
import torch
from torch import nn
from artifacts import write_metric, write_metric_plot, write_metric_time_plot
from training import _load_trial_checkpoint, _save_trial_checkpoint


def test_metric_plot_writes_png_with_epoch_metric_data(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    write_metric(metrics, {"epoch": 1, "metric": 0.3})
    write_metric(metrics, {"epoch": 2, "metric": 0.4})
    comparison = tmp_path / "comparison.jsonl"
    write_metric(comparison, {"epoch": 1, "metric": 0.35})
    write_metric(comparison, {"epoch": 2, "metric": 0.45})

    output = write_metric_plot(metrics, comparison, tmp_path / "curve.png", "validation accuracy")

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_metric_plot_titles_each_optimizer_wall_clock_duration(tmp_path: Path, monkeypatch):
    adamw = tmp_path / "adamw.jsonl"
    muon = tmp_path / "muon.jsonl"
    write_metric(adamw, {"epoch": 1, "metric": 0.3})
    write_metric(muon, {"epoch": 1, "metric": 0.4})
    titles = []
    monkeypatch.setattr(artifacts.plot, "close", lambda figure: titles.append(figure.axes[0].get_title()))

    write_metric_plot(
        adamw,
        muon,
        tmp_path / "curve.png",
        "validation accuracy",
        {"AdamW": 120.0, "Muon": 130.0},
    )

    assert titles == ["Wall-clock time\nAdamW: 2m 00s | Muon: 2m 10s"]


def test_metric_time_plot_writes_png(tmp_path: Path):
    adamw = tmp_path / "adamw.jsonl"
    muon = tmp_path / "muon.jsonl"
    for epoch, metric in ((1, 0.3), (2, 0.4)):
        write_metric(adamw, {"epoch": epoch, "metric": metric})
        write_metric(muon, {"epoch": epoch, "metric": metric + 0.05})

    output = write_metric_time_plot(
        adamw,
        muon,
        tmp_path / "time.png",
        "validation accuracy",
        {"AdamW": 120.0, "Muon": 130.0},
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_three_way_plots_identify_the_coordinate_and_all_runtime_values(tmp_path: Path, monkeypatch):
    adamw = tmp_path / "adamw.jsonl"
    muon = tmp_path / "muon.jsonl"
    muown = tmp_path / "muown.jsonl"
    for path, metric in ((adamw, 0.3), (muon, 0.4), (muown, 0.5)):
        write_metric(path, {"epoch": 1, "step": 2, "elapsed_seconds": 30.0, "metric": metric})
    titles = []
    monkeypatch.setattr(artifacts.plot, "close", lambda figure: titles.append(figure.axes[0].get_title()))
    runtimes = {"AdamW": 120.0, "Muon": 130.0, "Muown": 140.0}

    write_metric_plot(adamw, muon, tmp_path / "steps.png", "validation accuracy", runtimes, muown)
    write_metric_time_plot(adamw, muon, tmp_path / "time.png", "validation accuracy", runtimes, muown)

    assert titles == [
        "validation accuracy vs completed optimizer step\nWall-clock time\n"
        "AdamW: 2m 00s | Muon: 2m 10s | Muown: 2m 20s",
        "validation accuracy vs wall-clock time\n"
        "AdamW: 2m 00s | Muon: 2m 10s | Muown: 2m 20s",
    ]


def test_trial_checkpoint_restores_epoch_elapsed_time_and_optimizer_state(tmp_path: Path):
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = model.weight.detach().clone()
    checkpoint = tmp_path / "trial.checkpoint.pt"

    _save_trial_checkpoint(checkpoint, model, {"adamw": optimizer}, {"adamw": scheduler}, 3, 12.5)

    restored_model = nn.Linear(2, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=75)
    completed_epoch, elapsed_seconds = _load_trial_checkpoint(
        checkpoint,
        restored_model,
        {"adamw": restored_optimizer},
        {"adamw": restored_scheduler},
    )

    assert completed_epoch == 3
    assert elapsed_seconds == 12.5
    assert torch.equal(restored_model.weight, expected)
    assert restored_optimizer.state_dict()["state"]
