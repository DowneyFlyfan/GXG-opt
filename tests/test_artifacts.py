from pathlib import Path

import artifacts
import torch
from torch import nn
from artifacts import write_metric, write_metric_plot
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

    assert titles == ["Wall-clock time — AdamW: 2m 00s | Muon: 2m 10s"]


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
