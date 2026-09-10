import torch
from torch import nn


def test_muown_keeps_effective_rows_at_the_tracked_gains():
    from optimizers import Muown

    parameter = nn.Parameter(torch.tensor([[3.0, 4.0], [5.0, 12.0]]))
    parameter.grad = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
    optimizer = Muown([parameter], lr=0.01, weight_decay=0.0, ns_steps=1)

    optimizer.step()

    state = optimizer.state[parameter]
    assert torch.allclose(parameter.norm(dim=1), state["gain"], atol=1e-5, rtol=1e-5)
    assert state["direction"].shape == parameter.shape
    assert state["gain_exp_avg"].shape == state["gain"].shape


def test_build_optimizers_exposes_muown_and_keeps_auxiliary_parameters_in_adamw():
    from optimizers import Muown, build_optimizers

    model = nn.Sequential(nn.Linear(4, 4, bias=True), nn.LayerNorm(4))
    optimizers = build_optimizers(
        model, "muown", lr=0.01, weight_decay=0.0, auxiliary_lr=0.001
    )

    assert isinstance(optimizers["muown"], Muown)
    assert isinstance(optimizers["adamw_aux"], torch.optim.AdamW)


def test_muown_gpt_runner_exposes_matched_screen_controls():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "src/run_muown.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
    )

    assert "--learning-rate" in completed.stdout
    assert "--maximum-epochs" in completed.stdout


def test_muown_runner_defaults_to_the_paper_zero_weight_decay(monkeypatch):
    import run_muown
    import sys

    captured = {}

    def fake_run_trial(*args, **kwargs):
        captured.update(kwargs)
        return {"status": "mocked"}

    monkeypatch.setattr(run_muown, "run_trial", fake_run_trial)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_muown.py", "--label", "screen", "--learning-rate", "0.005"],
    )

    run_muown.main()

    assert captured["weight_decay"] == 0.0
