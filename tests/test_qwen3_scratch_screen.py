from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from run_qwen3_ppl import parse_args


def test_scratch_screen_exposes_the_per_epoch_training_token_budget():
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_qwen3_scratch_screen.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--train-tokens-per-epoch" in completed.stdout
    assert "--checkpoint-interval-updates" in completed.stdout


def test_qwen_runner_allows_curve_evaluation_without_periodic_checkpoints():
    arguments = parse_args(
        [
            "run",
            "--optimizer", "adamw",
            "--run-label", "formal",
            "--learning-rate", "0.001",
            "--evaluation-interval-updates", "100",
            "--checkpoint-interval-updates", "610",
        ]
    )

    assert arguments.evaluation_interval_updates == 100
    assert arguments.checkpoint_interval_updates == 610
