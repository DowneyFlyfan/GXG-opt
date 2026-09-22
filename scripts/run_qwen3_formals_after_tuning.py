"""Select completed screen winners and run the matched five-epoch baselines."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


FORMAL_LABEL = "scratch_16mep_5ep_tuned_batch64_v4_mb8a8"
GPU0_UUID = "GPU-d5c984dc-a988-2e9a-219f-df1577167aaf"
MAX_TRAINING_SECONDS = 7190
FORMAL_MAXIMUM_EPOCHS = 5
FORMAL_MAXIMUM_UPDATES = 610
SCREEN_LABELS = {
    "adamw": [
        "scratch_16mep_adamw_tune16m_v1_s300_mb8a8_lr0.001",
        "scratch_16mep_adamw_tune16m_v1_s300_mb8a8_lr0.003",
        "scratch_16mep_adamw_tune16m_v1_s300_mb8a8_lr0.006",
    ],
    "muon": [
        "scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.003_aux0.001",
        "scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.003_aux0.003",
    ],
    "muown": [
        "scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.01_gain0.0003_aux0.001",
        "scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.01_gain0.0003_aux0.003",
        "scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.01_gain0.0003_aux0.01",
    ],
}


def result_path(root: Path, optimizer: str, label: str) -> Path:
    return root / "results" / "nlp" / f"qwen3_0p6b__{label}__{optimizer}.ppl.json"


def checkpoint_path(root: Path, optimizer: str, label: str) -> Path:
    return (
        root / ".cache" / "qwen3_0p6b" / "checkpoints"
        / f"qwen3_0p6b__{label}__{optimizer}.checkpoint.pt"
    )


def load_winners(root: Path) -> dict[str, dict]:
    winners: dict[str, dict] = {}
    for optimizer, labels in SCREEN_LABELS.items():
        candidates: list[dict] = []
        for label in labels:
            path = result_path(root, optimizer, label)
            if not path.is_file():
                raise FileNotFoundError(f"missing screen result: {path}")
            payload = json.loads(path.read_text())
            if not (
                payload.get("optimizer") == optimizer
                and payload.get("initialization") == "scratch"
                and payload.get("completed_updates") == 300
                and isinstance(payload.get("final_perplexity"), (int, float))
            ):
                raise RuntimeError(f"invalid screen result: {path}")
            candidates.append(payload)
        winners[optimizer] = min(candidates, key=lambda payload: payload["final_perplexity"])
    manifests = {payload["data_manifest_sha256"] for payload in winners.values()}
    if len(manifests) != 1:
        raise RuntimeError("screen winners do not share the same data manifest")
    return winners


def wait_for_unowned_gpu0() -> None:
    while True:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        if GPU0_UUID not in completed.stdout:
            return
        time.sleep(30)


def formal_command(root: Path, optimizer: str, winner: dict) -> list[str]:
    config = winner["config"]
    command = [
        str(root / ".venv/bin/python"), "-u", str(root / "src/run_qwen3_ppl.py"), "run",
        "--initialization", "scratch", "--optimizer", optimizer, "--run-label", FORMAL_LABEL,
        "--data-directory", str(root / ".cache/Fineweb_Edu_2B"),
        "--micro-batch-size", "8", "--gradient-accumulation", "8", "--maximum-epochs", str(FORMAL_MAXIMUM_EPOCHS),
        "--maximum-updates", str(FORMAL_MAXIMUM_UPDATES),
        "--train-tokens-per-epoch", "16000000", "--validation-batches", "12",
        "--evaluation-interval-updates", "100", "--checkpoint-interval-updates", "610",
        "--warmup-updates", "100", "--schedule-updates", "610", "--minimum-lr-ratio", "0.1",
        "--gradient-clip", "1", "--weight-decay", "0.1", "--seed", "1337",
    ]
    if optimizer in {"adamw", "muon"}:
        command.extend(["--learning-rate", str(config["learning_rate"])])
    if optimizer in {"muon", "muown"}:
        command.extend(["--auxiliary-lr", str(config["auxiliary_lr"])])
    if optimizer == "muown":
        command.extend([
            "--direction-lr", str(config["direction_lr"]),
            "--gain-lr", str(config["gain_lr"]),
        ])
    return command


def validate_formal_result(optimizer: str, result: dict) -> None:
    """Validate the fixed five-epoch update budget, independent of epoch indexing."""
    config = result.get("config", {})
    if not (
        result.get("optimizer") == optimizer
        and result.get("initialization") == "scratch"
        and config.get("maximum_epochs") == FORMAL_MAXIMUM_EPOCHS
        and config.get("maximum_updates") == FORMAL_MAXIMUM_UPDATES
        and result.get("completed_updates") == FORMAL_MAXIMUM_UPDATES
    ):
        raise RuntimeError(f"incomplete formal run: {optimizer}")


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    winners = load_winners(root)
    record_path = root / "records/2026-09-15_qwen3_16m_final_selection.json"
    record_path.write_text(json.dumps({
        "status": "formal_runs_started",
        "formal_run_label": FORMAL_LABEL,
        "selection_rule": "minimum completed 300-update held-out perplexity within each optimizer",
        "winners": {
            optimizer: {
                "screen_run_label": result["run_label"],
                "final_perplexity": result["final_perplexity"],
                "rate_config": {
                    key: result["config"].get(key)
                    for key in ("learning_rate", "direction_lr", "gain_lr", "auxiliary_lr")
                },
            }
            for optimizer, result in winners.items()
        },
    }, indent=2) + "\n")
    removed_checkpoints: dict[str, int] = {}
    for optimizer in ("adamw", "muon", "muown"):
        path = result_path(root, optimizer, FORMAL_LABEL)
        if path.is_file():
            result = json.loads(path.read_text())
        else:
            wait_for_unowned_gpu0()
            try:
                subprocess.run(
                    formal_command(root, optimizer, winners[optimizer]),
                    cwd=root,
                    check=True,
                    timeout=MAX_TRAINING_SECONDS,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"two-hour training limit reached before completion: {optimizer}"
                ) from error
            result = json.loads(path.read_text())
        validate_formal_result(optimizer, result)
        if float(result.get("elapsed_seconds", float("inf"))) >= 7200:
            raise RuntimeError(f"two-hour limit exceeded: {optimizer}")
        subprocess.run([
            str(root / ".venv/bin/python"), "-u", str(root / "src/run_qwen3_ppl.py"),
            "evaluate-checkpoint", "--optimizer", optimizer, "--run-label", FORMAL_LABEL,
        ], cwd=root, check=True)
        checkpoint = checkpoint_path(root, optimizer, FORMAL_LABEL)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"full validation completed without final checkpoint: {checkpoint}")
        removed_checkpoints[optimizer] = checkpoint.stat().st_size
        checkpoint.unlink()
    subprocess.run([
        str(root / ".venv/bin/python"), "-u", str(root / "src/run_qwen3_ppl.py"),
        "render", "--run-label", FORMAL_LABEL,
    ], cwd=root, check=True)
    completed = {
        optimizer: json.loads(result_path(root, optimizer, FORMAL_LABEL).read_text())
        for optimizer in ("adamw", "muon", "muown")
    }
    record_path.write_text(json.dumps({
        "status": "formal_runs_completed",
        "formal_run_label": FORMAL_LABEL,
        "selection_rule": "minimum completed 300-update held-out perplexity within each optimizer",
        "winners": {optimizer: winners[optimizer]["run_label"] for optimizer in winners},
        "formal_results": {
            optimizer: {
                "final_perplexity": result["final_perplexity"],
                "elapsed_seconds": result["elapsed_seconds"],
                "completed_epochs": result["completed_epochs"],
                "completed_updates": result["completed_updates"],
            }
            for optimizer, result in completed.items()
        },
        "checkpoints_removed_after_full_validation_bytes": removed_checkpoints,
    }, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
