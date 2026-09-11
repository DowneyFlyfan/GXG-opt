from pathlib import Path

from scripts.run_gpt2_v2_comparison import load_config
from gpt2_v2_experiment import checkpoint_path, comparison_plan


def test_5070ti_smoke_profile_reports_its_actual_hardware_and_batching():
    config = load_config(Path("configs/experiments/gpt2_v2_5070ti_smoke.yaml"))

    plan = comparison_plan(config, blocks=232_000)

    assert plan["gpu"] == "NVIDIA GeForce RTX 5070 Ti"
    assert plan["minimum_gpu_memory_gib"] == 16
    assert plan["physical_batch_size"] == 4
    assert plan["effective_batch_size"] == 32
    assert plan["runs"] == 8


def test_v2_checkpoint_path_is_kept_under_cache_not_results():
    run_output = Path("results/gpt2_v2_5070ti_smoke/runs/adamw/seed_0")

    checkpoint = checkpoint_path(run_output)

    assert checkpoint.name == "checkpoint.pt"
    assert checkpoint.is_relative_to(Path(".cache/gpt2-v2/checkpoints").resolve())
    assert not checkpoint.is_relative_to(Path("results").resolve())
