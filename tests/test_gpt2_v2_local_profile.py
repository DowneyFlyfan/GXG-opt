from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.run_gpt2_v2_comparison import load_config
from gpt2_v2_experiment import checkpoint_path, comparison_plan, require_configured_gpu
from optimizer_v2.adapter import ProposalAdapter


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


def test_5070ti_profile_allows_its_configured_runtime_gpu(monkeypatch):
    config = load_config(Path("configs/experiments/gpt2_v2_5070ti_smoke.yaml"))
    monkeypatch.setattr(
        "gpt2_v2_experiment.torch.cuda.get_device_name", lambda index: "NVIDIA GeForce RTX 5070 Ti"
    )

    require_configured_gpu(config)


def test_local_torch_constructs_optimizer_v2_adapter_with_stock_muon():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=19, n_embd=8, n_layer=1, n_head=2))
    adapter = ProposalAdapter(model)

    assert any(isinstance(optimizer, torch.optim.Muon) for optimizer in adapter.optimizers)
