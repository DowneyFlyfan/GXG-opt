import json
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.render_gpt2_spectral_comparison import (
    METHODS,
    aggregate_curves,
    read_paired_curves,
)
from scripts.run_gpt2_spectral_comparison import (
    REQUIRED_METHODS,
    build_optimizer,
    causal_lm_loss,
    comparison_plan,
    deterministic_batch,
    load_config,
    optimizer_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "gpt2_wikitext103_spectral.yaml"


class TinyTokenBlocks:
    def __len__(self):
        return 20

    def __getitem__(self, indices):
        return {"input_ids": [[index, index + 1, index + 2] for index in indices]}


def test_config_is_fixed_step_three_method_5090_comparison():
    config = load_config(CONFIG_PATH)
    plan = comparison_plan(config)

    assert plan["methods"] == list(REQUIRED_METHODS)
    assert plan["runs"] == 9
    assert plan["gpu_count"] == 1
    assert plan["micro_batch_size"] == 16
    assert plan["effective_batch_size"] == 32
    assert plan["input_tokens_per_run"] == 32_768_000
    assert plan["time_limit_hours_per_run"] == 4.0


def test_batches_are_paired_by_seed_and_step():
    dataset = TinyTokenBlocks()
    first = deterministic_batch(dataset, seed=2, step=17, batch_size=4)
    second = deterministic_batch(dataset, seed=2, step=17, batch_size=4)
    different = deterministic_batch(dataset, seed=2, step=18, batch_size=4)

    torch.testing.assert_close(first, second)
    assert not torch.equal(first, different)


def test_causal_lm_loss_uses_next_token_targets():
    labels = torch.tensor([[0, 1, 2]])
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 1] = 4.0
    logits[0, 1, 2] = 4.0
    expected = torch.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 4), labels[:, 1:].reshape(-1)
    )

    torch.testing.assert_close(causal_lm_loss(logits, labels), expected)


def test_all_methods_build_with_identical_gpt2_matrix_routing():
    config = load_config(CONFIG_PATH)
    routed_names = []
    for method in config["methods"]:
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_ctx=16,
                n_embd=16,
                n_layer=1,
                n_head=2,
            )
        )
        bundle, runtime = build_optimizer(model, optimizer_config(method))
        assert bundle.optimizers
        assert runtime["matrix_names"]
        assert all("wte" not in name for name in runtime["matrix_names"])
        routed_names.append(runtime["matrix_names"])

    assert routed_names[0] == routed_names[1] == routed_names[2]


def test_graph_input_uses_only_completed_paired_seeds(tmp_path):
    jobs = []
    for method in METHODS:
        for seed in (0, 1):
            output = tmp_path / "runs" / method / f"seed_{seed}"
            output.mkdir(parents=True)
            metrics = [
                {"step": 0, "validation_nll": 3.0 + seed},
                {"step": 10, "validation_nll": 2.0 + seed},
            ]
            (output / "metrics.jsonl").write_text(
                "".join(json.dumps(metric) + "\n" for metric in metrics),
                encoding="utf-8",
            )
            jobs.append(
                {
                    "method": method,
                    "seed": seed,
                    "status": "completed" if seed == 0 else "pending",
                    "output_directory": str(output),
                }
            )
    (tmp_path / "comparison_state.json").write_text(
        json.dumps({"jobs": jobs}), encoding="utf-8"
    )

    curves, seeds = read_paired_curves(tmp_path)
    aggregates = aggregate_curves(curves, seeds)

    assert seeds == [0]
    assert aggregates["adamw"][-1] == (10, 2.0, 2.0, 2.0)
