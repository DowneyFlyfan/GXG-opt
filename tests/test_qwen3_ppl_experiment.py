from __future__ import annotations

import json

import torch
from torch import nn


def test_qwen_paths_keep_checkpoints_under_project_cache(tmp_path):
    from qwen3_ppl_experiment import qwen_trial_paths

    paths = qwen_trial_paths(tmp_path, "muown", "screen")

    assert paths.checkpoint.parent == tmp_path / ".cache" / "qwen3_0p6b" / "checkpoints"


def test_proposal_notch_is_a_valid_trial_but_not_a_baseline_render_requirement(tmp_path):
    from qwen3_ppl_experiment import qwen_trial_paths
    from run_qwen3_ppl import parse_args

    paths = qwen_trial_paths(tmp_path, "proposal_notch_v1", "screen")
    arguments = parse_args(
        ["run", "--optimizer", "proposal_notch_v1", "--run-label", "screen", "--learning-rate", "5e-5"]
    )

    assert paths.checkpoint.parent == tmp_path / ".cache" / "qwen3_0p6b" / "checkpoints"
    assert arguments.optimizer == "proposal_notch_v1"


def test_renderer_uses_perplexity_and_completed_optimizer_steps(tmp_path):
    from qwen3_ppl_experiment import qwen_trial_paths, render_qwen_comparison

    for optimizer, perplexity in (("adamw", 3.0), ("muon", 2.5), ("muown", 2.0)):
        paths = qwen_trial_paths(tmp_path, optimizer, "formal")
        paths.metric.parent.mkdir(parents=True, exist_ok=True)
        paths.metric.write_text(
            json.dumps({"step": 7, "elapsed_seconds": 2.0, "perplexity": perplexity}) + "\n"
        )

    step_png, time_png = render_qwen_comparison(tmp_path, run_label="formal")

    assert step_png.is_file()
    assert time_png.is_file()


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.head = nn.Linear(8, 32)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        del use_cache
        return self.head(self.embedding(input_ids))


def test_trial_writes_a_checkpoint_bound_to_the_cache_manifest(tmp_path, monkeypatch):
    from qwen3_data import prepare_qwen_fineweb_cache
    from qwen3_ppl_experiment import QwenTrialConfig, qwen_trial_paths, run_qwen_trial

    prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=17,
        validation_tokens=9,
        sequence_length=4,
        eos_token_id=31,
        source=[("train", "a", list(range(17))), ("validation", "b", list(range(9)))],
    )
    monkeypatch.setattr("qwen3_ppl_experiment.load_qwen3_model", lambda _: _TinyCausalLM())
    result = run_qwen_trial(
        QwenTrialConfig(
            root=tmp_path,
            optimizer="adamw",
            run_label="smoke",
            learning_rate=0.001,
            micro_batch_size=1,
            gradient_accumulation=1,
            maximum_updates=1,
            validation_batches=1,
            device="cpu",
        )
    )

    checkpoint = torch.load(qwen_trial_paths(tmp_path, "adamw", "smoke").checkpoint, weights_only=False)
    assert result["completed_updates"] == 1
    assert checkpoint["data_manifest_sha256"] == result["data_manifest_sha256"]
    assert result["peak_memory_mib"] is None


def test_trial_records_periodic_perplexity_at_completed_steps(tmp_path, monkeypatch):
    from qwen3_data import prepare_qwen_fineweb_cache
    from qwen3_ppl_experiment import QwenTrialConfig, qwen_trial_paths, run_qwen_trial

    prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=33,
        validation_tokens=9,
        sequence_length=4,
        eos_token_id=31,
        source=[
            ("train", "a", [value % 31 for value in range(33)]),
            ("validation", "b", [value % 31 for value in range(9)]),
        ],
    )
    monkeypatch.setattr("qwen3_ppl_experiment.load_qwen3_model", lambda _: _TinyCausalLM())

    run_qwen_trial(
        QwenTrialConfig(
            root=tmp_path,
            optimizer="adamw",
            run_label="periodic",
            learning_rate=0.001,
            micro_batch_size=1,
            maximum_updates=3,
            validation_batches=1,
            evaluation_interval_updates=1,
            device="cpu",
        )
    )

    records = [
        json.loads(line)
        for line in qwen_trial_paths(tmp_path, "adamw", "periodic").metric.read_text().splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2, 3]


def test_cli_parses_a_render_request_without_training_arguments():
    from run_qwen3_ppl import parse_args

    arguments = parse_args(["render", "--run-label", "formal"])

    assert arguments.command == "render"
    assert arguments.run_label == "formal"
