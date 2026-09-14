from __future__ import annotations

import json

import pytest
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

    routing_paths = qwen_trial_paths(tmp_path, "routing_resistance_v1", "screen")
    routing_arguments = parse_args(
        [
            "run",
            "--optimizer",
            "routing_resistance_v1",
            "--run-label",
            "screen",
            "--learning-rate",
            "5e-5",
            "--routing-rho",
            "0.5",
            "--routing-interval",
            "3",
        ]
    )
    assert routing_paths.checkpoint.parent == tmp_path / ".cache" / "qwen3_0p6b" / "checkpoints"
    assert routing_arguments.optimizer == "routing_resistance_v1"
    assert routing_arguments.routing_rho == 0.5
    assert routing_arguments.routing_interval == 3

    tied_paths = qwen_trial_paths(tmp_path, "tied_path_curvature_v1", "screen")
    tied_arguments = parse_args(
        ["run", "--optimizer", "tied_path_curvature_v1", "--run-label", "screen", "--learning-rate", "5e-5"]
    )
    assert tied_paths.checkpoint.parent == tmp_path / ".cache" / "qwen3_0p6b" / "checkpoints"
    assert tied_arguments.optimizer == "tied_path_curvature_v1"


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


def test_candidate_renderer_includes_the_three_matched_baselines(tmp_path):
    from qwen3_ppl_experiment import qwen_trial_paths, render_qwen_candidate_comparison

    for optimizer, perplexity in (
        ("adamw", 3.0),
        ("muon", 2.5),
        ("muown", 2.0),
        ("routing_resistance_v1", 1.9),
    ):
        paths = qwen_trial_paths(tmp_path, optimizer, "formal")
        paths.metric.parent.mkdir(parents=True, exist_ok=True)
        paths.metric.write_text(
            json.dumps({"step": 7, "elapsed_seconds": 2.0, "perplexity": perplexity}) + "\n"
        )

    step_png, time_png = render_qwen_candidate_comparison(
        tmp_path, run_label="formal", candidate="routing_resistance_v1"
    )

    assert step_png.is_file()
    assert time_png.is_file()
    assert "routing_resistance_v1" in step_png.name


def test_proposal_admission_requires_completed_manifest_matched_formal_baselines(tmp_path):
    from qwen3_ppl_experiment import _require_completed_qwen_baselines, qwen_trial_paths

    with pytest.raises(RuntimeError, match="missing completed formal baseline"):
        _require_completed_qwen_baselines(
            tmp_path,
            run_label="formal",
            manifest_digest="manifest-a",
            expected_epochs=3,
        )

    for optimizer in ("adamw", "muon", "muown"):
        paths = qwen_trial_paths(tmp_path, optimizer, "formal")
        paths.result.parent.mkdir(parents=True, exist_ok=True)
        paths.result.write_text(
            json.dumps(
                {
                    "optimizer": optimizer,
                    "run_label": "formal",
                    "completed_epochs": 3,
                    "completed_updates": 9,
                    "final_perplexity": 2.0,
                    "data_manifest_sha256": "manifest-a",
                }
            )
        )

    _require_completed_qwen_baselines(
        tmp_path,
        run_label="formal",
        manifest_digest="manifest-a",
        expected_epochs=3,
    )


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


def test_full_checkpoint_evaluation_uses_every_validation_block(tmp_path, monkeypatch):
    from qwen3_data import prepare_qwen_fineweb_cache
    from qwen3_ppl_experiment import (
        QwenTrialConfig,
        evaluate_qwen_checkpoint,
        qwen_full_evaluation_path,
        run_qwen_trial,
    )

    prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=17,
        validation_tokens=13,
        sequence_length=4,
        eos_token_id=31,
        source=[
            ("train", "a", list(range(17))),
            ("validation", "b", [value % 31 for value in range(13)]),
        ],
    )
    monkeypatch.setattr("qwen3_ppl_experiment.load_qwen3_model", lambda _: _TinyCausalLM())
    run_qwen_trial(
        QwenTrialConfig(
            root=tmp_path,
            optimizer="adamw",
            run_label="full-evaluation",
            learning_rate=0.001,
            micro_batch_size=2,
            maximum_updates=1,
            validation_batches=1,
            device="cpu",
        )
    )

    result = evaluate_qwen_checkpoint(
        tmp_path, optimizer="adamw", run_label="full-evaluation", device="cpu"
    )

    assert result["validation_batches"] == 2
    assert result["validation_tokens"] == 12
    assert result["full_validation"] is True
    assert qwen_full_evaluation_path(tmp_path, "adamw", "full-evaluation").is_file()


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


def test_trial_checkpoints_every_evaluation_interval(tmp_path, monkeypatch):
    from qwen3_data import prepare_qwen_fineweb_cache
    from qwen3_ppl_experiment import QwenTrialConfig, run_qwen_trial

    prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=17,
        validation_tokens=9,
        sequence_length=4,
        eos_token_id=31,
        source=[("train", "a", list(range(17))), ("validation", "b", list(range(9)))],
    )
    calls: list[int] = []
    monkeypatch.setattr("qwen3_ppl_experiment.load_qwen3_model", lambda _: _TinyCausalLM())
    monkeypatch.setattr("qwen3_ppl_experiment._write_qwen_checkpoint", lambda **payload: calls.append(payload["completed_updates"]))

    run_qwen_trial(
        QwenTrialConfig(
            root=tmp_path,
            optimizer="adamw",
            run_label="periodic-checkpoint",
            learning_rate=0.001,
            micro_batch_size=1,
            maximum_updates=2,
            validation_batches=1,
            evaluation_interval_updates=1,
            device="cpu",
        )
    )

    assert calls == [1, 2]


def test_trial_resumes_from_periodic_checkpoint_without_duplicate_metric_steps(tmp_path, monkeypatch):
    import qwen3_ppl_experiment
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
    real_write = qwen3_ppl_experiment._write_qwen_checkpoint

    def interrupt_after_first_checkpoint(**payload):
        real_write(**payload)
        if payload["completed_updates"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        "qwen3_ppl_experiment._write_qwen_checkpoint", interrupt_after_first_checkpoint
    )
    config = QwenTrialConfig(
        root=tmp_path,
        optimizer="adamw",
        run_label="resume",
        learning_rate=0.001,
        micro_batch_size=1,
        maximum_updates=3,
        validation_batches=1,
        evaluation_interval_updates=1,
        device="cpu",
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_qwen_trial(config)

    monkeypatch.setattr("qwen3_ppl_experiment._write_qwen_checkpoint", real_write)
    result = run_qwen_trial(
        QwenTrialConfig(**{**config.__dict__, "resume": True})
    )
    records = [
        json.loads(line)
        for line in qwen_trial_paths(tmp_path, "adamw", "resume").metric.read_text().splitlines()
    ]

    assert [record["step"] for record in records] == [1, 2, 3]
    assert result["completed_updates"] == 3

    control = run_qwen_trial(
        QwenTrialConfig(**{**config.__dict__, "run_label": "control"})
    )
    resumed_checkpoint = torch.load(
        qwen_trial_paths(tmp_path, "adamw", "resume").checkpoint, weights_only=False
    )
    control_checkpoint = torch.load(
        qwen_trial_paths(tmp_path, "adamw", "control").checkpoint, weights_only=False
    )
    for name, parameter in control_checkpoint["model"].items():
        torch.testing.assert_close(resumed_checkpoint["model"][name], parameter)
    assert result["final_perplexity"] == pytest.approx(control["final_perplexity"])


def test_trial_passes_the_training_batch_to_an_optimizer_prepare_batch_hook(tmp_path, monkeypatch):
    from qwen3_data import prepare_qwen_fineweb_cache
    from qwen3_ppl_experiment import QwenTrialConfig, run_qwen_trial

    class Recorder:
        def __init__(self) -> None:
            self.batches: list[torch.Tensor] = []

        def zero_grad(self, set_to_none: bool = True) -> None:
            del set_to_none

        def prepare_batch(self, input_ids: torch.Tensor) -> None:
            self.batches.append(input_ids.detach().clone())

        def step(self) -> None:
            pass

        def state_dict(self) -> dict:
            return {}

    prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=9,
        validation_tokens=5,
        sequence_length=4,
        eos_token_id=31,
        source=[("train", "a", list(range(9))), ("validation", "b", list(range(5)))],
    )
    recorder = Recorder()
    monkeypatch.setattr("qwen3_ppl_experiment.load_qwen3_model", lambda _: _TinyCausalLM())
    monkeypatch.setattr("qwen3_ppl_experiment.build_qwen_optimizers", lambda *args, **kwargs: {"hook": recorder})

    run_qwen_trial(
        QwenTrialConfig(
            root=tmp_path,
            optimizer="adamw",
            run_label="hook",
            learning_rate=0.001,
            micro_batch_size=1,
            maximum_updates=1,
            validation_batches=1,
            device="cpu",
        )
    )

    assert len(recorder.batches) == 1
    assert tuple(recorder.batches[0].shape) == (1, 4)


def test_cli_parses_a_render_request_without_training_arguments():
    from run_qwen3_ppl import parse_args

    arguments = parse_args(["render", "--run-label", "formal"])

    assert arguments.command == "render"
    assert arguments.run_label == "formal"


def test_cli_parses_a_candidate_render_request():
    from run_qwen3_ppl import parse_args

    arguments = parse_args(
        ["render-candidate", "--run-label", "formal", "--candidate", "routing_resistance_v1"]
    )

    assert arguments.command == "render-candidate"
    assert arguments.candidate == "routing_resistance_v1"


def test_cli_parses_a_full_checkpoint_evaluation_request():
    from run_qwen3_ppl import parse_args

    arguments = parse_args(
        ["evaluate-checkpoint", "--optimizer", "muon", "--run-label", "formal", "--device", "cpu"]
    )

    assert arguments.command == "evaluate-checkpoint"
    assert arguments.optimizer == "muon"
    assert arguments.device == "cpu"
