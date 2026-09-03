from dataclasses import replace
import json
import inspect

import pytest
import torch

from gn_guided_adam import GNGuidedAdamW
from models import DecoderTransformer


def test_guided_gn_contract_preserves_task_and_uses_effective_batch_sixty():
    from guided_gn_experiment import validate_guided_gn_contract

    contract = validate_guided_gn_contract(
        micro_batch_size=4,
        sequence_length=1024,
        gradient_accumulation=15,
        curvature_accumulation=15,
    )

    assert contract == {
        "sequence_length": 1024,
        "gradient_effective_batch_size": 60,
        "curvature_effective_batch_size": 60,
    }


def test_guided_runner_exposes_a_positive_step_limit_for_memory_probes(tmp_path):
    from guided_gn_experiment import run_guided_gn_trial

    assert "maximum_steps" in inspect.signature(run_guided_gn_trial).parameters
    with pytest.raises(ValueError, match="maximum_steps"):
        run_guided_gn_trial(
            tmp_path,
            micro_batch_size=2,
            gradient_accumulation=30,
            curvature_accumulation=30,
            maximum_seconds=60,
            maximum_steps=0,
        )


@pytest.mark.parametrize(
    ("sequence_length", "gradient_accumulation", "curvature_accumulation", "message"),
    (
        (512, 15, 15, "sequence_length"),
        (1024, 14, 14, "gradient effective batch"),
        (1024, 15, 14, "curvature effective batch"),
        (1024, 15, 16, "cannot exceed gradient accumulation"),
    ),
)
def test_guided_gn_contract_rejects_task_drift_or_underbatched_curvature(
    sequence_length, gradient_accumulation, curvature_accumulation, message
):
    from guided_gn_experiment import validate_guided_gn_contract

    with pytest.raises(ValueError, match=message):
        validate_guided_gn_contract(
            micro_batch_size=4,
            sequence_length=sequence_length,
            gradient_accumulation=gradient_accumulation,
            curvature_accumulation=curvature_accumulation,
        )


def test_guided_gn_config_selects_only_the_requested_transformer_matrix():
    from guided_gn_experiment import guided_gn_config

    model = DecoderTransformer(width=16, heads=2, layers=2, vocabulary_size=31)
    config = guided_gn_config(
        learning_rate=3.0e-4,
        weight_decay=0.01,
        curvature_accumulation=15,
        guided_block_patterns=(r"^blocks\.1\.feedforward_out\.weight$",),
        rank=2,
        refresh_interval=200,
        initial_damping=0.01,
        warmup_steps=0,
    )
    optimizer = GNGuidedAdamW(model, replace(config, gn=replace(config.gn, min_block_numel=1)))

    assert [block.name for block in optimizer.registry.enabled] == [
        "blocks.1.feedforward_out.weight"
    ]
    assert config.adamw.betas == (0.9, 0.95)
    assert config.gn.curvature_batches == 15
    assert config.gn.rank == 2


def test_functional_curvature_batches_capture_distinct_targets_and_ids():
    from guided_gn_experiment import functional_curvature_batches

    batches = (
        (torch.tensor([[1, 2]]), torch.tensor([[2, 3]])),
        (torch.tensor([[3, 4]]), torch.tensor([[4, 5]])),
    )
    functional_batches = functional_curvature_batches(batches, completed_steps=7)

    assert [batch.batch_id for batch in functional_batches] == ["step-7-micro-0", "step-7-micro-1"]
    logits = torch.zeros(1, 2, 8)
    assert torch.isfinite(functional_batches[0].loss_fn(logits))
    assert torch.isfinite(functional_batches[1].loss_fn(logits))
    shifted = logits.clone()
    shifted[0, 0, 2] = 4.0
    assert functional_batches[0].loss_fn(shifted) < functional_batches[1].loss_fn(shifted)


def test_guided_trial_is_included_in_metric_steps_and_time_plots(tmp_path):
    from gn_experiment import artifact_paths, write_gn_comparison_plots

    for optimizer, metric in (("adamw", 0.60), ("muon", 0.65)):
        path = artifact_paths(tmp_path, optimizer).metric
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"step": 10, "metric": metric, "elapsed_seconds": 20.0})
            + "\n",
            encoding="utf-8",
        )
    guided = artifact_paths(
        tmp_path, "gn_guided_adamw", run_label="guided-pilot"
    ).metric
    guided.write_text(
        json.dumps({"step": 10, "metric": 0.70, "elapsed_seconds": 22.0}) + "\n",
        encoding="utf-8",
    )

    outputs = write_gn_comparison_plots(
        tmp_path, guided_run_label="guided-pilot"
    )

    assert outputs is not None
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
