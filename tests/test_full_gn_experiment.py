import importlib.util
import inspect
import json

import pytest


def test_labeled_full_gn_trial_is_included_in_comparison_plots(tmp_path):
    from gn_experiment import artifact_paths, write_gn_comparison_plots

    for optimizer, metric in (("adamw", 0.60), ("muon", 0.65)):
        path = artifact_paths(tmp_path, optimizer).metric
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"step": 10, "metric": metric, "elapsed_seconds": 20.0})
            + "\n",
            encoding="utf-8",
        )
    labeled = artifact_paths(
        tmp_path, "full_ggn", run_label="curvature-60"
    ).metric
    labeled.write_text(
        json.dumps({"step": 10, "metric": 0.70, "elapsed_seconds": 22.0})
        + "\n",
        encoding="utf-8",
    )

    outputs = write_gn_comparison_plots(
        tmp_path, full_ggn_run_label="curvature-60"
    )

    assert outputs is not None
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)


def test_full_gn_task_uses_the_requested_curvature_batch_size():
    assert importlib.util.find_spec("full_gn_experiment") is not None
    from full_gn_experiment import full_gn_task

    task = full_gn_task(batch_size=2)

    assert task.micro_batch_size == 2
    assert task.gradient_accumulation == 1


def test_formal_full_gn_contract_requires_unchanged_context_and_effective_batches():
    from full_gn_experiment import validate_formal_full_gn_contract

    contract = validate_formal_full_gn_contract(
        batch_size=1,
        sequence_length=1024,
        gradient_accumulation_batches=60,
        curvature_accumulation_batches=60,
    )

    assert contract == {
        "sequence_length": 1024,
        "gradient_effective_batch_size": 60,
        "curvature_effective_batch_size": 60,
    }


@pytest.mark.parametrize(
    ("sequence_length", "gradient_batches", "curvature_batches", "message"),
    (
        (560, 60, 60, "sequence_length"),
        (1024, 59, 59, "gradient effective batch"),
        (1024, 60, 1, "curvature effective batch"),
    ),
)
def test_formal_full_gn_contract_rejects_changed_or_underbatched_tasks(
    sequence_length, gradient_batches, curvature_batches, message
):
    from full_gn_experiment import validate_formal_full_gn_contract

    with pytest.raises(ValueError, match=message):
        validate_formal_full_gn_contract(
            batch_size=1,
            sequence_length=sequence_length,
            gradient_accumulation_batches=gradient_batches,
            curvature_accumulation_batches=curvature_batches,
        )


def test_full_gn_batch_preserves_requested_sequences_and_truncates_tokens():
    from full_gn_experiment import prepare_full_gn_batch
    import torch

    tokens = torch.arange(24).reshape(3, 8)
    targets = tokens + 1

    selected_tokens, selected_targets = prepare_full_gn_batch(
        (tokens, targets), batch_size=2, sequence_length=4
    )

    assert torch.equal(selected_tokens, tokens[:2, :4])
    assert torch.equal(selected_targets, targets[:2, :4])


def test_full_gn_config_exposes_a_damping_floor():
    from full_gn_experiment import full_gn_config

    config = full_gn_config(maximum_cg_iterations=4, minimum_damping=0.01)

    assert config.maximum_cg_iterations == 4
    assert config.minimum_damping == 0.01


def test_full_gn_config_exposes_empirical_fisher_preconditioning():
    from full_gn_experiment import full_gn_config

    assert "preconditioner_exponent" in inspect.signature(full_gn_config).parameters
    config = full_gn_config(
        maximum_cg_iterations=6,
        minimum_damping=0.001,
        preconditioner_exponent=0.75,
    )

    assert config.preconditioner_exponent == 0.75


def test_full_gn_config_can_disable_stochastic_cg_warm_start():
    from full_gn_experiment import full_gn_config

    assert "cg_warm_start_decay" in inspect.signature(full_gn_config).parameters
    config = full_gn_config(
        maximum_cg_iterations=6,
        minimum_damping=0.001,
        cg_warm_start_decay=0.0,
    )

    assert config.cg_warm_start_decay == 0.0


def test_full_gn_batches_partition_contiguous_token_windows():
    from full_gn_experiment import prepare_full_gn_batches
    import torch

    tokens = torch.arange(24).reshape(3, 8)
    targets = tokens + 1

    selected = prepare_full_gn_batches(
        (tokens, targets), batch_size=2, sequence_length=4, curvature_batches=2
    )

    assert len(selected) == 2
    assert torch.equal(selected[0][0], tokens[:2, :4])
    assert torch.equal(selected[1][0], tokens[:2, 4:8])


def test_full_gn_batch_can_rotate_away_from_the_fixed_prefix():
    import full_gn_experiment
    import torch

    assert "window_offset" in inspect.signature(
        full_gn_experiment.prepare_full_gn_batches
    ).parameters
    tokens = torch.arange(24).reshape(3, 8)
    targets = tokens + 1

    selected = full_gn_experiment.prepare_full_gn_batches(
        (tokens, targets),
        batch_size=2,
        sequence_length=4,
        curvature_batches=1,
        window_offset=2,
    )

    assert torch.equal(selected[0][0], tokens[:2, 2:6])
    assert torch.equal(selected[0][1], targets[:2, 2:6])


def test_full_gn_rotating_offset_is_deterministic_and_not_prefix_only():
    import full_gn_experiment

    assert hasattr(full_gn_experiment, "rotating_window_offset")
    assert full_gn_experiment.rotating_window_offset(
        completed_steps=0, full_length=1024, selected_length=576
    ) == 0
    assert full_gn_experiment.rotating_window_offset(
        completed_steps=1, full_length=1024, selected_length=576
    ) == 127


def test_full_gn_accumulation_uses_distinct_loader_batches_without_changing_window_shape():
    from full_gn_experiment import prepare_accumulated_full_gn_batches
    import torch

    first = (torch.arange(24).reshape(3, 8), torch.arange(24).reshape(3, 8) + 1)
    second = (torch.arange(24, 48).reshape(3, 8), torch.arange(24, 48).reshape(3, 8) + 1)

    selected = prepare_accumulated_full_gn_batches(
        (first, second), batch_size=2, sequence_length=4, curvature_batches=1
    )

    assert len(selected) == 2
    assert torch.equal(selected[0][0], first[0][:2, :4])
    assert torch.equal(selected[1][0], second[0][:2, :4])


def test_full_gn_operator_keeps_all_accumulated_batches_for_gradient_but_not_curvature():
    import full_gn_experiment
    import torch
    from kronecker_ggn_common.curvature_operator import (
        FunctionalCurvatureBatch,
        GGNFullOperator,
    )

    assert hasattr(full_gn_experiment, "build_split_full_gn_operator")
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    first = GGNFullOperator(
        model,
        FunctionalCurvatureBatch(
            (torch.tensor([[1.0]], dtype=torch.float64),),
            lambda output: 0.5 * (output - 1.0).square().sum(),
        ),
    )
    second = GGNFullOperator(
        model,
        FunctionalCurvatureBatch(
            (torch.tensor([[3.0]], dtype=torch.float64),),
            lambda output: 0.5 * (output - 2.0).square().sum(),
        ),
    )

    operator = full_gn_experiment.build_split_full_gn_operator(
        (first, second),
        curvature_accumulation_batches=1,
        curvature_batches=1,
    )

    assert torch.allclose(
        operator.gradient(), torch.tensor([-3.5], dtype=torch.float64)
    )
    assert torch.allclose(
        operator.matvec(torch.tensor([0.25], dtype=torch.float64)),
        torch.tensor([0.25], dtype=torch.float64),
    )


def test_full_gn_operator_averages_every_accumulated_curvature_batch():
    from full_gn_experiment import build_split_full_gn_operator
    import torch
    from kronecker_ggn_common.curvature_operator import (
        FunctionalCurvatureBatch,
        GGNFullOperator,
    )

    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    operators = tuple(
        GGNFullOperator(
            model,
            FunctionalCurvatureBatch(
                (torch.tensor([[value]], dtype=torch.float64),),
                lambda output: 0.5 * output.square().sum(),
                batch_id=f"curvature-{index}",
            ),
        )
        for index, value in enumerate((1.0, 3.0))
    )

    operator = build_split_full_gn_operator(
        operators,
        curvature_accumulation_batches=2,
        curvature_batches=1,
    )

    assert torch.allclose(
        operator.matvec(torch.tensor([0.25], dtype=torch.float64)),
        torch.tensor([1.25], dtype=torch.float64),
    )
