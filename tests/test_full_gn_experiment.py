import importlib.util


def test_full_gn_task_uses_the_requested_curvature_batch_size():
    assert importlib.util.find_spec("full_gn_experiment") is not None
    from full_gn_experiment import full_gn_task

    task = full_gn_task(batch_size=2)

    assert task.micro_batch_size == 2
    assert task.gradient_accumulation == 1


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
