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
