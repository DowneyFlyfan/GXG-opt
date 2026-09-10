import pytest
import torch
import torch.nn.functional as functional


def test_time_limit_is_rechecked_after_evaluation():
    from paper_full_gn_experiment import paper_gn_time_limit_reached

    assert not paper_gn_time_limit_reached(13_199.9, 13_200.0)
    assert paper_gn_time_limit_reached(13_200.0, 13_200.0)
    assert paper_gn_time_limit_reached(13_201.0, 13_200.0)


def test_paper_comparison_uses_log_steps_and_only_requested_traces(
    tmp_path, monkeypatch
):
    import json

    import matplotlib.axes

    from gn_experiment import artifact_paths, write_gn_comparison_plots

    for optimizer, metric in (("adamw", 0.60), ("muon", 0.65)):
        path = artifact_paths(tmp_path, optimizer).metric
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"step": 10, "metric": metric, "elapsed_seconds": 20.0})
            + "\n",
            encoding="utf-8",
        )
    paper = artifact_paths(
        tmp_path, "paper_full_gn", run_label="paper-template"
    ).metric
    paper.write_text(
        json.dumps({"step": 1, "metric": 0.70, "elapsed_seconds": 22.0})
        + "\n",
        encoding="utf-8",
    )
    scales = []
    labels = []
    original_set_xscale = matplotlib.axes.Axes.set_xscale
    original_plot = matplotlib.axes.Axes.plot

    def record_scale(axis, value, *args, **kwargs):
        scales.append(value)
        return original_set_xscale(axis, value, *args, **kwargs)

    def record_plot(axis, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original_plot(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_xscale", record_scale)
    monkeypatch.setattr(matplotlib.axes.Axes, "plot", record_plot)

    outputs = write_gn_comparison_plots(
        tmp_path, paper_full_gn_run_label="paper-template"
    )

    assert outputs is not None
    assert scales == ["log"]
    assert set(labels) == {"AdamW", "Muon", "Paper Full GN"}


def test_formal_contract_counts_every_accumulated_inner_sequence():
    from paper_full_gn_experiment import validate_paper_gn_contract

    contract = validate_paper_gn_contract(
        micro_batch_size=1,
        sequence_length=1024,
        inner_steps=60,
        inner_gradient_accumulation=1,
    )

    assert contract == {
        "sequence_length": 1024,
        "inner_effective_batch_size": 1,
        "outer_effective_batch_size": 60,
        "outer_effective_tokens": 61_440,
        "line_search_effective_batch_size": 60,
    }


def test_formal_contract_rejects_an_outer_batch_smaller_than_sixty():
    from paper_full_gn_experiment import validate_paper_gn_contract

    with pytest.raises(ValueError, match="outer effective batch"):
        validate_paper_gn_contract(
            micro_batch_size=1,
            sequence_length=1024,
            inner_steps=59,
            inner_gradient_accumulation=1,
        )


def test_warmup_target_matches_five_percent_of_chinchilla_tokens():
    from paper_full_gn_experiment import chinchilla_warmup_tokens

    assert chinchilla_warmup_tokens(54_680_000) == 54_680_000


def test_language_model_curvature_batch_uses_cross_entropy_on_all_tokens():
    from paper_full_gn_experiment import language_model_curvature_batch

    logits = torch.tensor(
        [[[2.0, 0.0], [0.5, 1.5]]], dtype=torch.float64
    )
    targets = torch.tensor([[0, 1]])

    batch = language_model_curvature_batch((logits, targets), torch.device("cpu"))

    assert batch.args == (logits,)
    assert batch.loss_fn(logits) == pytest.approx(
        functional.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1)).item()
    )


def test_language_model_curvature_batch_has_exact_chunked_cross_entropy_hvp():
    from paper_full_gn_experiment import language_model_curvature_batch

    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.5, 1.5, -0.5]]], dtype=torch.float64
    )
    targets = torch.tensor([[0, 1]])
    tangent = torch.tensor(
        [[[0.2, -0.4, 0.3], [0.1, 0.7, -0.2]]], dtype=torch.float64
    )
    batch = language_model_curvature_batch((logits, targets), torch.device("cpu"))

    assert batch.output_hvp_fn is not None
    actual = batch.output_hvp_fn(logits, tangent)
    expected = torch.autograd.functional.hvp(
        lambda value: functional.cross_entropy(
            value.reshape(-1, value.size(-1)), targets.reshape(-1)
        ),
        logits,
        tangent,
    )[1]

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_checkpoint_restores_outer_inner_and_persistent_optimizer_state(tmp_path):
    from paper_full_gn_experiment import load_paper_gn_checkpoint
    from paper_full_gn_experiment import save_paper_gn_checkpoint

    outer = torch.nn.Linear(1, 1, bias=False)
    inner = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        outer.weight.fill_(1.0)
        inner.weight.fill_(3.0)
    optimizer = torch.optim.SGD(inner.parameters(), lr=0.1, momentum=0.9)
    inner.weight.grad = torch.ones_like(inner.weight)
    optimizer.step()
    checkpoint = tmp_path / "paper-gn.pt"

    save_paper_gn_checkpoint(
        checkpoint,
        outer,
        inner,
        {"inner": optimizer},
        completed_outer_steps=7,
        consumed_training_batches=140,
        elapsed_seconds=12.5,
    )
    restored_outer = torch.nn.Linear(1, 1, bias=False)
    restored_inner = torch.nn.Linear(1, 1, bias=False)
    restored_optimizer = torch.optim.SGD(
        restored_inner.parameters(), lr=0.1, momentum=0.9
    )

    restored = load_paper_gn_checkpoint(
        checkpoint,
        restored_outer,
        restored_inner,
        {"inner": restored_optimizer},
    )

    assert restored == (7, 140, 12.5)
    assert restored_outer.weight.item() == pytest.approx(1.0)
    assert restored_inner.weight.item() == pytest.approx(2.9)
    state = next(iter(restored_optimizer.state.values()))
    assert state["momentum_buffer"].item() == pytest.approx(1.0)
