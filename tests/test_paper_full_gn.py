import math

import pytest
import torch
from torch import nn

from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch


def _squared_error_batch(value: torch.Tensor, target: float):
    return FunctionalCurvatureBatch(
        args=(value,),
        loss_fn=lambda output: 0.5 * (output - target).square().sum(),
    )


def test_quadratic_gradient_includes_reference_gradient_and_ggn_delta():
    from paper_full_gn import quadratic_gradient

    reference = torch.nn.Linear(1, 1, bias=False).double()
    inner = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        reference.weight.fill_(2.0)
        inner.weight.fill_(4.0)

    gradient = quadratic_gradient(
        reference,
        inner,
        _squared_error_batch(torch.tensor([[3.0]], dtype=torch.float64), 1.0),
    )

    # At w0=2, g0=3*(3*2-1)=15 and G*(w-w0)=3^2*(4-2)=18.
    assert torch.equal(gradient, torch.tensor([33.0], dtype=torch.float64))


def test_held_out_line_search_updates_outer_but_preserves_warm_start_endpoint():
    from paper_full_gn import held_out_line_search

    outer = torch.nn.Linear(1, 1, bias=False).double()
    inner = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        outer.weight.zero_()
        inner.weight.fill_(4.0)
    batches = (_squared_error_batch(torch.ones(1, 1, dtype=torch.float64), 1.0),)

    result = held_out_line_search(outer, inner, batches, search_range=5)

    assert result.step_size == pytest.approx(0.25)
    assert outer.weight.item() == pytest.approx(1.0)
    assert inner.weight.item() == pytest.approx(4.0)
    assert result.loss == pytest.approx(0.0)


def test_inner_cosine_schedule_restarts_for_each_outer_step():
    from paper_full_gn import inner_cosine_multiplier

    first_cycle = [inner_cosine_multiplier(index, 4) for index in range(4)]
    second_cycle = [inner_cosine_multiplier(index, 4) for index in range(4)]

    assert first_cycle == pytest.approx(
        [1.0, 0.8535533905932737, 0.5, 0.14644660940672627]
    )
    assert second_cycle == pytest.approx(first_cycle)


def test_inner_step_assigns_quadratic_gradients_to_real_optimizer_parameters():
    from paper_full_gn import paper_inner_step

    reference = torch.nn.Linear(1, 1, bias=False).double()
    inner = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        reference.weight.fill_(2.0)
        inner.weight.fill_(4.0)
    optimizer = torch.optim.SGD(inner.parameters(), lr=0.1)

    norm = paper_inner_step(
        reference,
        inner,
        _squared_error_batch(torch.tensor([[3.0]], dtype=torch.float64), 1.0),
        {"sgd": optimizer},
        {"sgd": 0.1},
        inner_step=0,
        inner_steps=1,
        gradient_clip=100.0,
    )

    assert norm == pytest.approx(33.0)
    assert inner.weight.item() == pytest.approx(0.7)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_accumulated_inner_step_averages_gradient_and_curvature_batches():
    from paper_full_gn import paper_accumulated_inner_step

    reference = torch.nn.Linear(1, 1, bias=False).double()
    inner = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        reference.weight.zero_()
        inner.weight.zero_()
    optimizer = torch.optim.SGD(inner.parameters(), lr=0.1)
    batches = (
        _squared_error_batch(torch.tensor([[1.0]], dtype=torch.float64), 1.0),
        _squared_error_batch(torch.tensor([[3.0]], dtype=torch.float64), 2.0),
    )

    norm = paper_accumulated_inner_step(
        reference,
        inner,
        batches,
        {"sgd": optimizer},
        {"sgd": 0.1},
        inner_step=0,
        inner_steps=1,
        gradient_clip=100.0,
    )

    assert norm == pytest.approx(3.5)
    assert inner.weight.item() == pytest.approx(0.35)


def test_line_search_grid_matches_the_paper_sqrt_two_schedule():
    from paper_full_gn import line_search_step_sizes

    assert line_search_step_sizes(5) == pytest.approx(
        tuple(1.0 / math.sqrt(2.0) ** index for index in range(5))
    )


def test_paper_muon_uses_optax_width_transfer_scaling():
    from paper_full_gn import PaperMuon

    assert PaperMuon.scaled_lr(0.01, rows=512, columns=512) == pytest.approx(0.01)
    assert PaperMuon.scaled_lr(0.01, rows=2048, columns=512) == pytest.approx(0.02)
    assert PaperMuon.scaled_lr(0.01, rows=512, columns=2048) == pytest.approx(0.01)


def test_paper_inner_optimizer_uses_muon_for_hidden_matrix_and_adam_b2_999():
    from paper_full_gn import PaperMuon, build_paper_inner_optimizers

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.projection = nn.Linear(4, 4, bias=False)
            self.norm = nn.LayerNorm(4)

    model = TinyDecoder()
    optimizers = build_paper_inner_optimizers(
        model, learning_rate=0.01, weight_decay=0.001
    )

    assert isinstance(optimizers["muon"], PaperMuon)
    assert optimizers["muon"].param_groups[0]["weight_decay"] == 0.0
    assert optimizers["adamw_aux"].param_groups[0]["betas"] == (0.9, 0.999)
    assert optimizers["adamw_aux"].param_groups[0]["weight_decay"] == 0.001
    muon_parameters = optimizers["muon"].param_groups[0]["params"]
    assert muon_parameters == [model.projection.weight]
