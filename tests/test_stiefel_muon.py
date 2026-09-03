import math

import torch
import pytest


def _orthogonality_error(matrix: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(matrix.shape[1], dtype=matrix.dtype)
    return (matrix.T @ matrix - identity).norm()


def test_square_article_update_stays_on_orthogonal_manifold():
    from stiefel_muon import initialize_stiefel_matrix, square_stiefel_update

    torch.manual_seed(3)
    weight = initialize_stiefel_matrix(torch.randn(8, 8))
    gradient = torch.randn_like(weight)

    updated, direction = square_stiefel_update(
        weight, gradient, step_size=0.02, ns_steps=8
    )

    assert (weight.T @ direction + direction.T @ weight).norm() < 2.0e-4
    assert _orthogonality_error(updated) < 2.0e-4


def test_rectangular_article_update_stays_on_stiefel_manifold():
    from stiefel_muon import initialize_stiefel_matrix, rectangular_stiefel_update

    torch.manual_seed(5)
    weight = initialize_stiefel_matrix(torch.randn(12, 4))
    gradient = torch.randn_like(weight)

    updated, direction = rectangular_stiefel_update(
        weight, gradient, step_size=0.02, ns_steps=8
    )

    assert (weight.T @ direction + direction.T @ weight).norm() < 2.0e-4
    assert _orthogonality_error(updated) < 2.0e-4


def test_rectangular_qr_direction_matches_full_article_generator():
    from stiefel_muon import (
        _matrix_sign,
        initialize_stiefel_matrix,
        rectangular_stiefel_update,
    )

    torch.manual_seed(6)
    weight = initialize_stiefel_matrix(torch.randn(12, 4))
    gradient = torch.randn_like(weight)

    _, direction = rectangular_stiefel_update(
        weight, gradient, step_size=0.02, ns_steps=8
    )
    full_generator = _matrix_sign(gradient @ weight.T - weight @ gradient.T, ns_steps=8)
    full_generator = 0.5 * (full_generator - full_generator.T)

    torch.testing.assert_close(direction, full_generator @ weight, rtol=3e-4, atol=3e-4)


def test_row_stiefel_weight_is_updated_via_its_transpose():
    from stiefel_muon import initialize_stiefel_matrix, stiefel_update

    torch.manual_seed(7)
    weight = initialize_stiefel_matrix(torch.randn(11, 4)).T
    gradient = torch.randn_like(weight)

    updated, direction, route = stiefel_update(
        weight, gradient, step_size=0.02, ns_steps=8
    )

    assert route == "row_stiefel"
    assert (weight @ direction.T + direction @ weight.T).norm() < 2.0e-4
    assert (updated @ updated.T - torch.eye(weight.shape[0])).norm() < 2.0e-4


def test_optimizer_routes_only_baseline_muon_matrices_to_stiefel_updates():
    from models import DecoderTransformer
    from optimizers import build_optimizers, muon_parameter_names

    model = DecoderTransformer(width=16, heads=4, layers=1, vocabulary_size=64)
    optimizers = build_optimizers(
        model, "stiefel_muon", lr=0.001, weight_decay=0.01, auxiliary_lr=0.0003
    )
    selected = muon_parameter_names(model)
    selected_ids = {id(parameter) for parameter in optimizers["stiefel_muon"].param_groups[0]["params"]}

    assert selected_ids == {id(parameter) for name, parameter in model.named_parameters() if name in selected}
    assert optimizers["adamw_aux"].param_groups[0]["lr"] == 0.0003


def test_optimizer_preserves_initial_column_scale_while_enforcing_stiefel_geometry():
    from stiefel_muon import StiefelMuon

    torch.manual_seed(11)
    parameter = torch.nn.Parameter(0.02 * torch.randn(12, 4))
    expected_scale = parameter.norm() / parameter.shape[1] ** 0.5

    StiefelMuon([parameter], lr=0.001)

    actual_scale = parameter.norm() / parameter.shape[1] ** 0.5
    torch.testing.assert_close(actual_scale, expected_scale)
    assert _orthogonality_error(parameter / actual_scale) < 2.0e-4


def test_hybrid_optimizer_uses_muon_for_first_last_blocks_and_stiefel_interior():
    from models import DecoderTransformer
    from optimizers import build_optimizers, hybrid_muon_parameter_names, muon_parameter_names

    model = DecoderTransformer(width=16, heads=4, layers=3, vocabulary_size=64)
    optimizers = build_optimizers(
        model,
        "hybrid_stiefel_muon",
        lr=0.0003,
        stiefel_lr=0.003,
        weight_decay=0.01,
        auxiliary_lr=0.0003,
    )
    edge_names, middle_names = hybrid_muon_parameter_names(model)
    named = dict(model.named_parameters())
    selected = muon_parameter_names(model)

    edge_ids = {id(parameter) for group in optimizers["muon_edge"].param_groups for parameter in group["params"]}
    middle_ids = {
        id(parameter)
        for group in optimizers["stiefel_muon_middle"].param_groups
        for parameter in group["params"]
    }
    assert edge_names == {
        name
        for name in selected
        if name.startswith("blocks.0.") or name.startswith("blocks.2.")
    }
    assert middle_names == {name for name in selected if name.startswith("blocks.1.")}
    assert edge_ids == {id(named[name]) for name in edge_names}
    assert middle_ids == {id(named[name]) for name in middle_names}
    assert optimizers["muon_edge"].param_groups[0]["lr"] == 0.0003
    assert optimizers["stiefel_muon_middle"].param_groups[0]["lr"] == 0.003
    assert optimizers["stiefel_muon_middle"].param_groups[0]["nesterov"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
def test_triton_newton_schulz_polynomial_matches_dense_reference():
    """A stride or polynomial-coefficient bug must not change the NS iterate."""
    from stiefel_muon import triton_newton_schulz_polynomial

    torch.manual_seed(19)
    value = torch.randn(32, 16, device="cuda")
    gram = value.T @ value
    gram_squared = gram @ gram
    expected = 3.4445 * value + value @ (-4.775 * gram + 2.0315 * gram_squared)

    actual = triton_newton_schulz_polynomial(value, gram, gram_squared)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel requires CUDA")
def test_matrix_sign_takes_the_triton_square_path_without_changing_the_polar_factor():
    from stiefel_muon import _matrix_sign

    torch.manual_seed(23)
    matrix = torch.randn(32, 32, device="cuda")
    value = matrix / matrix.norm()
    for _ in range(3):
        gram = value.T @ value
        value = 3.4445 * value + value @ (-4.775 * gram + 2.0315 * (gram @ gram))

    actual = _matrix_sign(matrix, ns_steps=3, use_triton=True)

    torch.testing.assert_close(actual, value, rtol=3e-4, atol=3e-4)


def test_hybrid_optimizer_can_tune_square_and_rectangular_stiefel_groups_separately():
    from models import DecoderTransformer
    from optimizers import build_optimizers

    model = DecoderTransformer(width=16, heads=4, layers=3, vocabulary_size=64)
    optimizers = build_optimizers(
        model,
        "hybrid_stiefel_muon",
        lr=0.0003,
        stiefel_lr=0.003,
        stiefel_square_lr=0.004,
        stiefel_rectangular_lr=0.002,
        stiefel_nesterov=True,
        weight_decay=0.01,
        auxiliary_lr=0.0003,
    )
    groups = optimizers["stiefel_muon_middle"].param_groups

    assert len(groups) == 2
    assert groups[0]["lr"] == 0.004
    assert groups[1]["lr"] == 0.002
    assert groups[0]["nesterov"] is True
    assert {tuple(parameter.shape) for parameter in groups[0]["params"]} == {(16, 16)}
    assert {tuple(parameter.shape) for parameter in groups[1]["params"]} == {(48, 16), (64, 16), (16, 64)}


def test_square_closed_form_retraction_matches_blog_formula():
    from stiefel_muon import square_closed_form_retraction

    weight = torch.eye(2)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    step_size = 0.2

    updated = square_closed_form_retraction(weight, rotation, step_size=step_size)
    expected = (torch.eye(2) - step_size * rotation) / math.sqrt(1.0 + step_size**2)

    torch.testing.assert_close(updated, expected)
    torch.testing.assert_close(updated.T @ updated, torch.eye(2))
