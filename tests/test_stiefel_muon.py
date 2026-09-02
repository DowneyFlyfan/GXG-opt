import torch


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
