import torch


def _condition_number(matrix: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix)
    return float(singular_values.max() / singular_values.min())


def test_newton_schulz_polar_factor_is_nearly_column_orthonormal():
    from low_spectral_variance import _newton_schulz_polar_factor

    matrix = torch.tensor(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    polar = _newton_schulz_polar_factor(matrix, iterations=8)

    torch.testing.assert_close(
        polar.T @ polar,
        torch.eye(3),
        rtol=2.0e-3,
        atol=2.0e-3,
    )


def test_direct_cayley_curve_preserves_the_spectral_and_condition_constraints():
    from low_spectral_variance import _direct_feasible_curve

    weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.70, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    direction = torch.tensor(
        [
            [0.0, 0.10, 0.0],
            [0.10, 0.0, 0.0],
            [0.0, 0.0, 0.10],
            [0.10, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    left, singular_values, right_transpose = torch.linalg.svd(weight, full_matrices=False)

    updated, used_step = _direct_feasible_curve(
        weight,
        direction,
        left,
        singular_values,
        right_transpose,
        condition_limit=4.0,
        step_size=1.0e-3,
    )

    assert 0 < used_step <= 1.0e-3
    torch.testing.assert_close(
        torch.linalg.svdvals(updated).max(), torch.tensor(1.0), rtol=2.0e-5, atol=2.0e-5
    )
    assert _condition_number(updated) <= 4.0 * (1.0 + 2.0e-5)

def test_low_spectral_variance_update_preserves_scale_and_condition_limit():
    from low_spectral_variance import low_spectral_variance_update

    weight = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.70, 0.0],
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    gradient = torch.tensor(
        [
            [0.2, -0.4, 0.1],
            [-0.3, 0.1, 0.2],
            [0.2, 0.3, -0.1],
            [0.1, -0.2, 0.3],
            [0.0, 0.1, -0.2],
        ]
    )

    updated, diagnostics = low_spectral_variance_update(
        weight, gradient, condition_limit=4.0, step_size=1.0e-3, dual_steps=8
    )

    assert updated.shape == weight.shape
    assert torch.isfinite(updated).all()
    torch.testing.assert_close(
        torch.linalg.svdvals(updated).max(), torch.tensor(1.0), rtol=1.0e-5, atol=1.0e-5
    )
    assert _condition_number(updated) <= 4.0 * (1.0 + 1.0e-4)
    assert abs(diagnostics.top_tangent_inner_product) < 2.0e-3


def test_low_spectral_variance_update_handles_wide_matrices_in_transposed_geometry():
    from low_spectral_variance import low_spectral_variance_update

    weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.6, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0, 0.0],
        ]
    )
    gradient = torch.randn_like(weight)

    updated, _ = low_spectral_variance_update(
        weight, gradient, condition_limit=4.0, step_size=1.0e-3, dual_steps=8
    )

    assert updated.shape == weight.shape
    torch.testing.assert_close(
        torch.linalg.svdvals(updated).max(), torch.tensor(1.0), rtol=1.0e-5, atol=1.0e-5
    )
    assert _condition_number(updated) <= 4.0 * (1.0 + 1.0e-4)


def test_low_spectral_variance_update_retracts_when_extreme_singular_values_tie():
    """A repeated extreme value has no unique source rotation velocity."""
    from low_spectral_variance import low_spectral_variance_update

    weight = torch.diag(torch.tensor([1.0, 0.5, 0.5, 0.5]))
    gradient = torch.tensor(
        [
            [0.1, -0.2, 0.0, 0.1],
            [0.2, 0.1, -0.1, 0.0],
            [0.0, 0.1, 0.2, -0.2],
            [-0.1, 0.0, 0.2, 0.1],
        ]
    )

    updated, _ = low_spectral_variance_update(
        weight, gradient, condition_limit=4.0, step_size=1.0e-3, dual_steps=1
    )

    assert torch.isfinite(updated).all()
    torch.testing.assert_close(
        torch.linalg.svdvals(updated).max(), torch.tensor(1.0), rtol=1.0e-5, atol=1.0e-5
    )
    assert _condition_number(updated) <= 4.0 * (1.0 + 1.0e-4)


def test_low_spectral_variance_selects_only_matrices_feasible_under_the_cap():
    from low_spectral_variance import low_spectral_variance_parameter_names

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tall = torch.nn.Parameter(torch.diag(torch.tensor([1.0, 0.6, 0.3])))
            self.ill_conditioned = torch.nn.Parameter(
                torch.diag(torch.tensor([1.0, 0.1, 0.01]))
            )

    names = low_spectral_variance_parameter_names(Model(), condition_limit=4.0)

    assert names == {"tall"}


def test_optimizer_builder_routes_infeasible_matrices_to_adamw():
    from low_spectral_variance import LowSpectralVariance
    from optimizers import build_optimizers

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.diag(torch.tensor([1.0, 0.6, 0.3])))
            self.projection = torch.nn.Parameter(torch.diag(torch.tensor([1.0, 0.1, 0.01])))

    model = Model()
    optimizers = build_optimizers(
        model,
        "low_spectral_variance",
        lr=1.0e-3,
        weight_decay=0.0,
        low_spectral_condition_limit=4.0,
    )

    assert isinstance(optimizers["low_spectral_variance"], LowSpectralVariance)
    assert optimizers["low_spectral_variance"].param_groups[0]["params"] == [
        model.linear.weight
    ]
    assert model.projection in optimizers["adamw_aux"].param_groups[0]["params"]


def test_low_spectral_variance_artifacts_use_the_nlp_result_contract(tmp_path):
    from low_spectral_variance_experiment import (
        low_spectral_variance_paths,
        low_spectral_variance_task,
    )

    paths = low_spectral_variance_paths(tmp_path, "probe")

    assert paths.metric == (
        tmp_path / "metrics/nlp/nlp_gpt_12x512__low_spectral_variance_probe.jsonl"
    )
    assert paths.result == (
        tmp_path / "results/nlp/nlp_gpt_12x512__low_spectral_variance_probe.json"
    )
    task = low_spectral_variance_task(gradient_accumulation=12)
    assert task.micro_batch_size == 12
    assert task.gradient_accumulation == 12
