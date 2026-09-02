import pytest
import torch


def test_factored_kronecker_diagonal_matches_the_calibration_rayleigh_quotient():
    from recycled_low_rank_gn import build_factored_kronecker_diagonal

    gradient = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    curvature = torch.diag(torch.tensor([2.0, 3.0, 5.0, 7.0], dtype=torch.float64))
    damping = 0.25

    diagonal = build_factored_kronecker_diagonal(
        gradient,
        parameter_shapes=((2, 2),),
        curvature_matvec=lambda vector: curvature @ vector,
        damping=damping,
    )
    seed = gradient / gradient.norm()
    assert diagonal.shape == gradient.shape
    assert torch.all(diagonal > damping)
    assert torch.dot(seed, diagonal * seed) == pytest.approx(
        torch.dot(seed, (curvature + damping * torch.eye(4, dtype=gradient.dtype)) @ seed).item()
    )


def test_factored_diagonal_can_reuse_an_already_accumulated_curvature_product():
    from recycled_low_rank_gn import build_factored_kronecker_diagonal

    gradient = torch.tensor([1.0, 2.0], dtype=torch.float64)
    curvature = torch.diag(torch.tensor([3.0, 5.0], dtype=torch.float64))
    seed = gradient / gradient.norm()

    diagonal = build_factored_kronecker_diagonal(
        gradient,
        parameter_shapes=((1, 2),),
        curvature_matvec=lambda _: (_ for _ in ()).throw(
            AssertionError("must not recompute curvature")
        ),
        curvature_seed=curvature @ seed,
        damping=0.1,
    )

    assert torch.dot(seed, diagonal * seed) == pytest.approx(
        torch.dot(seed, (curvature + 0.1 * torch.eye(2)) @ seed).item()
    )


def test_factored_accumulator_uses_the_second_moment_before_gradient_averaging():
    from recycled_low_rank_gn import FactoredKroneckerAccumulator

    accumulator = FactoredKroneckerAccumulator(
        parameter_shapes=((2, 2),),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    second = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float64)
    accumulator.update_scaled_block(0, 0.25 * first, batch_weight=0.25)
    accumulator.update_scaled_block(0, 0.75 * second, batch_weight=0.75)

    actual = accumulator.diagonal().reshape(2, 2)
    squared = 0.25 * first.square() + 0.75 * second.square()
    row = squared.mean(dim=1)
    column = squared.mean(dim=0)
    expected = row[:, None] * column[None, :] / squared.mean()

    assert torch.allclose(actual, expected)
    averaged_gradient_square = (0.25 * first + 0.75 * second).square()
    assert not torch.allclose(squared, averaged_gradient_square)

def test_full_rank_signed_relative_residual_recovers_the_dense_inverse():
    from recycled_low_rank_gn import build_signed_relative_residual

    system = torch.tensor(
        [[5.0, 1.0, 0.3], [1.0, 3.0, 0.2], [0.3, 0.2, 2.0]],
        dtype=torch.float64,
    )
    base_diagonal = torch.tensor([4.0, 2.0, 1.5], dtype=torch.float64)
    residual = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    correction = build_signed_relative_residual(
        lambda vector: system @ vector,
        base_diagonal,
        seed=torch.tensor([1.0, 0.7, -0.4], dtype=torch.float64),
        rank=3,
    )

    actual = correction.apply(residual, base_diagonal)
    expected = torch.linalg.solve(system, residual)

    assert correction.rank == 3
    assert torch.allclose(actual, expected, atol=1.0e-10, rtol=1.0e-10)


def test_eigenvalue_clipping_keeps_the_low_rank_inverse_positive_definite():
    from recycled_low_rank_gn import LowRankResidualCorrection

    correction = LowRankResidualCorrection(
        basis=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        projected_matrix=torch.tensor([[-2.0]], dtype=torch.float64),
        minimum_relative_eigenvalue=0.1,
    )
    residual = torch.tensor([3.0, -2.0], dtype=torch.float64)

    preconditioned = correction.apply(residual, torch.ones(2, dtype=torch.float64))

    assert torch.dot(residual, preconditioned) > 0
    assert preconditioned[0] == pytest.approx(30.0)
    assert preconditioned[1] == pytest.approx(-2.0)


def test_exact_low_rank_preconditioner_makes_pcg_finish_without_an_iteration():
    from recycled_low_rank_gn import build_signed_relative_residual, pcg_solve

    system = torch.tensor(
        [[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64
    )
    base_diagonal = torch.tensor([3.0, 2.0], dtype=torch.float64)
    correction = build_signed_relative_residual(
        lambda vector: system @ vector,
        base_diagonal,
        seed=torch.tensor([1.0, 0.3], dtype=torch.float64),
        rank=2,
    )
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    def preconditioner(vector):
        return correction.apply(vector, base_diagonal)

    result = pcg_solve(
        lambda vector: system @ vector,
        rhs,
        preconditioner=preconditioner,
        maximum_iterations=3,
        relative_tolerance=1.0e-10,
        initial_direction=preconditioner(rhs),
    )

    assert result.iterations == 0
    assert result.relative_residual < 1.0e-10
    assert torch.allclose(result.direction, torch.linalg.solve(system, rhs))


def test_recycled_subspace_is_not_rebuilt_before_its_refresh_interval():
    from recycled_low_rank_gn import RecycledResidualState

    matrix = torch.tensor(
        [[3.0, 0.2], [0.2, 2.0]], dtype=torch.float64
    )
    matvec_calls = 0

    def matvec(vector):
        nonlocal matvec_calls
        matvec_calls += 1
        return matrix @ vector

    state = RecycledResidualState(rank=2, refresh_interval=4)
    base = torch.tensor([2.5, 1.5], dtype=torch.float64)
    seed = torch.tensor([1.0, 0.5], dtype=torch.float64)

    first, first_refreshed = state.get_or_build(
        outer_step=0,
        system_matvec=matvec,
        base_diagonal=base,
        seed=seed,
    )
    calls_after_first = matvec_calls
    reused, reused_refreshed = state.get_or_build(
        outer_step=3,
        system_matvec=matvec,
        base_diagonal=base,
        seed=seed,
    )
    refreshed, did_refresh = state.get_or_build(
        outer_step=4,
        system_matvec=matvec,
        base_diagonal=base,
        seed=seed,
    )

    assert first_refreshed
    assert not reused_refreshed
    assert reused is first
    assert matvec_calls == 2 * calls_after_first
    assert did_refresh
    assert refreshed is not first


def test_one_pass_secant_updates_recycle_a_bounded_low_rank_subspace():
    from recycled_low_rank_gn import RecycledResidualState

    system = torch.tensor(
        [[4.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 2.0]],
        dtype=torch.float64,
    )
    base = torch.tensor([3.0, 2.5, 1.5], dtype=torch.float64)
    state = RecycledResidualState(rank=2, refresh_interval=4)

    first, first_fit = state.update_from_secant(
        outer_step=0,
        base_diagonal=base,
        seed=torch.tensor([1.0, 0.2, -0.1], dtype=torch.float64),
        system_seed=system @ torch.tensor([1.0, 0.2, -0.1], dtype=torch.float64),
    )
    second, second_fit = state.update_from_secant(
        outer_step=1,
        base_diagonal=base,
        seed=torch.tensor([0.1, 1.0, 0.3], dtype=torch.float64),
        system_seed=system @ torch.tensor([0.1, 1.0, 0.3], dtype=torch.float64),
    )
    third, third_fit = state.update_from_secant(
        outer_step=2,
        base_diagonal=base,
        seed=torch.tensor([0.2, -0.1, 1.0], dtype=torch.float64),
        system_seed=system @ torch.tensor([0.2, -0.1, 1.0], dtype=torch.float64),
    )

    assert first.rank == 1
    assert second.rank == 2
    assert third.rank == 2
    assert all(0 <= fit <= 1 for fit in (first_fit, second_fit, third_fit))
    rhs = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    direction = third.apply(rhs, base)
    assert torch.dot(rhs, direction) > 0


def test_one_pass_secant_projects_an_inconsistent_residual_back_to_spd():
    from recycled_low_rank_gn import RecycledResidualState

    state = RecycledResidualState(
        rank=2,
        refresh_interval=1,
        minimum_relative_eigenvalue=0.25,
    )
    correction, _ = state.update_from_secant(
        outer_step=0,
        base_diagonal=torch.ones(2, dtype=torch.float64),
        seed=torch.tensor([1.0, 0.0], dtype=torch.float64),
        system_seed=torch.tensor([-10.0, 0.0], dtype=torch.float64),
    )

    eigenvalues = torch.linalg.eigvalsh(correction.projected_matrix)
    assert float(eigenvalues.min()) >= -0.75 - 1.0e-12


def test_pcg_rejects_a_non_positive_preconditioner():
    from recycled_low_rank_gn import pcg_solve

    with pytest.raises(RuntimeError, match="positive definite"):
        pcg_solve(
            lambda vector: vector,
            torch.ones(2, dtype=torch.float64),
            preconditioner=lambda vector: -vector,
            maximum_iterations=1,
        )
