import importlib.util
import inspect

import pytest
import torch


def test_conjugate_gradient_solves_damped_positive_system():
    assert importlib.util.find_spec("full_ggn") is not None
    import full_ggn

    assert hasattr(full_ggn, "conjugate_gradient")
    curvature = torch.diag(torch.tensor([2.0, 5.0], dtype=torch.float64))
    gradient = torch.tensor([4.0, -10.0], dtype=torch.float64)

    result = full_ggn.conjugate_gradient(
        lambda vector: curvature @ vector,
        -gradient,
        damping=1.0,
        maximum_iterations=4,
    )

    expected = torch.tensor([-4.0 / 3.0, 10.0 / 6.0], dtype=torch.float64)
    assert result.iterations == 2
    assert len(result.candidates) == 2
    assert torch.allclose(result.candidates[-1], result.direction, atol=1.0e-12)
    assert torch.allclose(result.direction, expected, atol=1.0e-12)
    assert result.residual_norm < 1.0e-12


def test_conjugate_gradient_uses_a_diagonal_preconditioner():
    import full_ggn

    assert "preconditioner" in inspect.signature(
        full_ggn.conjugate_gradient
    ).parameters
    curvature = torch.diag(torch.tensor([2.0, 100.0], dtype=torch.float64))
    right_hand_side = torch.tensor([6.0, 202.0], dtype=torch.float64)
    system_diagonal = torch.tensor([3.0, 101.0], dtype=torch.float64)

    result = full_ggn.conjugate_gradient(
        lambda vector: curvature @ vector,
        right_hand_side,
        damping=1.0,
        maximum_iterations=1,
        preconditioner=lambda residual: residual / system_diagonal,
    )

    assert torch.allclose(
        result.direction, torch.tensor([2.0, 2.0], dtype=torch.float64)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="checks CUDA candidate residency")
def test_conjugate_gradient_offloads_backtracking_candidates_from_cuda():
    import full_ggn

    right_hand_side = torch.ones(4, device="cuda")
    result = full_ggn.conjugate_gradient(
        lambda vector: vector,
        right_hand_side,
        damping=1.0,
        maximum_iterations=2,
    )

    assert result.direction.is_cuda
    assert result.candidates
    assert all(candidate.device.type == "cpu" for candidate in result.candidates)


def test_full_ggn_step_accepts_a_damped_descent_and_adapts_damping():
    import full_ggn
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch, GGNFullOperator

    assert hasattr(full_ggn, "FullGGNConfig")
    assert hasattr(full_ggn, "FullGGNState")
    assert hasattr(full_ggn, "full_ggn_step")
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    targets = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
    loss = lambda: 0.5 * (model(inputs) - targets).square().mean()
    operator = GGNFullOperator(model, FunctionalCurvatureBatch((inputs,), lambda output: 0.5 * (output - targets).square().mean()))
    state = full_ggn.FullGGNState(damping=1.0)

    result = full_ggn.full_ggn_step(
        operator,
        loss,
        state=state,
        config=full_ggn.FullGGNConfig(maximum_cg_iterations=4),
    )

    assert result.accepted
    assert result.final_loss < result.initial_loss
    assert 0 < result.reduction_ratio
    assert state.damping < 1.0


def test_full_ggn_step_can_try_a_larger_scale_and_backtrack_safely():
    import full_ggn
    from kronecker_ggn_common.curvature_operator import (
        FunctionalCurvatureBatch,
        GGNFullOperator,
    )

    assert "initial_step_scale" in full_ggn.FullGGNConfig.__dataclass_fields__
    model = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    targets = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
    loss = lambda: 0.5 * (model(inputs) - targets).square().mean()
    operator = GGNFullOperator(
        model,
        FunctionalCurvatureBatch(
            (inputs,), lambda output: 0.5 * (output - targets).square().mean()
        ),
    )

    result = full_ggn.full_ggn_step(
        operator,
        loss,
        state=full_ggn.FullGGNState(damping=1.0),
        config=full_ggn.FullGGNConfig(
            maximum_cg_iterations=4, initial_step_scale=2.0
        ),
    )

    assert result.accepted
    assert result.step_scale == 2.0
    assert result.final_loss < result.initial_loss


def test_full_ggn_step_predicts_reduction_at_the_original_parameters():
    import full_ggn

    class ParameterDependentOperator:
        def __init__(self) -> None:
            self.parameter = torch.nn.Parameter(torch.tensor([0.0]))
            self.parameters = (self.parameter,)
            self._sizes = (1,)

        def gradient(self) -> torch.Tensor:
            return torch.tensor([-1.0])

        def matvec(self, vector: torch.Tensor) -> torch.Tensor:
            return (1.0 + 10.0 * self.parameter.detach()) * vector

    operator = ParameterDependentOperator()
    loss = lambda: 0.5 * (operator.parameter - 1.0).square().sum()

    result = full_ggn.full_ggn_step(
        operator,
        loss,
        state=full_ggn.FullGGNState(damping=1.0),
        config=full_ggn.FullGGNConfig(maximum_cg_iterations=1),
    )

    assert result.accepted
    assert result.predicted_reduction == pytest.approx(0.375)


def test_averaged_full_ggn_operator_averages_gradient_and_curvature_products():
    import kronecker_ggn_common.curvature_operator as curvature_operator
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch, GGNFullOperator

    assert hasattr(curvature_operator, "AveragedGGNOperator")
    model = torch.nn.Linear(1, 1, bias=False).double()
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
    averaged = curvature_operator.AveragedGGNOperator((first, second))
    vector = torch.tensor([0.25], dtype=torch.float64)

    assert torch.allclose(averaged.gradient(), 0.5 * (first.gradient() + second.gradient()))
    assert torch.allclose(averaged.matvec(vector), 0.5 * (first.matvec(vector) + second.matvec(vector)))


def test_averaged_full_ggn_operator_returns_gradient_second_moment():
    import kronecker_ggn_common.curvature_operator as curvature_operator
    from kronecker_ggn_common.curvature_operator import (
        FunctionalCurvatureBatch,
        GGNFullOperator,
    )

    assert hasattr(curvature_operator.AveragedGGNOperator, "gradient_statistics")
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

    gradient, second_moment = curvature_operator.AveragedGGNOperator(
        (first, second)
    ).gradient_statistics()

    assert torch.allclose(gradient, torch.tensor([-3.5], dtype=torch.float64))
    assert torch.allclose(
        second_moment, torch.tensor([18.5], dtype=torch.float64)
    )


def test_split_batch_full_ggn_uses_large_gradient_batch_and_small_fixed_curvature_batch():
    import kronecker_ggn_common.curvature_operator as curvature_operator
    from kronecker_ggn_common.curvature_operator import FunctionalCurvatureBatch, GGNFullOperator

    assert hasattr(curvature_operator, "SplitBatchGGNOperator")
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
    operator = curvature_operator.SplitBatchGGNOperator(
        gradient_operator=curvature_operator.AveragedGGNOperator((first, second)),
        curvature_operator=curvature_operator.AveragedGGNOperator((first,)),
    )

    assert torch.allclose(
        operator.gradient(), torch.tensor([-3.5], dtype=torch.float64)
    )
    assert torch.allclose(
        operator.matvec(torch.tensor([0.25], dtype=torch.float64)),
        torch.tensor([0.25], dtype=torch.float64),
    )
