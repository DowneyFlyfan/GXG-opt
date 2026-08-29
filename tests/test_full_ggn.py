import importlib.util

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
