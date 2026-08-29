import copy
import logging

import torch
from torch import nn

from baseline_kronecker_ggn import KroneckerGGN
from kronecker_ggn_common import CurvatureUpdate, KroneckerGGNConfig


def config():
    return KroneckerGGNConfig(
        curvature_mode="exact_ggn",
        damping=0.3,
        factor_decay=0.0,
        spectral_update_interval=1,
        linear_algebra_dtype="float64",
        fallback_learning_rate=0.01,
    )


def initialize(optimizer):
    activation = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    output = torch.tensor([[1.5, 0.1], [0.1, 0.8]], dtype=torch.float64)
    optimizer.update_curvature(
        lambda *_: CurvatureUpdate("exact_ggn", {"<root>": (activation, output)})
    )


def test_baseline_direction_matches_dense_solve():
    model = nn.Linear(2, 2, bias=False).double()
    optimizer = KroneckerGGN(model, config())
    initialize(optimizer)
    gradient = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64)
    model.weight.grad = gradient.clone()
    operator = optimizer.layer_state["<root>"].spectral
    identity = torch.eye(4, dtype=torch.float64)
    dense = torch.stack(
        [
            operator.matvec(identity[:, index].reshape(2, 2)).reshape(-1)
            for index in range(4)
        ],
        dim=1,
    )

    direction = optimizer.compute_direction().directions["weight"]

    assert torch.allclose(
        direction.reshape(-1),
        torch.linalg.solve(dense, -gradient.reshape(-1)),
        atol=1.0e-11,
    )


def test_state_round_trip_produces_same_next_update():
    first_model = nn.Linear(2, 2).double()
    second_model = nn.Linear(2, 2).double()
    second_model.load_state_dict(first_model.state_dict())
    first = KroneckerGGN(first_model, config())
    initialize(first)
    first_model.weight.grad = torch.full_like(first_model.weight, 0.2)
    first_model.bias.grad = torch.full_like(first_model.bias, -0.1)
    first.step()
    second_model.load_state_dict(first_model.state_dict())
    second = KroneckerGGN(second_model, config())
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    gradient = torch.tensor([[0.1, -0.4], [0.3, 0.2]], dtype=torch.float64)
    for model in (first_model, second_model):
        model.weight.grad = gradient.clone()
        model.bias.grad = torch.tensor([0.2, -0.3], dtype=torch.float64)

    first.step()
    second.step()

    assert torch.equal(first_model.weight, second_model.weight)
    assert torch.equal(first_model.bias, second_model.bias)


def test_lm_damping_decreases_when_ggn_quadratic_agrees_with_loss():
    model = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = KroneckerGGN(
        model,
        KroneckerGGNConfig(
            curvature_mode="exact_ggn",
            learning_rate=1.0,
            damping=0.3,
            factor_decay=0.0,
            spectral_update_interval=1,
            linear_algebra_dtype="float64",
            adaptive_damping=True,
        ),
    )
    optimizer.update_curvature(
        lambda *_: CurvatureUpdate(
            "exact_ggn",
            {"<root>": (torch.ones((1, 1), dtype=torch.float64),) * 2},
        )
    )
    model.weight.grad = torch.ones_like(model.weight)
    initial_damping = optimizer.layer_state["<root>"].damping

    optimizer.step(
        acceptance_closure=lambda: 0.5 * model.weight.square().sum()
    )

    assert optimizer.layer_state["<root>"].damping < initial_damping
    assert optimizer.get_metrics()["damping/reduction_ratio"] > 0.75


def test_unsupported_parameters_are_registered_and_logged(caplog):
    model = nn.Sequential(nn.Embedding(8, 2), nn.Linear(2, 2))

    with caplog.at_level(logging.WARNING):
        optimizer = KroneckerGGN(model, KroneckerGGNConfig())

    events = {event["parameter"]: event for event in optimizer.fallback_events}
    assert events["0.weight"]["reason"] == "embedding"
    assert events["1.bias"]["reason"] == "bias_or_vector"
    assert "Kronecker GGN fallback" in caplog.text
