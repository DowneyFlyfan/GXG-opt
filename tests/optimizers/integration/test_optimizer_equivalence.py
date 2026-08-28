import copy
from pathlib import Path

import torch
from torch import nn

from baseline_kronecker_ggn import KroneckerGGN
from kronecker_ggn_common import (
    CurvatureUpdate,
    KroneckerGGNConfig,
    LowRankCorrectedKroneckerGGNConfig,
)
from low_rank_corrected_kronecker_ggn import LowRankCorrectedKroneckerGGN
from optimizers import build_optimizer


def factors():
    return (
        torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64),
        torch.tensor([[1.5, 0.1], [0.1, 0.8]], dtype=torch.float64),
    )


def baseline_config():
    return KroneckerGGNConfig(
        curvature_mode="exact_ggn",
        damping=0.3,
        factor_decay=0.0,
        spectral_update_interval=1,
        linear_algebra_dtype="float64",
    )


def corrected_config(rank=0, memory_mb=16):
    return LowRankCorrectedKroneckerGGNConfig(
        curvature_mode="exact_ggn",
        damping=0.3,
        factor_decay=0.0,
        spectral_update_interval=1,
        linear_algebra_dtype="float64",
        correction_dtype="float64",
        correction_rank=rank,
        correction_warmup_steps=0,
        correction_refresh_interval=1,
        correction_max_age=10,
        correction_memory_budget_mb=memory_mb,
        active_layer_count=1,
    )


def test_rank_zero_matches_baseline_update_and_trajectory_exactly():
    baseline_model = nn.Linear(2, 2, bias=False).double()
    corrected_model = copy.deepcopy(baseline_model)
    baseline = KroneckerGGN(baseline_model, baseline_config())
    corrected = LowRankCorrectedKroneckerGGN(corrected_model, corrected_config(rank=0))
    update = lambda *_: CurvatureUpdate("exact_ggn", {"<root>": factors()})
    baseline.update_curvature(update)
    corrected.update_curvature(update)

    for gradient in (
        torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64),
        torch.tensor([[-0.3, 0.4], [0.1, -0.2]], dtype=torch.float64),
        torch.tensor([[0.1, 0.2], [-0.4, 0.3]], dtype=torch.float64),
    ):
        baseline_model.weight.grad = gradient.clone()
        corrected_model.weight.grad = gradient.clone()
        baseline.step()
        corrected.step()

    assert torch.equal(baseline_model.weight, corrected_model.weight)


def test_exact_rank_one_mismatch_recovers_dense_gn_direction_online():
    model = nn.Linear(2, 2, bias=False).double()
    optimizer = LowRankCorrectedKroneckerGGN(
        model,
        corrected_config(rank=1),
    )
    activation, output = factors()
    # Initialize shared M before defining the exact rank-one GGN operator.
    optimizer.update_curvature(
        lambda *_: CurvatureUpdate("exact_ggn", {"<root>": (activation, output)})
    )
    kron = optimizer.layer_state["<root>"].spectral
    identity = torch.eye(4, dtype=torch.float64)
    square_root = torch.stack(
        [
            kron.apply_sqrt(identity[:, index].reshape(2, 2)).reshape(-1)
            for index in range(4)
        ],
        dim=1,
    )
    basis = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    mismatch_value = torch.tensor(1.5, dtype=torch.float64)
    hessian = (
        square_root
        @ (identity + mismatch_value * basis[:, None] @ basis[None, :])
        @ square_root
    )
    ggn = hessian - optimizer.config.damping * identity

    class DenseGGN:
        def matvec(self, _layer_id, value):
            return (ggn @ value.reshape(-1)).reshape(2, 2)

    optimizer.update_curvature(
        lambda *_: CurvatureUpdate(
            "exact_ggn", {"<root>": (activation, output)}, {"<root>": DenseGGN()}
        )
    )
    gradient = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64)
    model.weight.grad = gradient.clone()

    direction = optimizer.compute_direction().directions["weight"]

    assert optimizer.correction_state["<root>"].accepted_rank == 1
    assert torch.allclose(
        direction.reshape(-1),
        torch.linalg.solve(hessian, -gradient.reshape(-1)),
        atol=1.0e-9,
    )


def test_invalid_or_over_budget_correction_falls_back_to_baseline():
    baseline_model = nn.Linear(2, 2, bias=False).double()
    corrected_model = copy.deepcopy(baseline_model)
    baseline = KroneckerGGN(baseline_model, baseline_config())
    corrected = LowRankCorrectedKroneckerGGN(
        corrected_model, corrected_config(rank=2, memory_mb=1.0e-6)
    )
    update = lambda *_: CurvatureUpdate("exact_ggn", {"<root>": factors()})
    baseline.update_curvature(update)
    corrected.update_curvature(update)
    gradient = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64)
    baseline_model.weight.grad = gradient.clone()
    corrected_model.weight.grad = gradient.clone()

    baseline_direction = baseline.compute_direction().directions["weight"]
    corrected_direction = corrected.compute_direction().directions["weight"]

    assert torch.equal(baseline_direction, corrected_direction)
    assert not corrected.correction_state["<root>"].valid


def test_zero_mismatch_and_corrected_checkpoint_resume_match_baseline():
    first_model = nn.Linear(2, 2, bias=False).double()
    second_model = copy.deepcopy(first_model)
    first = LowRankCorrectedKroneckerGGN(first_model, corrected_config(rank=1))
    second = LowRankCorrectedKroneckerGGN(second_model, corrected_config(rank=1))
    activation, output = factors()
    first.update_curvature(
        lambda *_: CurvatureUpdate("exact_ggn", {"<root>": (activation, output)})
    )
    spectral = first.layer_state["<root>"].spectral

    class MatchingGGN:
        def matvec(self, _layer_id, value):
            return spectral.matvec(value) - first.config.damping * value

    first.update_curvature(
        lambda *_: CurvatureUpdate(
            "exact_ggn",
            {"<root>": (activation, output)},
            {"<root>": MatchingGGN()},
        )
    )
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    gradient = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64)
    first_model.weight.grad = gradient.clone()
    second_model.weight.grad = gradient.clone()

    first_direction = first.compute_direction().directions["weight"]
    second_direction = second.compute_direction().directions["weight"]
    baseline = first.baseline_direction_for_layer("<root>", gradient)

    assert torch.allclose(first_direction, baseline, atol=1.0e-11)
    assert torch.equal(first_direction, second_direction)
    assert first.correction_state["<root>"].basis.grad_fn is None


def test_checked_in_configs_build_through_repo_registry():
    root = Path(__file__).resolve().parents[3]
    model = nn.Linear(2, 2, bias=False)
    baseline = build_optimizer(
        "kronecker_ggn",
        model,
        config=KroneckerGGNConfig.from_yaml(
            root / "configs/optimizer/kronecker_ggn.yaml"
        ),
    )
    corrected = build_optimizer(
        "low_rank_corrected_kronecker_ggn",
        copy.deepcopy(model),
        config=LowRankCorrectedKroneckerGGNConfig.from_yaml(
            root / "configs/optimizer/low_rank_corrected_kronecker_ggn.yaml"
        ),
    )

    assert isinstance(baseline, KroneckerGGN)
    assert isinstance(corrected, LowRankCorrectedKroneckerGGN)


@torch.no_grad()
def test_cuda_fp32_spectral_smoke_when_available():
    if not torch.cuda.is_available():
        return
    activation = torch.eye(4, device="cuda")
    output = torch.eye(3, device="cuda")
    from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator

    operator = KroneckerSpectralOperator(activation, output, 0.1)
    value = torch.randn(3, 4, device="cuda")
    recovered = operator.apply_inverse(operator.matvec(value))
    assert torch.allclose(recovered, value, atol=1.0e-5, rtol=1.0e-5)
