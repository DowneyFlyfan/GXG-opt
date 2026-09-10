from dataclasses import replace

import pytest
import torch
from torch import nn

from gxg_opt_adam.adam_backend import AdamBackend
from gxg_opt_adam.bridge import blend_candidates, bridge_weight, secant_scale
from gxg_opt_adam.config import AdamConfig, BridgeConfig, GNConfig
from gxg_opt_adam.gn_inner_solvers import solve_quadratic
from gxg_opt_adam.layer_partition import LayerGroup
from gxg_opt_adam.layerwise_gn import LayerwiseGN, layer_gradient_and_gvp
from gxg_opt_adam.line_search import global_line_search
from gxg_opt_adam.probes import cosine, gain_per_second, gradient_noise_ratio, reduction_ratio, transfer_ratio
from gxg_opt_adam.types import FunctionalBatch


def test_gn_vector_product_matches_explicit_tiny_matrix_in_double_precision():
    model = nn.Linear(2, 1, bias=False).double()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    targets = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum())
    vector = {"weight": torch.tensor([[0.2, -0.1]], dtype=torch.float64)}

    _, product = layer_gradient_and_gvp(model, batch, ("weight",), vector)
    explicit = (inputs.T @ inputs @ vector["weight"].T).T

    assert product is not None
    assert torch.allclose(product["weight"], explicit, atol=1.0e-10, rtol=1.0e-10)


def test_gn_product_excludes_model_output_second_derivatives():
    class NonlinearOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, value):
            return value * self.weight.square()

    model = NonlinearOutput()
    batch = FunctionalBatch(
        (torch.tensor(1.0, dtype=torch.float64),),
        lambda output: 0.5 * output.square(),
    )
    _, product = layer_gradient_and_gvp(
        model,
        batch,
        ("weight",),
        {"weight": torch.tensor(1.0, dtype=torch.float64)},
    )

    assert product is not None
    assert product["weight"].item() == pytest.approx(4.0)
    assert product["weight"].item() != pytest.approx(6.0)  # Full Hessian value.


def test_positive_damping_cg_produces_finite_descent_candidate():
    gradient = {"weight": torch.tensor([2.0, -1.0], dtype=torch.float64)}
    damping = 0.5

    def operator(value):
        return {"weight": (torch.tensor([[3.0, 0.5], [0.5, 2.0]], dtype=torch.float64) @ value["weight"]) + damping * value["weight"]}

    config = replace(
        GNConfig(),
        inner_optimizer_matrix="cg",
        inner_optimizer_vector="cg",
        inner_steps=8,
        relative_residual_tolerance=1.0e-10,
    )
    result = solve_quadratic(gradient, operator, {}, {}, set(), config)

    assert result.finite
    assert result.descent
    assert result.predicted_reduction > 0
    assert result.residual_norm < 1.0e-8


def test_layer_proposals_merge_as_union_of_disjoint_parameters():
    class TwoBranches(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Linear(1, 1, bias=False)
            self.right = nn.Linear(1, 1, bias=False)

        def forward(self, value):
            return self.left(value) + self.right(value)

    model = TwoBranches()
    groups = (LayerGroup("left", ("left.weight",)), LayerGroup("right", ("right.weight",)))
    config = replace(GNConfig(), inner_optimizer_matrix="cg", inner_optimizer_vector="cg", inner_steps=4)
    backend = LayerwiseGN(model, groups, config)
    inputs = torch.tensor([[1.0], [2.0]])
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * output.square().sum())

    proposal = backend.propose(batch)

    assert set(proposal.direction) == {"left.weight", "right.weight"}
    assert set(proposal.layer_results) == {"left", "right"}


def test_line_search_accepts_descent_and_noop_rejects_harmful_direction():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    closure = lambda: (model.weight - 1.0).square().sum().float()

    accepted = global_line_search(model, {"weight": torch.ones_like(model.weight)}, closure, (1.0, 0.5, 0.0), 1.0)
    assert accepted.accepted
    assert accepted.alpha == 1.0
    assert model.weight.item() == pytest.approx(1.0)

    rejected = global_line_search(model, {"weight": torch.ones_like(model.weight)}, closure, (1.0, 0.0), 1.0)
    assert not rejected.accepted
    assert rejected.alpha == 0.0
    assert model.weight.item() == pytest.approx(1.0)


def test_line_search_restores_parameters_when_reference_closure_fails():
    model = nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("reference failure")
        return model.weight.square().sum()

    with pytest.raises(RuntimeError, match="reference failure"):
        global_line_search(model, {"weight": torch.ones_like(model.weight)}, closure, (1.0, 0.0), 1.0)

    assert torch.equal(model.weight, before)


def test_line_search_rejects_nonfinite_reference_loss_without_mutation():
    model = nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()

    result = global_line_search(
        model,
        {"weight": torch.ones_like(model.weight)},
        lambda: torch.tensor(float("nan")),
        (1.0, 0.0),
        1.0,
    )

    assert not result.accepted
    assert not result.finite
    assert torch.equal(model.weight, before)


def test_probe_formulas_are_exact_on_simple_vectors():
    first = {"p": torch.tensor([1.0, 0.0])}
    second = {"p": torch.tensor([0.0, 1.0])}

    assert gradient_noise_ratio(first, second) == pytest.approx(2.0)
    assert cosine(first, second) == pytest.approx(0.0)
    assert reduction_ratio(2.0, 4.0) == pytest.approx(0.5)
    assert transfer_ratio(1.0, 2.0) == pytest.approx(0.5)
    assert gain_per_second(3.0, 2.0) == pytest.approx(1.5)


def test_shadow_adam_uses_mean_microbatch_squares_and_variance_floor():
    model = nn.Linear(1, 1, bias=False)
    group = (LayerGroup("root", ("weight",)),)
    backend = AdamBackend(model, group, AdamConfig(lr=0.1, betas=(0.5, 0.5), variance_floor_ratio=1.0))
    microbatches = ({"weight": torch.ones_like(model.weight)}, {"weight": 3 * torch.ones_like(model.weight)})

    backend.update_moments({"weight": 2 * torch.ones_like(model.weight)}, microbatches)
    assert backend.state["weight"]["v"].item() == pytest.approx(2.5)

    backend.capture_variance_floor()
    backend.state["weight"]["v"].zero_()
    assert torch.isfinite(backend.candidate()["weight"]).all()
    assert backend.candidate()["weight"].abs().item() < 1.0


def test_shadow_adam_uses_effective_decay_for_equivalent_batches():
    model = nn.Linear(1, 1, bias=False)
    backend = AdamBackend(
        model,
        (LayerGroup("root", ("weight",)),),
        AdamConfig(betas=(0.5, 0.5)),
    )
    gradient = {"weight": 2 * torch.ones_like(model.weight)}

    backend.update_moments(gradient, adam_equivalent_batches=2.0)

    assert backend.state["weight"]["m"].item() == pytest.approx(1.5)
    assert backend.state["weight"]["v"].item() == pytest.approx(3.0)
    assert backend.state["weight"]["step"] == pytest.approx(2.0)


def test_adam_only_backend_matches_torch_adamw_without_weight_decay():
    ours_model = nn.Linear(2, 1, bias=False)
    torch_model = nn.Linear(2, 1, bias=False)
    torch_model.load_state_dict(ours_model.state_dict())
    group = (LayerGroup("root", ("weight",)),)
    config = AdamConfig(lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0)
    ours = AdamBackend(ours_model, group, config)
    baseline = torch.optim.AdamW(torch_model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0)

    for scale in (1.0, 0.5, -0.25):
        gradient = scale * torch.tensor([[0.2, -0.4]])
        ours_model.weight.grad = gradient.clone()
        torch_model.weight.grad = gradient.clone()
        ours.step()
        baseline.step()
        baseline.zero_grad(set_to_none=True)

    assert torch.allclose(ours_model.weight, torch_model.weight, atol=1.0e-7, rtol=1.0e-6)


def test_bridge_schedule_norm_matching_descent_guard_and_secant_scale():
    config = BridgeConfig(length_adam_steps=4, rho_start=0.5, rho_end=0.0, schedule="linear")
    group = (LayerGroup("root", ("weight",)),)
    adam = {"weight": torch.tensor([-1.0, 0.0])}
    gn = {"weight": torch.tensor([0.0, -10.0])}
    reference = {"weight": torch.tensor([1.0, 1.0])}

    result = blend_candidates(gn, adam, group, reference, config, 0, {"root": 1.0})

    assert result.accepted and result.used_gn
    assert result.rho == pytest.approx(0.5)
    assert bridge_weight(config, 0) == pytest.approx(0.5)
    assert bridge_weight(config, 3) == pytest.approx(0.0)
    assert bridge_weight(config, 4) == pytest.approx(0.0)
    assert secant_scale({"weight": torch.tensor([2.0])}, {"weight": torch.tensor([1.0])}, 1.0e-12) == pytest.approx(2.0)

    guarded = blend_candidates(
        {"weight": torch.tensor([10.0, 0.0])},
        adam,
        group,
        {"weight": torch.tensor([1.0, 0.0])},
        replace(config, per_layer_norm_match=False, rho_start=1.0),
        0,
        {"root": None},
    )
    assert guarded.accepted
    assert not guarded.used_gn
    assert guarded.reason == "gn_descent_guard"
