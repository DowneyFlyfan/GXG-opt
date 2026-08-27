from dataclasses import replace

import pytest
import torch
from torch import nn

from gn_guided_adam import FunctionalBatch, GNGuidedAdamW, GuidedAdamConfig, GuidedStepContext
from gn_guided_adam.config import AdamWConfig, FixedEpochDutyCycleConfig, GNConfig


def baseline_config():
    return GuidedAdamConfig(
        adamw=AdamWConfig(lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0),
        gn=replace(GNConfig(), enabled=False, min_block_numel=1),
    )


def guided_config(**changes):
    values = {
        "warmup_steps": 0,
        "rank": 1,
        "refresh_interval": 10,
        "min_block_numel": 1,
        "trust_radius": 100.0,
        "max_relative_block_update": 100.0,
        "max_basis_age": 10,
        "max_parameter_drift": 10.0,
        "rho_min": 0.0,
    }
    values.update(changes)
    gn = replace(GNConfig(), **values)
    return GuidedAdamConfig(
        adamw=AdamWConfig(lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0),
        gn=gn,
    )


def assign_gradient(model, gradient):
    for name, parameter in model.named_parameters():
        parameter.grad = gradient[name].clone()


def test_disabled_guidance_matches_torch_adamw_across_steps():
    ours_model = nn.Linear(2, 1, bias=False)
    baseline_model = nn.Linear(2, 1, bias=False)
    baseline_model.load_state_dict(ours_model.state_dict())
    ours = GNGuidedAdamW(ours_model, baseline_config())
    baseline = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )

    for scale in (1.0, 0.5, -0.25):
        gradient = scale * torch.tensor([[0.2, -0.4]])
        ours_model.weight.grad = gradient.clone()
        baseline_model.weight.grad = gradient.clone()
        result = ours.step()
        baseline.step()
        baseline.zero_grad(set_to_none=True)
        assert not result.guidance_used

    assert torch.allclose(ours_model.weight, baseline_model.weight, atol=1.0e-7, rtol=1.0e-6)


def test_disabled_guidance_matches_adamw_when_a_parameter_has_no_gradient():
    ours_model = nn.Sequential(
        nn.Linear(1, 1, bias=False),
        nn.Linear(1, 1, bias=False),
    )
    baseline_model = nn.Sequential(
        nn.Linear(1, 1, bias=False),
        nn.Linear(1, 1, bias=False),
    )
    baseline_model.load_state_dict(ours_model.state_dict())
    ours = GNGuidedAdamW(ours_model, baseline_config())
    baseline = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    ours_before = ours_model[1].weight.detach().clone()
    baseline_before = baseline_model[1].weight.detach().clone()
    ours_model[0].weight.grad = torch.tensor([[0.25]])
    baseline_model[0].weight.grad = torch.tensor([[0.25]])

    ours.step()
    baseline.step()

    assert torch.allclose(ours_model[0].weight, baseline_model[0].weight, atol=1.0e-7)
    assert torch.equal(ours_model[1].weight, ours_before)
    assert torch.equal(baseline_model[1].weight, baseline_before)
    assert ours.adam.state["1.weight"]["step"] == 0


def test_fixed_refresh_accepts_useful_guidance_and_keeps_adam_moments_gradient_only():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = GNGuidedAdamW(model, guided_config())
    inputs = torch.tensor([[1.0], [2.0]])
    targets = 2 * inputs
    loss = 0.5 * (model(inputs) - targets).square().sum()
    loss.backward()
    real_gradient = model.weight.grad.detach().clone()
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum(), batch_id="independent")

    result = optimizer.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch, tokens=2))

    assert result.refreshed
    assert result.guidance_used and result.guidance_accepted
    assert model.weight.item() > 1.5
    expected_m = (1 - optimizer.config.adamw.betas[0]) * real_gradient
    assert torch.allclose(optimizer.adam.state["weight"]["m"], expected_m)
    assert optimizer.guidance["weight"].basis is not None
    assert result.measurements["block/weight/orthogonality_error"] < 1.0e-6


def test_fixed_epoch_duty_cycle_switches_off_and_refreshes_on_reactivation():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    base = guided_config(refresh_interval=100)
    config = replace(
        base,
        fixed_epoch_duty_cycle=FixedEpochDutyCycleConfig(
            enabled=True,
            start_epoch=0,
            on_epochs=1,
            off_epochs=1,
            refresh_on_activation=True,
        ),
    )
    optimizer = GNGuidedAdamW(model, config)
    inputs = torch.tensor([[1.0], [2.0]])
    targets = 2 * inputs
    batch = FunctionalBatch(
        (inputs,),
        lambda output: 0.5 * (output - targets).square().sum(),
        batch_id="fixed-duty-cycle",
    )

    def train_step(epoch, with_batches):
        optimizer.zero_grad(set_to_none=True)
        (0.5 * (model(inputs) - targets).square().sum()).backward()
        return optimizer.step(
            GuidedStepContext(
                curvature_batch=batch if with_batches else None,
                acceptance_batch=batch if with_batches else None,
                epoch=epoch,
            )
        )

    first = train_step(epoch=0, with_batches=True)
    off = train_step(epoch=1, with_batches=False)

    assert first.refreshed and first.guidance_accepted
    assert not off.refreshed and not off.guidance_used
    assert off.fallback_reason == "fixed_epoch_duty_cycle_off"
    assert off.measurements["schedule/fixed_duty_cycle_active"] == 0.0
    schedule_state = optimizer.state_dict()
    assert schedule_state["last_epoch"] == 1
    assert schedule_state["last_duty_active"] is False
    reactivated = train_step(epoch=2, with_batches=True)
    assert reactivated.refreshed
    assert reactivated.measurements["schedule/fixed_duty_cycle_active"] == 1.0


def test_rejected_refresh_applies_exact_adam_candidate_and_invalidates_basis():
    ours_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ours_model.weight.fill_(1.0)
    baseline_model = nn.Linear(1, 1, bias=False)
    baseline_model.load_state_dict(ours_model.state_dict())
    config = guided_config(acceptance_margin=1.0e6)
    ours = GNGuidedAdamW(ours_model, config)
    baseline = torch.optim.AdamW(
        baseline_model.parameters(),
        lr=0.01,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    inputs = torch.tensor([[1.0], [2.0]])
    targets = 2 * inputs
    ours_loss = 0.5 * (ours_model(inputs) - targets).square().sum()
    baseline_loss = 0.5 * (baseline_model(inputs) - targets).square().sum()
    ours_loss.backward()
    baseline_loss.backward()
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum())

    result = ours.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch))
    baseline.step()

    assert not result.guidance_accepted
    assert result.fallback_reason == "adam_candidate_better"
    assert torch.allclose(ours_model.weight, baseline_model.weight, atol=1.0e-7)
    baseline_state = baseline.state[baseline_model.weight]
    assert torch.allclose(ours.adam.state["weight"]["m"], baseline_state["exp_avg"])
    assert torch.allclose(ours.adam.state["weight"]["v"], baseline_state["exp_avg_sq"])
    assert ours.guidance["weight"].basis is None
    assert ours.guidance["weight"].damping > config.gn.initial_damping


def test_nonfinite_curvature_fails_closed_to_adamw():
    ours_model = nn.Linear(1, 1, bias=False)
    baseline_model = nn.Linear(1, 1, bias=False)
    baseline_model.load_state_dict(ours_model.state_dict())
    ours = GNGuidedAdamW(ours_model, guided_config())
    baseline = torch.optim.AdamW(
        baseline_model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0
    )
    gradient = torch.tensor([[0.5]])
    ours_model.weight.grad = gradient.clone()
    baseline_model.weight.grad = gradient.clone()
    batch = FunctionalBatch((torch.ones(1, 1),), lambda output: output.sum() * torch.tensor(float("nan")))

    result = ours.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch))
    baseline.step()

    assert not result.guidance_accepted
    assert result.fallback_reason == "all_curvature_builds_failed"
    assert torch.allclose(ours_model.weight, baseline_model.weight, atol=1.0e-7)
    assert torch.isfinite(ours_model.weight).all()


def test_curvature_exception_injection_falls_back_without_partial_parameter_update(monkeypatch):
    ours_model = nn.Linear(1, 1, bias=False)
    baseline_model = nn.Linear(1, 1, bias=False)
    baseline_model.load_state_dict(ours_model.state_dict())
    ours = GNGuidedAdamW(ours_model, guided_config())
    baseline = torch.optim.AdamW(
        baseline_model.parameters(), lr=0.01, betas=(0.9, 0.99), eps=1.0e-8, weight_decay=0.0
    )
    gradient = torch.tensor([[0.5]])
    ours_model.weight.grad = gradient.clone()
    baseline_model.weight.grad = gradient.clone()
    batch = FunctionalBatch((torch.ones(1, 1),), lambda output: output.square().sum())

    def fail_operator(*args, **kwargs):
        raise RuntimeError("injected out of memory")

    monkeypatch.setattr("gn_guided_adam.optimizer.GGNBlockOperator", fail_operator)
    result = ours.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch))
    baseline.step()

    assert result.fallback_reason == "all_curvature_builds_failed"
    assert torch.allclose(ours_model.weight, baseline_model.weight, atol=1.0e-7)


def test_repeated_refresh_failures_activate_fixed_cooldown():
    config = guided_config(refresh_interval=1, failures_before_cooldown=2, failure_cooldown_steps=5)
    model = nn.Linear(1, 1, bias=False)
    optimizer = GNGuidedAdamW(model, config)
    batch = FunctionalBatch((torch.ones(1, 1),), lambda output: output.sum() * torch.tensor(float("nan")))

    for _ in range(2):
        model.weight.grad = torch.ones_like(model.weight)
        optimizer.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch))

    assert optimizer.guidance["weight"].cooldown_until > optimizer.step_count
    assert optimizer.get_metrics()["cooldown_block_count"] == 1.0


def test_stale_basis_reuse_keeps_adam_and_guidance_components_orthogonal():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.5]]))
    optimizer = GNGuidedAdamW(model, guided_config(rank=1))
    inputs = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    targets = torch.tensor([[2.0], [2.0]])
    (0.5 * (model(inputs) - targets).square().sum()).backward()
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum())
    first = optimizer.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch))
    assert first.guidance_accepted
    optimizer.zero_grad(set_to_none=True)
    (0.5 * (model(inputs) - targets).square().sum()).backward()
    gradient = model.weight.grad.detach().clone()
    optimizer.adam.update({"weight": gradient})
    basis = optimizer.guidance["weight"].basis
    complement = optimizer.adam.candidate_for("weight", optimizer.lr, basis).reshape(-1)

    assert torch.norm(basis.T @ complement).item() < 1.0e-6


def test_checkpoint_resume_restores_basis_and_produces_same_next_update(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = GNGuidedAdamW(model, guided_config())
    inputs = torch.tensor([[1.0], [2.0]])
    targets = 2 * inputs
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum())
    (0.5 * (model(inputs) - targets).square().sum()).backward()
    optimizer.step(GuidedStepContext(curvature_batch=batch, acceptance_batch=batch))
    optimizer.zero_grad(set_to_none=True)
    path = optimizer.save_checkpoint(tmp_path / "guided.pt", extra={"tag": "resume"})

    resumed_model = nn.Linear(1, 1, bias=False)
    resumed = GNGuidedAdamW(resumed_model, guided_config())
    assert resumed.load_checkpoint(path) == {"tag": "resume"}
    assert torch.equal(resumed.guidance["weight"].basis, optimizer.guidance["weight"].basis)

    (0.5 * (model(inputs) - targets).square().sum()).backward()
    (0.5 * (resumed_model(inputs) - targets).square().sum()).backward()
    first = optimizer.step()
    second = resumed.step()

    assert first.guidance_accepted and second.guidance_accepted
    assert torch.allclose(model.weight, resumed_model.weight, atol=1.0e-7)
    assert optimizer.step_count == resumed.step_count


def test_metrics_write_machine_readable_jsonl(tmp_path):
    model = nn.Linear(1, 1, bias=False)
    optimizer = GNGuidedAdamW(model, baseline_config())
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step(GuidedStepContext(tokens=8))

    path = optimizer.metrics.write_jsonl(tmp_path / "optimizer.jsonl")

    assert path.read_text(encoding="utf-8").count("\n") == 1
    assert optimizer.get_metrics()["step"] == 1.0


def test_nonfinite_real_gradient_is_rejected_before_parameter_update():
    model = nn.Linear(1, 1, bias=False)
    optimizer = GNGuidedAdamW(model, baseline_config())
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))

    with pytest.raises(FloatingPointError, match="Nonfinite real gradient"):
        optimizer.step()

    assert torch.equal(model.weight, before)


def test_nonfinite_gradient_rejection_does_not_partially_advance_adam_state():
    model = nn.Sequential(
        nn.Linear(1, 1, bias=False),
        nn.Linear(1, 1, bias=False),
    )
    optimizer = GNGuidedAdamW(model, baseline_config())
    model[0].weight.grad = torch.ones_like(model[0].weight)
    model[1].weight.grad = torch.full_like(model[1].weight, float("nan"))

    with pytest.raises(FloatingPointError, match="Nonfinite real gradient"):
        optimizer.step()

    for state in optimizer.adam.state.values():
        assert state["step"] == 0
        assert torch.count_nonzero(state["m"]) == 0
        assert torch.count_nonzero(state["v"]) == 0
