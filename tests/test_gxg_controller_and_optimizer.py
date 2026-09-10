from dataclasses import replace

import pytest
import torch
from torch import nn

from gxg_opt_adam import EvalContext, FunctionalBatch, GXGConfig, GXGOptimizer, GXGPhase, StepContext, gxg_optimizer
from gxg_opt_adam.config import AdamConfig, BridgeConfig, DutyCycleConfig, FinalMileConfig, GNConfig
from gxg_opt_adam.controller import GXGController


def duty_config(**duty_changes):
    return GXGConfig(
        duty_cycle=DutyCycleConfig(gn_epochs=1, adam_epochs=2, **duty_changes),
        gn=replace(GNConfig(), inner_optimizer_matrix="cg", inner_optimizer_vector="cg", inner_steps=5),
    )


def test_fixed_epoch_duty_cycle_is_the_only_normal_switch_signal():
    controller = GXGController(duty_config())

    assert controller.desired_optimizer(0) == "gn"
    assert controller.route_epoch(0) is None
    assert controller.route_epoch(1) == "fixed_duty_cycle_enter_adam"
    assert controller.state.phase == GXGPhase.BRIDGE_TO_ADAM
    controller.bridge_complete()
    assert controller.route_epoch(2) is None
    assert controller.state.phase == GXGPhase.ADAM
    assert controller.route_epoch(3) == "fixed_duty_cycle_enter_gn"
    assert controller.state.phase == GXGPhase.GN_CORRECTION
    assert controller.state.switch_count == 2


def test_epoch_must_be_monotonic():
    controller = GXGController(duty_config())
    controller.route_epoch(2)

    with pytest.raises(ValueError, match="monotonic"):
        controller.route_epoch(1)


def test_gn_hard_failure_suppresses_only_the_current_epoch():
    controller = GXGController(duty_config())
    controller.route_epoch(0)

    reason = controller.after_gn(
        accepted=False,
        reduction_ratio=0.0,
        transfer_ratio=-1.0,
        finite=True,
        update_norm_safe=True,
    )

    assert reason == "gn_safety_exit_for_epoch"
    assert controller.desired_optimizer(0) == "adam"
    assert controller.desired_optimizer(3) == "gn"


@pytest.mark.parametrize("phase", list(GXGPhase))
def test_controller_state_round_trips_every_serializable_phase(phase):
    first = GXGController(duty_config())
    first.state.phase = phase
    first.state.current_epoch = 7
    first.state.phase_step = 11
    second = GXGController(duty_config())

    second.load_state_dict(first.state_dict())

    assert second.state.phase == phase
    assert second.state.current_epoch == 7
    assert second.state.phase_step == 11


def test_nominal_budget_routes_to_quality_check_without_an_extra_update():
    config = duty_config(start_phase="adam")
    model = nn.Linear(1, 1, bias=False)
    optimizer = GXGOptimizer(model, config)
    before = model.weight.detach().clone()

    result = optimizer.step(StepContext(epoch=0, nominal_budget_exhausted=True))

    assert not result.update_accepted
    assert result.phase == GXGPhase.FINAL_QUALITY_CHECK
    assert torch.equal(model.weight, before)


def test_adam_only_public_optimizer_matches_adamw_and_round_trips_state():
    config = GXGConfig(
        adam=AdamConfig(lr=0.01, betas=(0.9, 0.99)),
        gn=replace(GNConfig(), enabled=False),
        duty_cycle=DutyCycleConfig(gn_epochs=1, adam_epochs=1, start_phase="adam"),
    )
    ours_model = nn.Linear(2, 1, bias=False)
    baseline_model = nn.Linear(2, 1, bias=False)
    baseline_model.load_state_dict(ours_model.state_dict())
    ours = gxg_optimizer(ours_model, config)
    baseline = torch.optim.AdamW(baseline_model.parameters(), lr=0.01, betas=(0.9, 0.99), weight_decay=0.0)
    gradient = torch.tensor([[0.2, -0.4]])
    ours_model.weight.grad = gradient.clone()
    baseline_model.weight.grad = gradient.clone()

    ours.step(StepContext(epoch=0))
    baseline.step()

    assert torch.allclose(ours_model.weight, baseline_model.weight, atol=1.0e-7, rtol=1.0e-6)
    resumed_model = nn.Linear(2, 1, bias=False)
    resumed_model.load_state_dict(ours_model.state_dict())
    resumed = GXGOptimizer(resumed_model, config)
    resumed.load_state_dict(ours.state_dict())
    assert resumed.get_phase() == ours.get_phase()
    assert resumed.adam.state["weight"]["step"] == ours.adam.state["weight"]["step"]


def test_one_cpu_gn_step_reduces_independent_reference_loss():
    config = duty_config()
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = GXGOptimizer(model, config)
    curvature_x = torch.tensor([[1.0], [2.0]])
    curvature_y = torch.tensor([[2.0], [4.0]])
    reference_x = torch.tensor([[3.0]])
    reference_y = torch.tensor([[6.0]])
    batch = FunctionalBatch(
        (curvature_x,),
        lambda output: 0.5 * (output - curvature_y).square().sum(),
        batch_id="curvature-0",
    )
    reference = lambda: 0.5 * (model(reference_x) - reference_y).square().sum().float()
    initial = reference().item()
    real_gradient = {"weight": torch.tensor([[-10.0]])}

    result = optimizer.step(
        StepContext(
            epoch=0,
            gn_batch=batch,
            reference_loss_closure=reference,
            reference_batch_id="reference-0",
            shadow_microbatch_gradients=(real_gradient,),
        )
    )

    assert result.update_accepted
    assert result.gn_update_norm > 0
    assert reference().item() < initial
    assert optimizer.get_phase() == GXGPhase.GN_BOOTSTRAP
    assert optimizer.adam.state["weight"]["m"].item() == pytest.approx(-1.0)
    assert optimizer._latest_gn_direction["weight"].item() > 0


def test_scheduled_gn_to_adam_boundary_runs_finite_bridge_then_adam():
    config = replace(
        duty_config(),
        bridge=BridgeConfig(length_adam_steps=2, rho_start=0.3, rho_end=0.0, schedule="linear"),
    )
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    optimizer = GXGOptimizer(model, config)
    inputs = torch.tensor([[1.0], [2.0]])
    targets = torch.tensor([[2.0], [4.0]])
    batch = FunctionalBatch((inputs,), lambda output: 0.5 * (output - targets).square().sum())
    reference = lambda: 0.5 * (model(torch.tensor([[3.0]])) - 6.0).square().sum().float()
    gradient = {"weight": torch.tensor([[-1.0]])}
    optimizer.step(
        StepContext(
            epoch=0,
            gn_batch=batch,
            reference_loss_closure=reference,
            shadow_microbatch_gradients=(gradient,),
        )
    )

    first = optimizer.step(
        StepContext(epoch=1, reference_gradient=gradient, shadow_microbatch_gradients=(gradient,))
    )
    second = optimizer.step(
        StepContext(epoch=1, reference_gradient=gradient, shadow_microbatch_gradients=(gradient,))
    )

    assert first.update_accepted
    assert first.phase == GXGPhase.BRIDGE_TO_ADAM
    assert first.switch_reason == "fixed_duty_cycle_enter_adam"
    assert second.update_accepted
    assert second.phase == GXGPhase.ADAM
    assert second.switch_reason == "bridge_complete"


def test_final_recovery_uses_best_validation_checkpoint_and_patience():
    config = GXGConfig(
        gn=replace(GNConfig(), inner_optimizer_matrix="cg", inner_optimizer_vector="cg"),
        duty_cycle=DutyCycleConfig(gn_epochs=1, adam_epochs=1, start_phase="adam"),
        final_mile=FinalMileConfig(
            enabled=True,
            metric_threshold=0.9,
            patience_evaluations=1,
            max_recovery_attempts=1,
        ),
    )
    model = nn.Linear(1, 1, bias=False)
    optimizer = GXGOptimizer(model, config)
    optimizer.step(StepContext(epoch=0, nominal_budget_exhausted=True))
    best_weight = model.weight.detach().clone()

    below = optimizer.evaluate_quality(EvalContext(metric=0.8, loss=0.5, checkpoint_id="best"))
    assert below.phase == GXGPhase.FINAL_GN_RECOVERY
    with torch.no_grad():
        model.weight.add_(10.0)
    failed = optimizer.evaluate_quality(EvalContext(metric=0.7, loss=0.6, checkpoint_id="worse"))

    assert failed.phase == GXGPhase.DONE
    assert not failed.target_met
    assert torch.equal(model.weight, best_weight)
    assert optimizer.should_stop()


def test_quality_modes_and_test_split_leakage_guard():
    config = GXGConfig(
        duty_cycle=DutyCycleConfig(start_phase="adam"),
        final_mile=FinalMileConfig(enabled=True, metric_mode="min", metric_threshold=0.2),
    )
    optimizer = GXGOptimizer(nn.Linear(1, 1), config)
    optimizer.controller.transition(GXGPhase.FINAL_QUALITY_CHECK, "test_setup")

    with pytest.raises(ValueError, match="final-reporting-only"):
        optimizer.evaluate_quality(EvalContext(split="test", metric=0.1))
    result = optimizer.evaluate_quality(EvalContext(metric=0.19))
    assert result.target_met
    assert result.phase == GXGPhase.DONE


def test_atomic_checkpoint_contains_model_optimizer_and_external_states(tmp_path):
    config = GXGConfig(gn=replace(GNConfig(), enabled=False), duty_cycle=DutyCycleConfig(start_phase="adam"))
    model = nn.Linear(1, 1)
    optimizer = GXGOptimizer(model, config)
    path = optimizer.save_checkpoint(tmp_path / "gxg.pt", extra={"epoch": 4})

    payload = torch.load(path)

    assert set(payload) == {"model", "gxg_optimizer", "scheduler", "scaler", "sampler", "rng", "extra"}
    assert payload["extra"] == {"epoch": 4}

    saved_weight = model.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(5.0)
    extra = optimizer.load_checkpoint(path)
    assert torch.equal(model.weight, saved_weight)
    assert extra == {"epoch": 4}
