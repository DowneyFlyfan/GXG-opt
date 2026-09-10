from __future__ import annotations

import time

from .config import GXGConfig
from .state import ControllerState, GXGPhase


class GXGController:
    """Route phases by a fixed, repeatable epoch duty cycle.

    Adaptive probes and wall-clock competitiveness do not affect routing in this
    first experiment. Numerical hard failures may suppress GN for the remainder of
    the current epoch, but the next configured GN epoch is still attempted.
    """

    _NORMAL_PHASES = {
        GXGPhase.GN_BOOTSTRAP,
        GXGPhase.BRIDGE_TO_ADAM,
        GXGPhase.ADAM,
        GXGPhase.GN_CORRECTION,
    }

    def __init__(self, config: GXGConfig) -> None:
        self.config = config
        starts_with_gn = config.gn.enabled and config.duty_cycle.start_phase == "gn"
        initial = GXGPhase.GN_BOOTSTRAP if starts_with_gn else GXGPhase.ADAM
        self.state = ControllerState(phase=initial, phase_started_at=time.perf_counter())

    def transition(self, phase: GXGPhase, reason: str) -> str:
        previous = self.state.phase
        if previous == phase:
            return reason
        now = time.perf_counter()
        self.state.transition_events.append(
            {
                "from": previous.value,
                "to": phase.value,
                "reason": reason,
                "epoch": self.state.current_epoch,
                "wall_time": now,
            }
        )
        self.state.reason_history.append(reason)
        self.state.phase = phase
        self.state.phase_step = 0
        self.state.accepted_gn_in_phase = 0
        self.state.consecutive_gn_rejections = 0
        self.state.phase_started_at = now
        self.state.phase_started_epoch = max(self.state.current_epoch, 0)
        return reason

    def desired_optimizer(self, epoch: int) -> str:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("StepContext.epoch must be a non-negative integer")
        if not self.config.gn.enabled or self.state.gn_suppressed_epoch == epoch:
            return "adam"
        duty = self.config.duty_cycle
        cycle = duty.gn_epochs + duty.adam_epochs
        offset = epoch % cycle
        if duty.start_phase == "gn":
            return "gn" if offset < duty.gn_epochs else "adam"
        return "adam" if offset < duty.adam_epochs else "gn"

    def route_epoch(self, epoch: int, nominal_budget_exhausted: bool = False) -> str | None:
        state = self.state
        if state.phase not in self._NORMAL_PHASES:
            return None
        if epoch < state.current_epoch:
            raise ValueError("StepContext.epoch must be monotonic")
        state.current_epoch = epoch
        if nominal_budget_exhausted:
            return self.transition(GXGPhase.FINAL_QUALITY_CHECK, "nominal_budget_exhausted")
        desired = self.desired_optimizer(epoch)
        if desired == "gn" and state.phase in {GXGPhase.ADAM, GXGPhase.BRIDGE_TO_ADAM}:
            state.switch_count += 1
            return self.transition(GXGPhase.GN_CORRECTION, "fixed_duty_cycle_enter_gn")
        if desired == "adam" and state.phase in {GXGPhase.GN_BOOTSTRAP, GXGPhase.GN_CORRECTION}:
            state.switch_count += 1
            next_phase = GXGPhase.BRIDGE_TO_ADAM if self.config.duty_cycle.bridge_on_gn_to_adam else GXGPhase.ADAM
            return self.transition(next_phase, "fixed_duty_cycle_enter_adam")
        return None

    def begin_step(self) -> None:
        self.state.phase_step += 1

    def after_gn(
        self,
        *,
        accepted: bool,
        reduction_ratio: float,
        transfer_ratio: float,
        finite: bool,
        update_norm_safe: bool,
        data_exhausted: bool = False,
    ) -> str | None:
        state, config = self.state, self.config
        state.accepted_gn_in_phase += int(accepted)
        if not accepted or reduction_ratio < config.gn.min_reduction_ratio:
            state.consecutive_gn_rejections += 1
        else:
            state.consecutive_gn_rejections = 0
        hard_failure = (
            not finite
            or not update_norm_safe
            or transfer_ratio < 0
            or state.consecutive_gn_rejections >= config.gn.rejection_persistence
        )
        if state.phase == GXGPhase.FINAL_GN_RECOVERY:
            state.final_recovery_accepted_steps += int(accepted)
            wall_limit = config.final_mile.max_wall_time_seconds
            exhausted = (
                data_exhausted
                or hard_failure
                or (wall_limit is not None and time.perf_counter() - state.phase_started_at >= wall_limit)
            )
            if exhausted:
                return self.transition(GXGPhase.DONE, "final_gn_recovery_exhausted")
            return None
        if hard_failure:
            state.gn_suppressed_epoch = state.current_epoch
            next_phase = GXGPhase.BRIDGE_TO_ADAM if config.duty_cycle.bridge_on_gn_to_adam else GXGPhase.ADAM
            return self.transition(next_phase, "gn_safety_exit_for_epoch")
        return None

    def bridge_complete(self, fallback: bool = False) -> str:
        reason = "bridge_guard_fallback" if fallback else "bridge_complete"
        return self.transition(GXGPhase.ADAM, reason)

    def state_dict(self):
        return self.state.state_dict()

    def load_state_dict(self, state) -> None:
        self.state = ControllerState.from_state_dict(dict(state))
