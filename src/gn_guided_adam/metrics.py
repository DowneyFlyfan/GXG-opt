from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OptimizerEvent:
    step: int
    refreshed: bool
    accepted: bool
    reason: str
    wall_time_seconds: float
    curvature_time_seconds: float
    acceptance_time_seconds: float
    adam_step_ema_seconds: float
    gn_cost_adam_steps: float
    metrics: dict[str, float]


class OptimizerMetrics:
    def __init__(self) -> None:
        self.events: list[OptimizerEvent] = []
        self.adam_step_ema_seconds = 0.0
        self.cumulative_optimizer_seconds = 0.0
        self.cumulative_curvature_seconds = 0.0
        self.cumulative_acceptance_seconds = 0.0

    def observe_adam_time(self, seconds: float) -> None:
        self.adam_step_ema_seconds = seconds if self.adam_step_ema_seconds == 0 else 0.9 * self.adam_step_ema_seconds + 0.1 * seconds

    def add(self, event: OptimizerEvent) -> None:
        self.events.append(event)
        self.cumulative_optimizer_seconds += event.wall_time_seconds
        self.cumulative_curvature_seconds += event.curvature_time_seconds
        self.cumulative_acceptance_seconds += event.acceptance_time_seconds

    def summary(self) -> dict[str, float]:
        accepted = sum(event.accepted for event in self.events if event.refreshed)
        refreshed = sum(event.refreshed for event in self.events)
        return {
            "optimizer_seconds": self.cumulative_optimizer_seconds,
            "curvature_seconds": self.cumulative_curvature_seconds,
            "acceptance_seconds": self.cumulative_acceptance_seconds,
            "adam_step_ema_seconds": self.adam_step_ema_seconds,
            "refresh_events": float(refreshed),
            "acceptance_rate": accepted / max(refreshed, 1),
        }

    def write_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
        return destination

    def state_dict(self) -> dict[str, Any]:
        return {
            "events": [asdict(event) for event in self.events],
            "adam_step_ema_seconds": self.adam_step_ema_seconds,
            "cumulative_optimizer_seconds": self.cumulative_optimizer_seconds,
            "cumulative_curvature_seconds": self.cumulative_curvature_seconds,
            "cumulative_acceptance_seconds": self.cumulative_acceptance_seconds,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.events = [OptimizerEvent(**event) for event in state["events"]]
        self.adam_step_ema_seconds = float(state["adam_step_ema_seconds"])
        self.cumulative_optimizer_seconds = float(state["cumulative_optimizer_seconds"])
        self.cumulative_curvature_seconds = float(state["cumulative_curvature_seconds"])
        self.cumulative_acceptance_seconds = float(state["cumulative_acceptance_seconds"])
