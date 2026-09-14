"""Non-mutating Muon proposals for Qwen optimizer ideas.

The adapter mirrors the repository's custom :class:`optimizers.Muon` update
exactly, while making the learning and decay increments separately available
to a proposal filter.  A caller must commit each proposal at most once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from optimizers import Muon
from optimizer_v2.temporal import fixed_sketch, guarded_filter_step


@dataclass(frozen=True)
class QwenProposal:
    """One parameter's baseline update, before an optional correction."""

    learning: torch.Tensor
    decay: torch.Tensor
    direction: torch.Tensor
    value: torch.Tensor
    next_state: dict[str, torch.Tensor]


class QwenMuonProposalAdapter:
    """Compute and atomically commit custom-Muon proposals without mutation."""

    def __init__(
        self,
        parameters: dict[str, torch.nn.Parameter],
        matrix_names: set[str],
        *,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if not 0 <= momentum < 1:
            raise ValueError("momentum must lie in [0, 1)")
        if ns_steps <= 0:
            raise ValueError("ns_steps must be positive")
        if not matrix_names <= parameters.keys():
            raise ValueError("every matrix name must name a supplied parameter")
        if any(parameters[name].ndim != 2 for name in matrix_names):
            raise ValueError("Muon proposals require two-dimensional matrices")
        self.parameters = parameters
        self.matrix_names = set(matrix_names)
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.state: dict[str, dict[str, torch.Tensor]] = {}

    @torch.no_grad()
    def propose(
        self,
        learning_rates: dict[str, float],
        weight_decays: dict[str, float],
    ) -> dict[str, QwenProposal]:
        """Return exact custom-Muon updates without changing values or state."""
        if set(learning_rates) != self.matrix_names or set(weight_decays) != self.matrix_names:
            raise ValueError("rates and decays must cover exactly the selected matrices")
        proposals: dict[str, QwenProposal] = {}
        for name in sorted(self.matrix_names):
            parameter = self.parameters[name]
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError(f"missing gradient for selected matrix {name}")
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"nonfinite gradient for selected matrix {name}")
            learning_rate = float(learning_rates[name])
            weight_decay = float(weight_decays[name])
            if learning_rate <= 0 or weight_decay < 0:
                raise ValueError("learning rates must be positive and decays non-negative")
            flattened = gradient.reshape(gradient.shape[0], -1)
            old_buffer = self.state.get(name, {}).get("momentum_buffer")
            buffer = torch.zeros_like(flattened) if old_buffer is None else old_buffer.clone()
            buffer.mul_(self.momentum).add_(flattened)
            update = flattened.add(buffer, alpha=self.momentum) if self.nesterov else buffer
            direction = Muon.orthogonalize(update, self.ns_steps).reshape_as(parameter)
            scale = Muon.scaled_lr(learning_rate, flattened.shape[0], flattened.shape[1])
            learning = direction.mul(-scale)
            decay = parameter.detach().mul(-learning_rate * weight_decay)
            proposals[name] = QwenProposal(
                learning=learning,
                decay=decay,
                direction=direction.mul(scale / learning_rate),
                value=parameter.detach() + learning + decay,
                next_state={"momentum_buffer": buffer},
            )
        return proposals

    @torch.no_grad()
    def commit(
        self,
        proposals: dict[str, QwenProposal],
        corrections: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Commit each proposal and optional learning-increment correction once."""
        if set(proposals) != self.matrix_names:
            raise ValueError("proposals must cover exactly the selected matrices")
        corrections = corrections or {}
        if not set(corrections) <= self.matrix_names:
            raise ValueError("corrections may target only selected matrices")
        for name in sorted(self.matrix_names):
            proposal = proposals[name]
            correction = corrections.get(name)
            if correction is not None and (
                correction.shape != self.parameters[name].shape or not torch.isfinite(correction).all()
            ):
                raise ValueError(f"invalid correction for {name}")
        for name in sorted(self.matrix_names):
            proposal = proposals[name]
            self.parameters[name].copy_(proposal.value)
            if name in corrections:
                self.parameters[name].add_(corrections[name])
            self.state[name] = {key: value.detach().clone() for key, value in proposal.next_state.items()}


@torch.no_grad()
def qwen_notch_corrections(
    proposals: dict[str, QwenProposal],
    parameters: dict[str, torch.nn.Parameter],
    state: dict,
    *,
    step: int,
    seed: int,
    learning_rates: dict[str, float],
    radius: float = 0.8,
) -> tuple[dict[str, torch.Tensor], dict, dict[str, dict]]:
    """Return post-polar notch corrections without committing a proposal.

    ``QwenProposal.direction`` is the Muon direction after its shape scaling
    and before the scalar learning rate.  This is precisely the signal the
    resonance specification permits the notch to observe and filter.
    """
    if set(proposals) != set(learning_rates) or not set(proposals) <= parameters.keys():
        raise ValueError("proposal parameters and learning rates must agree")
    filters = dict(state.get("filters", {}))
    signs = dict(state.get("signs", {}))
    corrections: dict[str, torch.Tensor] = {}
    diagnostics: dict[str, dict] = {}
    for index, name in enumerate(sorted(proposals)):
        direction = proposals[name].direction
        sketch, signs[name] = fixed_sketch(direction, seed + 10_007 * (index + 1), signs.get(name))
        output, filters[name], diagnostics[name] = guarded_filter_step(
            direction,
            parameters[name],
            filters.get(name, {}),
            step,
            sketch,
            radius=radius,
        )
        filters[name].setdefault("active", False)
        corrections[name] = -float(learning_rates[name]) * (output - direction)
    return corrections, {"filters": filters, "signs": signs}, diagnostics
