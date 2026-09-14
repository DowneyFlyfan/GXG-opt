"""Non-mutating Muon proposals for Qwen optimizer ideas.

The adapter mirrors the repository's custom :class:`optimizers.Muon` update
exactly, while making the learning and decay increments separately available
to a proposal filter.  A caller must commit each proposal at most once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from optimizers import Muon
from qwen3_attention import QwenAttentionReplayCapture, qwen_route_head_corrections
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


class QwenProposalNotchOptimizer:
    """Muon with the active proposal-notch mechanism for selected matrices.

    This class owns only the matrix route.  A full Qwen candidate combines it
    with the unchanged AdamW auxiliary route, preserving the baseline split.
    """

    def __init__(
        self,
        parameters: dict[str, torch.nn.Parameter],
        matrix_names: set[str],
        *,
        learning_rate: float,
        weight_decay: float,
        radius: float = 0.8,
        seed: int = 0,
    ) -> None:
        if learning_rate <= 0 or weight_decay < 0 or not 0 < radius < 1:
            raise ValueError("invalid proposal-notch hyperparameters")
        self.adapter = QwenMuonProposalAdapter(parameters, matrix_names)
        self.learning_rates = {name: learning_rate for name in matrix_names}
        self.weight_decays = {name: weight_decay for name in matrix_names}
        self.radius = radius
        self.seed = seed
        self.steps = 0
        self.state: dict = {}
        self.last_diagnostics: dict[str, dict] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for name in self.adapter.matrix_names:
            parameter = self.adapter.parameters[name]
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        proposals = self.adapter.propose(self.learning_rates, self.weight_decays)
        corrections, next_state, diagnostics = qwen_notch_corrections(
            proposals,
            self.adapter.parameters,
            self.state,
            step=self.steps + 1,
            seed=self.seed,
            learning_rates=self.learning_rates,
            radius=self.radius,
        )
        self.adapter.commit(proposals, corrections)
        self.steps += 1
        self.state = next_state
        self.last_diagnostics = diagnostics

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "steps": self.steps,
            "learning_rates": self.learning_rates,
            "weight_decays": self.weight_decays,
            "radius": self.radius,
            "seed": self.seed,
            "state": self.state,
            "adapter_state": self.adapter.state,
        }

    def load_state_dict(self, saved: dict) -> None:
        if (
            saved.get("version") != 1
            or saved.get("learning_rates") != self.learning_rates
            or saved.get("weight_decays") != self.weight_decays
            or saved.get("radius") != self.radius
            or saved.get("seed") != self.seed
        ):
            raise ValueError("proposal-notch optimizer configuration changed")
        self.steps = int(saved["steps"])
        self.state = saved["state"]
        self.adapter.state = saved["adapter_state"]


class QwenRoutingResistanceOptimizer:
    """Muon proposal filter for one rotating Qwen query/key attention head.

    ``prepare_forward`` installs a short-lived hook only on scheduled active
    steps.  It captures one sequence from the normal training forward; ``step``
    then commits exactly one filtered or baseline proposal and removes the
    hook.  A zero strength is a direct Muon bypass with no hook allocation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        parameters: dict[str, torch.nn.Parameter],
        matrix_names: set[str],
        *,
        learning_rate: float,
        weight_decay: float,
        rho: float = 1.0,
        interval: int = 8,
        query_rows: int = 4,
        edges_per_row: int = 4,
        mixture: float = 0.05,
        seed: int = 0,
    ) -> None:
        if (
            learning_rate <= 0
            or weight_decay < 0
            or rho < 0
            or interval <= 0
            or query_rows <= 0
            or edges_per_row <= 0
            or not 0 <= mixture <= 1
        ):
            raise ValueError("invalid routing-resistance hyperparameters")
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) < 3:
            raise ValueError("routing resistance requires interior Qwen layers")
        eligible = []
        for layer_index in range(1, len(layers) - 1):
            prefix = f"model.layers.{layer_index}.self_attn."
            if prefix + "q_proj.weight" in matrix_names and prefix + "k_proj.weight" in matrix_names:
                eligible.append(layer_index)
        if not eligible:
            raise ValueError("routing resistance requires a selected Q/K projection pair")
        self.model = model
        self.adapter = QwenMuonProposalAdapter(parameters, matrix_names)
        self.learning_rates = {name: learning_rate for name in matrix_names}
        self.weight_decays = {name: weight_decay for name in matrix_names}
        self.rho = rho
        self.interval = interval
        self.query_rows = query_rows
        self.edges_per_row = edges_per_row
        self.mixture = mixture
        self.seed = seed
        self.layer_indices = tuple(eligible)
        self.steps = 0
        self.last_diagnostics: dict = {}
        self._capture: QwenAttentionReplayCapture | None = None
        self._target: tuple[int, int] | None = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        for name in self.adapter.matrix_names:
            parameter = self.adapter.parameters[name]
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.zero_()

    def _clear_capture(self) -> None:
        if self._capture is not None:
            self._capture.__exit__(None, None, None)
            self._capture = None
        self._target = None

    def prepare_forward(self) -> None:
        """Capture the next scheduled active step from its normal forward."""
        self._clear_capture()
        if self.rho == 0 or self.steps % self.interval:
            return
        event = self.steps // self.interval
        layer_index = self.layer_indices[event % len(self.layer_indices)]
        attention = self.model.model.layers[layer_index].self_attn
        query_head = (event // len(self.layer_indices)) % int(attention.config.num_attention_heads)
        self._capture = QwenAttentionReplayCapture(attention)
        self._capture.__enter__()
        self._target = (layer_index, query_head)

    def _generator(self, device: torch.device) -> torch.Generator:
        return torch.Generator(device=device).manual_seed(self.seed + 104_729 * (self.steps + 1))

    @torch.no_grad()
    def step(self) -> None:
        proposals = self.adapter.propose(self.learning_rates, self.weight_decays)
        diagnostics: dict = {"step": self.steps + 1, "disabled": self.rho == 0}
        corrections: dict[str, torch.Tensor] = {}
        try:
            if self.rho != 0 and self._capture is not None and self._target is not None:
                layer_index, query_head = self._target
                x, query, key = self._capture.replay(head=query_head)
                if x.shape[0] < 2:
                    diagnostics["capture_bypass"] = "sequence_too_short"
                else:
                    generator = self._generator(x.device)
                    valid_rows = torch.arange(1, x.shape[0], device=x.device)
                    row_count = min(self.query_rows, valid_rows.numel())
                    rows = valid_rows[
                        torch.randperm(valid_rows.numel(), device=x.device, generator=generator)[:row_count]
                    ].sort().values
                    attention = self.model.model.layers[layer_index].self_attn
                    ratio = int(attention.config.num_attention_heads) // int(attention.config.num_key_value_heads)
                    key_head = query_head // ratio
                    prefix = f"model.layers.{layer_index}.self_attn."
                    query_name, key_name = prefix + "q_proj.weight", prefix + "k_proj.weight"
                    query_correction, key_correction, route_diagnostic = qwen_route_head_corrections(
                        x,
                        query,
                        key,
                        rows,
                        proposals[query_name].learning,
                        proposals[key_name].learning,
                        query_head=query_head,
                        key_head=key_head,
                        head_dim=int(attention.head_dim),
                        edges_per_row=self.edges_per_row,
                        mixture=self.mixture,
                        rho=self.rho,
                        generator=generator,
                    )
                    corrections = {query_name: query_correction, key_name: key_correction}
                    diagnostics.update(layer=layer_index)
                    diagnostics.update(route_diagnostic)
            elif self.rho != 0:
                diagnostics["capture_bypass"] = "no_scheduled_forward_capture"
            self.adapter.commit(proposals, corrections)
            self.steps += 1
            self.last_diagnostics = diagnostics
        finally:
            self._clear_capture()

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "steps": self.steps,
            "learning_rates": self.learning_rates,
            "weight_decays": self.weight_decays,
            "rho": self.rho,
            "interval": self.interval,
            "query_rows": self.query_rows,
            "edges_per_row": self.edges_per_row,
            "mixture": self.mixture,
            "seed": self.seed,
            "layer_indices": self.layer_indices,
            "adapter_state": self.adapter.state,
        }

    def load_state_dict(self, saved: dict) -> None:
        expected = self.state_dict()
        for key in (
            "version",
            "learning_rates",
            "weight_decays",
            "rho",
            "interval",
            "query_rows",
            "edges_per_row",
            "mixture",
            "seed",
            "layer_indices",
        ):
            if saved.get(key) != expected[key]:
                raise ValueError("routing-resistance optimizer configuration changed")
        self.steps = int(saved["steps"])
        self.adapter.state = saved["adapter_state"]
