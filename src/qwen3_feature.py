"""Training-only dense-factor probes for Qwen feature-drift remapping."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from optimizer_v2.temporal import predictive_maps, remap_gradient
from qwen3_proposals import QwenMuonProposalAdapter


def collect_qwen_dense_factors(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    module_names: list[str],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Capture ``(X, E)`` factors without writing into any parameter gradient.

    Qwen uses PyTorch linear weights in ``[out, in]`` order.  Consequently
    ``X.T @ E`` is the mathematical `[in, out]` gradient used by the
    feature-remapping specification and equals ``weight.grad.T``.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] < 2 or not module_names:
        raise ValueError("factor probes need nonempty modules and two-token input sequences")
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise ValueError(f"unknown Qwen probe modules: {missing}")
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    for name in module_names:
        def hook(module, inputs, output, *, name=name):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"{name} did not produce a tensor")
            captured[name] = (inputs[0], output)
        handles.append(modules[name].register_forward_hook(hook))
    try:
        output = model(input_ids=input_ids, use_cache=False)
        logits = getattr(output, "logits", output)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("Qwen factor probe did not return logits")
        loss = functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1)
        )
        errors = torch.autograd.grad(loss, [captured[name][1] for name in module_names])
        return {
            name: (
                captured[name][0].detach().reshape(-1, captured[name][0].shape[-1]).float(),
                error.detach().reshape(-1, error.shape[-1]).float(),
            )
            for name, error in zip(module_names, errors)
        }
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def qwen_cohort_momentum_step(
    historical: torch.Tensor,
    fresh: torch.Tensor,
    gradient: torch.Tensor,
    *,
    beta: float,
    maps: tuple[list[torch.Tensor], list[torch.Tensor]] | None,
    refresh: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance Qwen's age-separated momentum in the documented orientation.

    The feature-map equations use matrices `[in, out]`; PyTorch Qwen linear
    weights and their momentum buffers use `[out, in]`.  Transposition around
    ``cohort_step`` makes the map act on the correct historical cohort only.
    """
    if historical.shape != fresh.shape or historical.shape != gradient.shape or historical.ndim != 2:
        raise ValueError("cohort tensors must be equally shaped matrices")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    # Muon's buffer is unnormalised: ``m <- beta*m + g``.  Preserve that
    # coefficient exactly; the generic reference helper uses EMA convention.
    next_historical = beta * historical.T
    next_fresh = beta * fresh.T + gradient.T
    if maps is not None:
        next_historical = remap_gradient(next_historical.float(), *maps).to(next_historical.dtype)
    momentum = next_historical + next_fresh
    momentum, next_historical, next_fresh = momentum.T, next_historical.T, next_fresh.T
    if refresh:
        return momentum, momentum, torch.zeros_like(momentum)
    return momentum, next_historical, next_fresh


def qwen_feature_prediction_diagnostics(
    old_fit: dict[str, tuple[torch.Tensor, torch.Tensor]],
    new_fit: dict[str, tuple[torch.Tensor, torch.Tensor]],
    old_check: dict[str, tuple[torch.Tensor, torch.Tensor]],
    new_check: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    block_size: int = 32,
) -> dict[str, dict]:
    """Evaluate the held-out gradient-prediction gate before any remapping.

    This reports the combined feature/error map's held-out error for each
    target.  It intentionally does not mutate momentum or return an optimizer
    correction, keeping the feature-remap preflight separate from training.
    """
    names = set(old_fit)
    if not names or names != set(new_fit) or names != set(old_check) or names != set(new_check):
        raise ValueError("feature prediction requires matching nonempty target dictionaries")
    diagnostics: dict[str, dict] = {}
    for name in sorted(names):
        _, diagnostics[name] = predictive_maps(
            old_fit[name], new_fit[name], old_check[name], new_check[name], block_size=block_size
        )
    return diagnostics


def qwen_feature_drift_preflight(
    before: torch.nn.Module,
    after: torch.nn.Module,
    fit_input_ids: torch.Tensor,
    check_input_ids: torch.Tensor,
    module_names: list[str],
    *,
    block_size: int = 32,
) -> dict[str, dict]:
    """Measure held-out feature-map prediction between two Qwen snapshots.

    Both snapshots use deterministic evaluation mode for every factor pass.
    The caller supplies fixed training-only fit and check anchors; this helper
    neither samples data nor writes model parameter gradients.
    """
    if next(before.parameters()).device != next(after.parameters()).device:
        raise ValueError("feature-drift snapshots must be on the same device")
    device = next(before.parameters()).device
    fit_input_ids = fit_input_ids.to(device)
    check_input_ids = check_input_ids.to(device)
    modes = [(module, module.training) for model in (before, after) for module in model.modules()]
    try:
        before.eval()
        after.eval()
        return qwen_feature_prediction_diagnostics(
            collect_qwen_dense_factors(before, fit_input_ids, module_names),
            collect_qwen_dense_factors(after, fit_input_ids, module_names),
            collect_qwen_dense_factors(before, check_input_ids, module_names),
            collect_qwen_dense_factors(after, check_input_ids, module_names),
            block_size=block_size,
        )
    finally:
        for module, training in modes:
            module.training = training


class QwenFeatureCohortOptimizer:
    """Muon proposal route with age-separated, externally checked momentum.

    This small controller deliberately has no factor collection policy: the
    training driver supplies accepted maps only after its fixed-anchor check.
    That keeps a rejected or failed probe on the exact baseline route.
    """

    def __init__(
        self,
        parameters: dict[str, torch.nn.Parameter],
        matrix_names: set[str],
        *,
        learning_rate: float,
        weight_decay: float,
        momentum: float = 0.95,
    ) -> None:
        if learning_rate <= 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("invalid feature-cohort optimizer configuration")
        self.adapter = QwenMuonProposalAdapter(parameters, matrix_names, momentum=momentum)
        self.learning_rates = {name: learning_rate for name in matrix_names}
        self.weight_decays = {name: weight_decay for name in matrix_names}
        self.momentum = momentum
        self.cohorts: dict[str, dict[str, torch.Tensor]] = {}
        self.last_momentum: dict[str, torch.Tensor] = {}
        self.model: torch.nn.Module | None = None
        self.fit_ids: torch.Tensor | None = None
        self.check_ids: torch.Tensor | None = None
        self.module_names: list[str] = []
        self.interval = 8
        self.mode = "map"
        self.snapshots: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
        self.steps = 0
        self.last_diagnostics: dict[str, dict] = {}

    def configure_anchors(
        self,
        model: torch.nn.Module,
        fit_ids: torch.Tensor,
        check_ids: torch.Tensor,
        module_names: list[str],
        *,
        interval: int = 8,
        mode: str = "map",
    ) -> None:
        if interval <= 0 or mode not in {"map", "scalar"} or not module_names or fit_ids.ndim != 2 or check_ids.ndim != 2:
            raise ValueError("feature anchors require nonempty two-dimensional IDs and interval")
        self.model, self.fit_ids, self.check_ids = model, fit_ids.detach().cpu(), check_ids.detach().cpu()
        self.module_names, self.interval, self.mode = list(module_names), interval, mode

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.adapter.parameters.values():
            if parameter.grad is not None:
                parameter.grad = None if set_to_none else parameter.grad.zero_()

    @torch.no_grad()
    def step_with_maps(
        self, maps: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]] | None], *, refresh: bool
    ) -> None:
        if set(maps) != self.adapter.matrix_names:
            raise ValueError("maps must cover exactly the selected matrices")
        buffers: dict[str, torch.Tensor] = {}
        next_cohorts: dict[str, dict[str, torch.Tensor]] = {}
        for name in sorted(self.adapter.matrix_names):
            gradient = self.adapter.parameters[name].grad
            if gradient is None:
                raise RuntimeError(f"missing gradient for {name}")
            previous = self.cohorts.get(name, {})
            historical = previous.get("historical", torch.zeros_like(gradient))
            fresh = previous.get("fresh", torch.zeros_like(gradient))
            value, next_historical, next_fresh = qwen_cohort_momentum_step(
                historical, fresh, gradient, beta=self.momentum, maps=maps[name], refresh=refresh
            )
            buffers[name] = value
            next_cohorts[name] = {"historical": next_historical, "fresh": next_fresh}
        proposals = self.adapter.propose(self.learning_rates, self.weight_decays, buffers)
        self.adapter.commit(proposals)
        self.cohorts = next_cohorts
        self.last_momentum = {name: value.detach().clone() for name, value in buffers.items()}

    def step(self) -> None:
        due = self.steps % self.interval == 0
        maps = {name: None for name in self.adapter.matrix_names}
        diagnostics: dict[str, dict] = {}
        if due and self.model is not None and self.fit_ids is not None and self.check_ids is not None:
            device = next(self.model.parameters()).device
            modes = [(module, module.training) for module in self.model.modules()]
            try:
                self.model.eval()
                fit = collect_qwen_dense_factors(self.model, self.fit_ids.to(device), self.module_names)
                check = collect_qwen_dense_factors(self.model, self.check_ids.to(device), self.module_names)
            finally:
                for module, training in modes:
                    module.training = training
            old = self.snapshots
            if old:
                for module in self.module_names:
                    accepted_maps, diagnostic = predictive_maps(
                        old["fit"][module], fit[module], old["check"][module], check[module]
                    )
                    name = module + ".weight"
                    if self.mode == "map":
                        maps[name] = accepted_maps if diagnostic["accepted"] else None
                    else:
                        old_gradient = old["check"][module][0].T @ old["check"][module][1]
                        current_gradient = check[module][0].T @ check[module][1]
                        scalar = float((old_gradient * current_gradient).sum() / old_gradient.square().sum().clamp_min(1e-30))
                        if scalar > 0:
                            left, right = accepted_maps
                            maps[name] = ([torch.eye(len(value), device=value.device, dtype=value.dtype) * scalar**0.5 for value in left], right)
                        diagnostic["scalar"] = scalar
                    diagnostics[name] = diagnostic
            self.snapshots = {"fit": fit, "check": check}
        self.step_with_maps(maps, refresh=due)
        self.steps += 1
        self.last_diagnostics = diagnostics

    def state_dict(self) -> dict:
        return {
            "version": 1, "momentum": self.momentum, "learning_rates": self.learning_rates,
            "weight_decays": self.weight_decays, "cohorts": self.cohorts,
            "last_momentum": self.last_momentum, "snapshots": self.snapshots,
            "steps": self.steps, "adapter_state": self.adapter.state,
        }

    def load_state_dict(self, saved: dict) -> None:
        if (
            saved.get("version") != 1 or saved.get("momentum") != self.momentum
            or saved.get("learning_rates") != self.learning_rates or saved.get("weight_decays") != self.weight_decays
        ):
            raise ValueError("feature-cohort optimizer configuration changed")
        self.cohorts, self.last_momentum = saved["cohorts"], saved["last_momentum"]
        self.snapshots, self.steps, self.adapter.state = saved["snapshots"], int(saved["steps"]), saved["adapter_state"]
