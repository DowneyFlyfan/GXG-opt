"""Training-only dense-factor probes for Qwen feature-drift remapping."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from optimizer_v2.temporal import cohort_step, predictive_maps


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
    momentum, next_historical, next_fresh = cohort_step(
        historical.T,
        fresh.T,
        gradient.T,
        beta,
        maps,
    )
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
