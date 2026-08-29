from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from kronecker_ggn_common.curvature_operator import (
    FunctionalCurvatureBatch,
    GGNLinearOperator,
)
from kronecker_ggn_common.kronecker_factors import KroneckerFactorEstimator
from kronecker_ggn_common.layer_registry import LayerRegistry
from kronecker_ggn_common.types import CurvatureUpdate


def _logits(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    output = model(token_ids)
    return output.logits if hasattr(output, "logits") else output


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


def nlp_mc_ggn_update(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    registry: LayerRegistry,
    *,
    include_exact_residual_operator: bool,
    seed: int,
    residual_batch_size: int = 1,
    residual_sequence_length: int = 128,
) -> CurvatureUpdate:
    """Estimate language-model MC-GGN factors and optional exact GGN products.

    The factor estimate uses labels sampled from the current categorical model,
    so the captured preactivation gradients are Monte-Carlo generalized
    Gauss–Newton tangents.  The optional operator retains the exact
    cross-entropy output Hessian for the low-rank residual correction.
    """
    token_ids, targets = batch
    device = next(model.parameters()).device
    token_ids = token_ids.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    generator = torch.Generator(device=device).manual_seed(seed)

    def sampled_curvature_loss() -> torch.Tensor:
        logits = _logits(model, token_ids)
        probabilities = logits.detach().float().softmax(dim=-1).reshape(
            -1, logits.size(-1)
        )
        sampled_targets = torch.multinomial(
            probabilities, 1, generator=generator
        ).reshape_as(targets)
        # Cross entropy is a mean over N tokens.  Its output gradient carries
        # 1/N, while a covariance of sampled gradients would then carry 1/N².
        # Scaling the pseudo-loss by sqrt(N) gives the required 1/N curvature.
        return _cross_entropy(logits.float(), sampled_targets) * token_ids.numel() ** 0.5

    estimator = KroneckerFactorEstimator(dtype=torch.float32)
    factors = estimator.capture_from_loss(
        registry,
        sampled_curvature_loss,
        curvature_mode="mc_ggn",
    )
    operators = {}
    if include_exact_residual_operator:
        residual_tokens = token_ids[:residual_batch_size, :residual_sequence_length]
        residual_targets = targets[:residual_batch_size, :residual_sequence_length]
        if residual_tokens.numel() == 0:
            raise ValueError("The exact residual curvature subbatch is empty")
        curvature_batch = FunctionalCurvatureBatch(
            args=(residual_tokens,),
            loss_fn=lambda output: _cross_entropy(
                (output.logits if hasattr(output, "logits") else output).float(),
                residual_targets,
            ),
            batch_id=f"nlp-{seed}",
        )
        operator = GGNLinearOperator(model, registry, curvature_batch)
        operators = {layer.layer_id: operator for layer in registry.supported}
    return CurvatureUpdate(
        curvature_mode="mc_ggn",
        factors=factors,
        ggn_operators=operators,
        measurements={"curvature/sample_count": float(token_ids.numel())},
    )
