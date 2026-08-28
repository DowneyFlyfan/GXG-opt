from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .hooks import LinearCapture
from .layer_registry import LayerRegistry
from .types import CurvatureFactors


def update_factor_ema(
    previous: Tensor | None, estimate: Tensor, decay: float
) -> Tensor:
    if not 0 <= decay < 1:
        raise ValueError("factor decay must be in [0, 1)")
    estimate = 0.5 * (estimate + estimate.T)
    if previous is None:
        return estimate.detach().clone()
    if previous.shape != estimate.shape:
        raise ValueError("factor shape changed")
    return previous.mul(decay).add(estimate, alpha=1 - decay)


@dataclass
class FactorAccumulator:
    activation: Tensor | None = None
    output: Tensor | None = None
    updates: int = 0

    def update(
        self, factors: CurvatureFactors, decay: float, dtype: torch.dtype
    ) -> CurvatureFactors:
        activation = factors.activation.detach().to(dtype=dtype)
        output = factors.output.detach().to(dtype=dtype)
        if activation.ndim != 2 or activation.shape[0] != activation.shape[1]:
            raise ValueError("activation factor must be square")
        if output.ndim != 2 or output.shape[0] != output.shape[1]:
            raise ValueError("output factor must be square")
        if not torch.isfinite(activation).all() or not torch.isfinite(output).all():
            raise ValueError("factor estimate contains non-finite values")
        self.activation = update_factor_ema(self.activation, activation, decay)
        self.output = update_factor_ema(self.output, output, decay)
        self.updates += 1
        return CurvatureFactors(self.activation, self.output, factors.sample_count)


class KroneckerFactorEstimator:
    """Reference factor estimators with deliberately explicit curvature semantics."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        if dtype not in {torch.float32, torch.float64}:
            raise ValueError("factor dtype must be float32 or float64")
        self.dtype = dtype

    def from_output_curvatures(
        self, activations: Tensor, output_curvatures: Tensor
    ) -> CurvatureFactors:
        """Exact/reference factors from C_i matrices supplied by a curvature backend."""
        activations = activations.reshape(-1, activations.shape[-1]).to(
            dtype=self.dtype
        )
        if (
            output_curvatures.ndim != 3
            or output_curvatures.shape[0] != activations.shape[0]
        ):
            raise ValueError(
                "output_curvatures must have shape [samples, output_dim, output_dim]"
            )
        output_curvatures = output_curvatures.to(
            device=activations.device, dtype=self.dtype
        )
        count = activations.shape[0]
        activation = activations.T @ activations / max(count, 1)
        output = output_curvatures.mean(dim=0)
        return CurvatureFactors(activation, output, count)

    def from_mc_tangents(
        self, activations: Tensor, curvature_tangents: Tensor
    ) -> CurvatureFactors:
        """MC-GGN factors from likelihood-compatible sampled preactivation tangents."""
        return self._outer_product_factors(activations, curvature_tangents)

    def from_empirical_gradients(
        self, activations: Tensor, preactivation_gradients: Tensor
    ) -> CurvatureFactors:
        """Empirical-Fisher factors; callers must record that name, never GGN."""
        return self._outer_product_factors(activations, preactivation_gradients)

    def capture_from_loss(
        self,
        registry: LayerRegistry,
        loss_closure,
        *,
        curvature_mode: str,
    ) -> dict[str, CurvatureFactors]:
        """Capture MC tangents or empirical gradients without changing parameter grads.

        ``loss_closure`` takes no arguments, performs the model forward on the
        caller-selected curvature batch, and returns a scalar. For ``mc_ggn`` it
        must be a likelihood-compatible pseudo-loss whose preactivation gradient
        is the sampled curvature tangent. Exact GGN requires explicit ``C_i`` or
        a :class:`GGNLinearOperator` and is intentionally rejected here.
        """
        if curvature_mode not in {"mc_ggn", "empirical_fisher"}:
            raise ValueError(
                "Hook capture supports only mc_ggn or empirical_fisher; "
                "exact_ggn requires output-curvature matrices"
            )
        with LinearCapture(registry) as capture:
            loss = loss_closure()
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise ValueError("loss_closure must return a scalar tensor")
        layer_outputs = [
            output
            for layer in registry.supported
            for output in capture.outputs.get(layer.layer_id, [])
        ]
        if not layer_outputs:
            raise RuntimeError("No registered linear outputs were captured")
        gradients = torch.autograd.grad(loss, layer_outputs, allow_unused=True)
        gradient_index = 0
        factors = {}
        for layer in registry.supported:
            outputs = capture.outputs.get(layer.layer_id, [])
            layer_gradients = gradients[gradient_index : gradient_index + len(outputs)]
            gradient_index += len(outputs)
            if not outputs or any(value is None for value in layer_gradients):
                continue
            activation = capture.flattened_activations(layer.layer_id)
            tangent = torch.cat(
                [
                    value.detach().reshape(-1, value.shape[-1])
                    for value in layer_gradients
                ],
                dim=0,
            )
            if curvature_mode == "mc_ggn":
                factors[layer.layer_id] = self.from_mc_tangents(activation, tangent)
            else:
                factors[layer.layer_id] = self.from_empirical_gradients(
                    activation, tangent
                )
        return factors

    def _outer_product_factors(
        self, activations: Tensor, tangents: Tensor
    ) -> CurvatureFactors:
        activations = activations.reshape(-1, activations.shape[-1]).to(
            dtype=self.dtype
        )
        tangents = tangents.reshape(-1, tangents.shape[-1]).to(
            device=activations.device, dtype=self.dtype
        )
        if activations.shape[0] != tangents.shape[0]:
            raise ValueError("activation and tangent sample counts differ")
        count = activations.shape[0]
        return CurvatureFactors(
            activation=activations.T @ activations / max(count, 1),
            output=tangents.T @ tangents / max(count, 1),
            sample_count=count,
        )
