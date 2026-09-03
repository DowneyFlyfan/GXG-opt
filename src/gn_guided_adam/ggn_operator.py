from __future__ import annotations

import math
import time
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, grad, jvp, vjp

from .blocks import BlockSpec
from .execution import functional_model_state, preserve_execution_state
from .types import FunctionalBatch


class GGNOperatorError(RuntimeError):
    pass


def _tree_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (tuple, list)):
        return all(_tree_is_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_tree_is_finite(item) for item in value.values())
    return True


def softmax_cross_entropy_hvp(
    logits: torch.Tensor,
    vector: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.shape != vector.shape:
        raise ValueError("Logits and HVP vector shapes must match")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be mean, sum, or none")
    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probabilities = logits.to(compute_dtype).softmax(dim=-1)
    tangent = vector.to(compute_dtype)
    product = probabilities * tangent - probabilities * (probabilities * tangent).sum(dim=-1, keepdim=True)
    if mask is not None:
        if mask.shape != logits.shape[:-1]:
            raise ValueError("Mask must match logits without the class dimension")
        product = product * mask.to(product.dtype).unsqueeze(-1)
    if reduction == "mean":
        denominator = mask.sum() if mask is not None else logits.numel() // logits.shape[-1]
        product = product / denominator.clamp_min(1) if isinstance(denominator, torch.Tensor) else product / max(denominator, 1)
    return product.to(vector.dtype)


class GGNBlockOperator:
    """Matrix-free J^T H_loss J product for one tensor block."""

    def __init__(self, model: nn.Module, block: BlockSpec, batch: FunctionalBatch) -> None:
        if len(block.parameters) != 1:
            raise ValueError("The fixed prototype supports one parameter tensor per block")
        if not block.enabled:
            raise ValueError(f"Cannot build GGN operator for disabled block: {block.disabled_reason}")
        self.model = model
        self.block = block
        self.batch = batch
        self.parameter_name = block.parameter_names[0]
        self.parameter = block.parameters[0]
        self.matvec_count = 0
        self.matvec_time_seconds = 0.0

    @property
    def numel(self) -> int:
        return self.parameter.numel()

    def _functions(self):
        state = functional_model_state(self.model)
        base = state[self.parameter_name]

        def block_function(value: torch.Tensor) -> Any:
            candidate = dict(state)
            candidate[self.parameter_name] = value
            return functional_call(
                self.model,
                candidate,
                self.batch.args,
                dict(self.batch.kwargs),
                strict=False,
            )

        return base, block_function

    def gradient(self) -> torch.Tensor:
        with preserve_execution_state(self.model):
            base, block_function = self._functions()
            output, pullback = vjp(block_function, base)
            loss = self.batch.loss_fn(output)
            output_gradient = grad(self.batch.loss_fn)(output)
            if not torch.isfinite(loss).all() or not _tree_is_finite(output_gradient):
                raise GGNOperatorError("Curvature loss or output gradient is nonfinite")
            block_gradient = pullback(output_gradient)[0]
        dtype = torch.float64 if block_gradient.dtype == torch.float64 else torch.float32
        return block_gradient.detach().to(dtype=dtype).reshape(-1)

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.numel() != self.numel:
            raise ValueError("Block vector has the wrong size")
        started = time.perf_counter()
        with preserve_execution_state(self.model):
            base, block_function = self._functions()
            tangent = vector.reshape_as(base).to(device=base.device, dtype=base.dtype)
            output, pullback = vjp(block_function, base)
            loss = self.batch.loss_fn(output)
            output_gradient_fn = grad(self.batch.loss_fn)
            output_gradient = output_gradient_fn(output)
            if not torch.isfinite(loss).all() or not _tree_is_finite(output_gradient):
                raise GGNOperatorError("Curvature loss or output gradient is nonfinite")
            _, output_tangent = jvp(block_function, (base,), (tangent,))
            _, output_hvp = jvp(output_gradient_fn, (output,), (output_tangent,))
            product = pullback(output_hvp)[0]
        dtype = torch.float64 if product.dtype == torch.float64 else torch.float32
        result = product.detach().to(dtype=dtype).reshape(-1)
        self.matvec_count += 1
        self.matvec_time_seconds += time.perf_counter() - started
        if not torch.isfinite(result).all():
            raise GGNOperatorError("GGN product is nonfinite")
        return result

    def explicit_matrix_for_testing(self, maximum_elements: int = 256) -> torch.Tensor:
        if self.numel > maximum_elements:
            raise ValueError("Explicit GGN construction is test-only and limited to tiny blocks")
        dtype = torch.float64 if self.parameter.dtype == torch.float64 else torch.float32
        identity = torch.eye(self.numel, device=self.parameter.device, dtype=dtype)
        return torch.stack([self.matvec(identity[:, index]) for index in range(self.numel)], dim=1)

    def check_quadratic_form(self, vector: torch.Tensor, tolerance: float = 1.0e-6) -> float:
        value = float(torch.dot(vector.reshape(-1).float(), self.matvec(vector).float()).item())
        if not math.isfinite(value) or value < -tolerance:
            raise GGNOperatorError(f"GGN PSD check failed: {value}")
        return value


class AveragedGGNBlockOperator:
    """Average a fixed set of block-GGN microbatches for every Krylov product."""

    def __init__(self, operators: tuple[GGNBlockOperator, ...]) -> None:
        if not operators:
            raise ValueError("Averaged block GGN requires at least one operator")
        first = operators[0]
        if any(
            operator.model is not first.model
            or operator.parameter is not first.parameter
            for operator in operators[1:]
        ):
            raise ValueError("Averaged block GGN operators must share one parameter")
        self.operators = operators
        self.model = first.model
        self.block = first.block
        self.parameter = first.parameter
        self.matvec_count = 0
        self.matvec_time_seconds = 0.0
        self.batch_ids = tuple(operator.batch.batch_id for operator in operators)

    @property
    def numel(self) -> int:
        return self.operators[0].numel

    def gradient(self) -> torch.Tensor:
        total = None
        for operator in self.operators:
            gradient = operator.gradient()
            total = gradient if total is None else total.add_(gradient.to(total))
        assert total is not None
        return total / len(self.operators)

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        total = torch.zeros_like(vector.reshape(-1))
        for operator in self.operators:
            total.add_(operator.matvec(vector).to(total))
        self.matvec_count += 1
        self.matvec_time_seconds += time.perf_counter() - started
        return total / len(self.operators)
