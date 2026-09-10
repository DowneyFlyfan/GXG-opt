"""Muon updates constrained to square orthogonal and Stiefel manifolds.

The square route implements the analytic direction from Scientific Spaces
article 11215.  The tall route implements the closed-form skew generator from
article 11864.  Wide weights are treated as the transpose of a tall Stiefel
matrix, so every matrix has orthonormal columns in its logical orientation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - CPU-only environments use the torch path.
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _triton_ns_polynomial_kernel(
        value,
        polynomial,
        output,
        rows: tl.constexpr,
        columns: tl.constexpr,
        value_stride_row: tl.constexpr,
        value_stride_column: tl.constexpr,
        polynomial_stride_row: tl.constexpr,
        polynomial_stride_column: tl.constexpr,
        output_stride_row: tl.constexpr,
        output_stride_column: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLUMNS: tl.constexpr,
        BLOCK_REDUCTION: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        column_offsets = tl.program_id(1) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
        accumulator = tl.zeros((BLOCK_ROWS, BLOCK_COLUMNS), dtype=tl.float32)
        for reduction_start in range(0, columns, BLOCK_REDUCTION):
            reduction_offsets = reduction_start + tl.arange(0, BLOCK_REDUCTION)
            left = tl.load(
                value
                + row_offsets[:, None] * value_stride_row
                + reduction_offsets[None, :] * value_stride_column,
                mask=(row_offsets[:, None] < rows)
                & (reduction_offsets[None, :] < columns),
                other=0.0,
            )
            right = tl.load(
                polynomial
                + reduction_offsets[:, None] * polynomial_stride_row
                + column_offsets[None, :] * polynomial_stride_column,
                mask=(reduction_offsets[:, None] < columns)
                & (column_offsets[None, :] < columns),
                other=0.0,
            )
            accumulator += tl.dot(left, right, input_precision="ieee")
        original = tl.load(
            value
            + row_offsets[:, None] * value_stride_row
            + column_offsets[None, :] * value_stride_column,
            mask=(row_offsets[:, None] < rows) & (column_offsets[None, :] < columns),
            other=0.0,
        )
        tl.store(
            output
            + row_offsets[:, None] * output_stride_row
            + column_offsets[None, :] * output_stride_column,
            accumulator + 3.4445 * original,
            mask=(row_offsets[:, None] < rows) & (column_offsets[None, :] < columns),
        )


def triton_newton_schulz_polynomial(
    value: Tensor, gram: Tensor, gram_squared: Tensor
) -> Tensor:
    """Fused Triton implementation of ``3.4445 X + X(-4.775G + 2.0315G²)``."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if (
        value.ndim != 2
        or gram.shape != (value.shape[1], value.shape[1])
        or gram_squared.shape != gram.shape
        or not value.is_cuda
    ):
        raise ValueError("Triton Newton--Schulz inputs must be CUDA-compatible matrices")
    source = value.float().contiguous()
    polynomial = (-4.775 * gram.float() + 2.0315 * gram_squared.float()).contiguous()
    output = torch.empty_like(source)
    block_rows = 32
    block_columns = 32
    block_reduction = 32
    grid = (triton.cdiv(source.shape[0], block_rows), triton.cdiv(source.shape[1], block_columns))
    _triton_ns_polynomial_kernel[grid](
        source,
        polynomial,
        output,
        source.shape[0],
        source.shape[1],
        source.stride(0),
        source.stride(1),
        polynomial.stride(0),
        polynomial.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_ROWS=block_rows,
        BLOCK_COLUMNS=block_columns,
        BLOCK_REDUCTION=block_reduction,
    )
    return output.to(value.dtype)


def _matrix_sign(
    matrix: Tensor,
    *,
    ns_steps: int,
    epsilon: float = 1.0e-7,
    use_triton: bool = True,
) -> Tensor:
    """Approximate the polar matrix sign with the Muon Newton--Schulz iteration."""
    if matrix.ndim != 2:
        raise ValueError("matrix sign requires a two-dimensional matrix")
    if ns_steps <= 0:
        raise ValueError("ns_steps must be positive")
    source_dtype = matrix.dtype
    value = matrix.float()
    value = value / value.norm().clamp_min(epsilon)
    for _ in range(ns_steps):
        gram = value.T @ value
        gram_squared = gram @ gram
        if (
            use_triton
            and _TRITON_AVAILABLE
            and value.is_cuda
            and value.shape[0] == value.shape[1]
            and value.shape[0] <= 512
        ):
            value = triton_newton_schulz_polynomial(value, gram, gram_squared)
        else:
            value = 3.4445 * value + value @ (-4.775 * gram + 2.0315 * gram_squared)
    return value.to(source_dtype)


def initialize_stiefel_matrix(matrix: Tensor) -> Tensor:
    """Project a tall matrix to column-orthonormal columns through QR."""
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("Stiefel initialization requires rows >= columns")
    orthogonal, triangular = torch.linalg.qr(matrix.float(), mode="reduced")
    signs = torch.where(
        torch.diagonal(triangular) < 0,
        -torch.ones(matrix.shape[1], dtype=orthogonal.dtype, device=matrix.device),
        torch.ones(matrix.shape[1], dtype=orthogonal.dtype, device=matrix.device),
    )
    return (orthogonal * signs).to(matrix.dtype)


def _column_retraction(candidate: Tensor, *, ns_steps: int) -> Tensor:
    """Apply the polar retraction and fall back to QR only for numeric drift."""
    retracted = _matrix_sign(candidate, ns_steps=ns_steps)
    identity = torch.eye(
        candidate.shape[1], dtype=retracted.dtype, device=retracted.device
    )
    error = (retracted.T @ retracted - identity).norm()
    if not torch.isfinite(error) or error > 1.0e-3:
        return initialize_stiefel_matrix(candidate)
    return retracted


def square_closed_form_retraction(
    weight: Tensor, rotation: Tensor, *, step_size: float
) -> Tensor:
    """Use the analytic orthogonal retraction from article 11215, Eq. (16)."""
    if (
        weight.ndim != 2
        or weight.shape[0] != weight.shape[1]
        or rotation.shape != weight.shape
    ):
        raise ValueError("square retraction requires equally shaped square matrices")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    identity = torch.eye(weight.shape[0], dtype=weight.dtype, device=weight.device)
    support_projection = rotation.T @ rotation
    normalization = identity - support_projection + support_projection / math.sqrt(
        1.0 + step_size**2
    )
    return weight @ ((identity - step_size * rotation) @ normalization)


def square_stiefel_update(
    weight: Tensor, momentum: Tensor, *, step_size: float, ns_steps: int
) -> tuple[Tensor, Tensor]:
    """Article-11215 square orthogonal update and polar retraction."""
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("square update requires a square matrix")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    skew = 0.5 * (weight.T @ momentum - momentum.T @ weight)
    rotation = _matrix_sign(skew, ns_steps=ns_steps)
    rotation = 0.5 * (rotation - rotation.T)
    direction = weight @ rotation
    candidate = weight - step_size * direction
    retracted = square_closed_form_retraction(weight, rotation, step_size=step_size)
    identity = torch.eye(weight.shape[1], dtype=retracted.dtype, device=retracted.device)
    error = (retracted.T @ retracted - identity).norm()
    if not torch.isfinite(error) or error > 1.0e-3:
        retracted = _column_retraction(candidate, ns_steps=ns_steps)
    return retracted, direction


def rectangular_stiefel_update(
    weight: Tensor, momentum: Tensor, *, step_size: float, ns_steps: int
) -> tuple[Tensor, Tensor]:
    """Article-11864 closed-form Stiefel direction and polar retraction."""
    if weight.ndim != 2 or weight.shape[0] < weight.shape[1]:
        raise ValueError("rectangular update requires a tall matrix")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    rows, columns = weight.shape
    if rows >= 2 * columns:
        basis, triangular = torch.linalg.qr(
            torch.cat((momentum, weight), dim=1).float(), mode="reduced"
        )
        identity = torch.eye(columns, device=weight.device, dtype=basis.dtype)
        zero = torch.zeros_like(identity)
        skew_block = torch.cat(
            (torch.cat((zero, identity), dim=1), torch.cat((-identity, zero), dim=1)),
            dim=0,
        )
        small_skew = triangular @ skew_block @ triangular.T
        generator_small = _matrix_sign(small_skew, ns_steps=ns_steps)
        generator_small = 0.5 * (generator_small - generator_small.T)
        direction = basis @ (generator_small @ (basis.T @ weight.float()))
        direction = direction.to(weight.dtype)
    else:
        generator = momentum @ weight.T - weight @ momentum.T
        generator = _matrix_sign(generator, ns_steps=ns_steps)
        generator = 0.5 * (generator - generator.T)
        direction = generator @ weight
    candidate = weight - step_size * direction
    return _column_retraction(candidate, ns_steps=ns_steps), direction


def stiefel_update(
    parameter: Tensor, gradient: Tensor, *, step_size: float, ns_steps: int
) -> tuple[Tensor, Tensor, str]:
    """Route a stored matrix to square, column-, or row-Stiefel geometry."""
    if parameter.ndim != 2 or gradient.shape != parameter.shape:
        raise ValueError("Stiefel update requires equal two-dimensional tensors")
    if parameter.shape[0] == parameter.shape[1]:
        updated, direction = square_stiefel_update(
            parameter, gradient, step_size=step_size, ns_steps=ns_steps
        )
        return updated, direction, "square"
    if parameter.shape[0] > parameter.shape[1]:
        updated, direction = rectangular_stiefel_update(
            parameter, gradient, step_size=step_size, ns_steps=ns_steps
        )
        return updated, direction, "column_stiefel"
    updated, direction = rectangular_stiefel_update(
        parameter.T, gradient.T, step_size=step_size, ns_steps=ns_steps
    )
    return updated.T, direction.T, "row_stiefel"


@torch.no_grad()
def initialize_stiefel_parameters(parameters: Iterable[Tensor]) -> None:
    """Initialize each stored parameter on its corresponding Stiefel manifold."""
    for parameter in parameters:
        if parameter.ndim != 2:
            raise ValueError("Stiefel-Muon accepts only matrix parameters")
        if parameter.shape[0] >= parameter.shape[1]:
            parameter.copy_(initialize_stiefel_matrix(parameter))
        else:
            parameter.copy_(initialize_stiefel_matrix(parameter.T).T)


class StiefelMuon(torch.optim.Optimizer):
    """Momentum Muon with the blog's EMA-gradient Stiefel direction."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps),
        )
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.ndim != 2:
                        raise ValueError("Stiefel-Muon accepts only matrix parameters")
                    logical = parameter if parameter.shape[0] >= parameter.shape[1] else parameter.T
                    scale = logical.norm() / math.sqrt(logical.shape[1])
                    if not torch.isfinite(scale) or scale <= 0:
                        raise ValueError("Stiefel-Muon requires a finite non-zero parameter scale")
                    logical.copy_(initialize_stiefel_matrix(logical) * scale)
                    self.state[parameter]["stiefel_scale"] = float(scale)

    @staticmethod
    def scaled_lr(lr: float, rows: int, columns: int) -> float:
        return lr * 0.2 * math.sqrt(max(rows, columns))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim != 2:
                    raise ValueError("Stiefel-Muon received a non-matrix parameter")
                gradient = parameter.grad
                state = self.state[parameter]
                buffer = state.setdefault("momentum_buffer", torch.zeros_like(parameter))
                buffer.mul_(group["momentum"]).add_(gradient)
                momentum = (
                    gradient.add(buffer, alpha=group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                rows, columns = parameter.shape
                scale = float(state["stiefel_scale"])
                updated, _, route = stiefel_update(
                    parameter / scale,
                    momentum,
                    step_size=self.scaled_lr(group["lr"], rows, columns),
                    ns_steps=group["ns_steps"],
                )
                parameter.copy_(scale * updated)
                state["route"] = route
        return loss
