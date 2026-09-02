"""Compute-efficient matrix-free Gauss--Newton directions.

The module never forms a curvature matrix or its inverse.  It approximates the
inverse action with a factored-diagonal Kronecker base, a signed low-rank
relative-residual correction, and a short preconditioned conjugate-gradient
solve.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


def _positive_epsilon(value: Tensor) -> Tensor:
    return torch.as_tensor(torch.finfo(value.dtype).eps, device=value.device, dtype=value.dtype)


def _factored_block_diagonal(block: Tensor) -> Tensor:
    """Return a rank-one row/column second-moment approximation."""
    if block.ndim < 2:
        return block.square()
    matrix = block.reshape(block.shape[0], -1)
    squared = matrix.square()
    row_moment = squared.mean(dim=1)
    column_moment = squared.mean(dim=0)
    normalizer = squared.mean().clamp_min(_positive_epsilon(block))
    return (row_moment[:, None] * column_moment[None, :] / normalizer).reshape(-1)


class FactoredKroneckerAccumulator:
    """Accumulate factored gradient second moments before batch averaging."""

    def __init__(
        self,
        *,
        parameter_shapes: Sequence[torch.Size | tuple[int, ...]],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.parameter_shapes = tuple(tuple(shape) for shape in parameter_shapes)
        self.device = device
        self.dtype = dtype
        self._statistics: list[tuple[Tensor, Tensor] | Tensor] = []
        for shape in self.parameter_shapes:
            if len(shape) >= 2:
                rows = shape[0]
                columns = math.prod(shape[1:])
                self._statistics.append(
                    (
                        torch.zeros(rows, device=device, dtype=dtype),
                        torch.zeros(columns, device=device, dtype=dtype),
                    )
                )
            else:
                self._statistics.append(
                    torch.zeros(shape, device=device, dtype=dtype)
                )

    @torch.no_grad()
    def update_scaled_block(
        self, index: int, scaled_gradient: Tensor, *, batch_weight: float
    ) -> None:
        if not 0 < batch_weight <= 1:
            raise ValueError("batch_weight must lie in (0, 1]")
        shape = self.parameter_shapes[index]
        if tuple(scaled_gradient.shape) != shape:
            raise ValueError("scaled gradient shape does not match its parameter")
        # Backpropagating batch_weight * loss exposes batch_weight * g in a
        # hook.  Dividing its square by the weight yields weight * g^2, the
        # desired contribution to E[g^2].
        squared = scaled_gradient.detach().to(self.dtype).square() / batch_weight
        statistics = self._statistics[index]
        if isinstance(statistics, tuple):
            matrix = squared.reshape(shape[0], -1)
            statistics[0].add_(matrix.mean(dim=1))
            statistics[1].add_(matrix.mean(dim=0))
        else:
            statistics.add_(squared)

    def diagonal(self, minimum_diagonal_ratio: float = 1.0e-6) -> Tensor:
        if not 0 < minimum_diagonal_ratio <= 1:
            raise ValueError("minimum_diagonal_ratio must lie in (0, 1]")
        blocks = []
        for statistics in self._statistics:
            if isinstance(statistics, tuple):
                row, column = statistics
                normalizer = row.mean().clamp_min(_positive_epsilon(row))
                blocks.append((row[:, None] * column[None, :] / normalizer).reshape(-1))
            else:
                blocks.append(statistics.reshape(-1))
        diagonal = torch.cat(tuple(blocks))
        floor = (
            diagonal.mean().abs().clamp_min(_positive_epsilon(diagonal))
            * minimum_diagonal_ratio
        )
        return diagonal.clamp_min(floor)


def _split_flat_vector(vector: Tensor, parameter_shapes: Sequence[torch.Size | tuple[int, ...]]):
    offset = 0
    for shape in parameter_shapes:
        shape = tuple(shape)
        size = 1
        for dimension in shape:
            size *= dimension
        yield vector[offset : offset + size].reshape(shape)
        offset += size
    if offset != vector.numel():
        raise ValueError("parameter shapes do not cover the flat vector")


def build_factored_kronecker_diagonal(
    gradient: Tensor,
    *,
    parameter_shapes: Sequence[torch.Size | tuple[int, ...]],
    curvature_matvec: Callable[[Tensor], Tensor],
    damping: float,
    minimum_diagonal_ratio: float = 1.0e-6,
    accumulated_second_moment_diagonal: Tensor | None = None,
    curvature_seed: Tensor | None = None,
) -> Tensor:
    """Build a cheap diagonal Kronecker approximation to ``G + damping I``.

    Matrix-shaped parameter blocks use a factored row/column second moment.
    The global scale is calibrated with one exact matrix-free curvature product
    along the normalized gradient direction.
    """
    if gradient.ndim != 1:
        raise ValueError("gradient must be flat")
    if gradient.numel() == 0:
        raise ValueError("gradient must be non-empty")
    if damping <= 0:
        raise ValueError("damping must be positive")
    if not 0 < minimum_diagonal_ratio <= 1:
        raise ValueError("minimum_diagonal_ratio must lie in (0, 1]")
    if accumulated_second_moment_diagonal is None:
        raw_blocks = tuple(
            _factored_block_diagonal(block)
            for block in _split_flat_vector(gradient, parameter_shapes)
        )
        raw = torch.cat(tuple(block.reshape(-1) for block in raw_blocks))
        raw_floor = (
            raw.mean().abs().clamp_min(_positive_epsilon(raw))
            * minimum_diagonal_ratio
        )
        raw = raw.clamp_min(raw_floor)
    else:
        if accumulated_second_moment_diagonal.shape != gradient.shape:
            raise ValueError("accumulated second-moment diagonal has the wrong shape")
        raw = accumulated_second_moment_diagonal.to(gradient)
    gradient_norm = gradient.norm()
    if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
        raise ValueError("gradient must have a finite non-zero norm")
    seed = gradient / gradient_norm
    if curvature_seed is None:
        curvature_seed = curvature_matvec(seed)
    elif curvature_seed.shape != gradient.shape:
        raise ValueError("curvature_seed has the wrong shape")
    curvature_seed = curvature_seed.to(gradient)
    curvature_rayleigh = torch.dot(seed, curvature_seed).clamp_min(0)
    raw_rayleigh = torch.dot(seed, raw * seed).clamp_min(_positive_epsilon(raw))
    scale = curvature_rayleigh / raw_rayleigh
    diagonal = scale * raw + damping
    if not torch.isfinite(diagonal).all() or not torch.all(diagonal > 0):
        raise RuntimeError("factored Kronecker diagonal is not finite and positive")
    return diagonal


@dataclass(frozen=True)
class LowRankResidualCorrection:
    """Signed correction to a diagonal base inverse in whitened coordinates."""

    basis: Tensor
    projected_matrix: Tensor
    minimum_relative_eigenvalue: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.basis.ndim != 2:
            raise ValueError("basis must be a matrix")
        if self.projected_matrix.shape != (self.basis.shape[1], self.basis.shape[1]):
            raise ValueError("projected matrix shape does not match the basis")
        if not 0 < self.minimum_relative_eigenvalue <= 1:
            raise ValueError("minimum_relative_eigenvalue must lie in (0, 1]")

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    def apply(self, residual: Tensor, base_diagonal: Tensor) -> Tensor:
        if residual.ndim != 1 or residual.shape != base_diagonal.shape:
            raise ValueError("residual and base diagonal must be equal flat vectors")
        if self.basis.shape[0] != residual.numel():
            raise ValueError("basis dimension does not match the residual")
        inverse_square_root = base_diagonal.rsqrt()
        whitened = inverse_square_root * residual
        basis = self.basis.to(device=residual.device)
        projected = self.projected_matrix.to(device=residual.device, dtype=residual.dtype)
        projected = 0.5 * (projected + projected.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(projected)
        clipped = eigenvalues.clamp_min(-1.0 + self.minimum_relative_eigenvalue)
        inverse_adjustment = clipped.add(1.0).reciprocal() - 1.0
        # A rank-four basis for the 54.7M-parameter model is large enough that
        # materializing an FP32 copy defeats the memory saving.  Accumulate in
        # FP32 while reading a BF16 basis in bounded chunks.
        chunk_size = 1_048_576
        coefficients = torch.zeros(
            basis.shape[1], device=residual.device, dtype=residual.dtype
        )
        for start in range(0, basis.shape[0], chunk_size):
            stop = min(start + chunk_size, basis.shape[0])
            chunk = basis[start:stop].to(dtype=residual.dtype)
            coefficients.add_(chunk.T @ whitened[start:stop])
        rotated = eigenvectors.T @ coefficients
        small_correction = eigenvectors @ (inverse_adjustment * rotated)
        correction = torch.empty_like(whitened)
        for start in range(0, basis.shape[0], chunk_size):
            stop = min(start + chunk_size, basis.shape[0])
            chunk = basis[start:stop].to(dtype=residual.dtype)
            correction[start:stop] = chunk @ small_correction
        result = inverse_square_root * (whitened + correction)
        if not torch.isfinite(result).all():
            raise RuntimeError("low-rank preconditioner produced a non-finite value")
        return result


def build_signed_relative_residual(
    system_matvec: Callable[[Tensor], Tensor],
    base_diagonal: Tensor,
    *,
    seed: Tensor,
    rank: int,
    breakdown_tolerance: float = 1.0e-10,
    minimum_relative_eigenvalue: float = 1.0e-3,
    storage_dtype: torch.dtype | None = None,
) -> LowRankResidualCorrection:
    """Lanczos approximation of ``M^-1/2 A M^-1/2 - I``.

    The projected matrix is signed: negative residual eigenvalues are retained,
    unlike a positive-semidefinite Nyström approximation.
    """
    if base_diagonal.ndim != 1 or seed.shape != base_diagonal.shape:
        raise ValueError("base diagonal and seed must be equal flat vectors")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if breakdown_tolerance <= 0:
        raise ValueError("breakdown_tolerance must be positive")
    inverse_square_root = base_diagonal.rsqrt()

    def residual_matvec(vector: Tensor) -> Tensor:
        unwhitened = inverse_square_root * vector
        return inverse_square_root * system_matvec(unwhitened) - vector

    normalized_seed = seed / seed.norm().clamp_min(_positive_epsilon(seed))
    if not torch.isfinite(normalized_seed).all():
        raise ValueError("seed must be finite")
    storage_dtype = seed.dtype if storage_dtype is None else storage_dtype
    basis = torch.empty(
        (seed.numel(), min(rank, seed.numel())),
        device=seed.device,
        dtype=storage_dtype,
    )
    basis_count = 0
    diagonal: list[Tensor] = []
    off_diagonal: list[Tensor] = []
    previous = torch.zeros_like(normalized_seed)
    previous_beta = torch.zeros((), device=seed.device, dtype=seed.dtype)
    current = normalized_seed
    maximum_rank = min(rank, seed.numel())
    for index in range(maximum_rank):
        basis[:, index].copy_(current)
        basis_count += 1
        product = residual_matvec(current)
        alpha = torch.dot(current, product)
        residual = product - alpha * current - previous_beta * previous
        # Two passes are cheap at the intended small rank and prevent loss of
        # orthogonality for nearly repeated residual eigenvalues.
        for _ in range(2):
            for basis_index in range(basis_count):
                basis_vector = basis[:, basis_index].to(dtype=seed.dtype)
                residual = residual - torch.dot(basis_vector, residual) * basis_vector
        diagonal.append(alpha)
        if index + 1 == maximum_rank:
            break
        beta = residual.norm()
        if not torch.isfinite(beta):
            raise RuntimeError("Lanczos residual is non-finite")
        if beta <= breakdown_tolerance:
            break
        off_diagonal.append(beta)
        previous = current
        current = residual / beta
        previous_beta = beta

    basis = basis[:, :basis_count]
    projected = torch.diag(torch.stack(diagonal))
    if off_diagonal:
        values = torch.stack(off_diagonal)
        indices = torch.arange(values.numel(), device=values.device)
        projected[indices, indices + 1] = values
        projected[indices + 1, indices] = values
    return LowRankResidualCorrection(
        basis=basis,
        projected_matrix=projected,
        minimum_relative_eigenvalue=minimum_relative_eigenvalue,
    )


@dataclass
class RecycledResidualState:
    rank: int
    refresh_interval: int
    minimum_relative_eigenvalue: float = 1.0e-3
    storage_dtype: torch.dtype | None = None
    correction: LowRankResidualCorrection | None = None
    built_at_step: int | None = None

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.refresh_interval <= 0:
            raise ValueError("refresh_interval must be positive")

    def get_or_build(
        self,
        *,
        outer_step: int,
        system_matvec: Callable[[Tensor], Tensor],
        base_diagonal: Tensor,
        seed: Tensor,
    ) -> tuple[LowRankResidualCorrection, bool]:
        if outer_step < 0:
            raise ValueError("outer_step must be non-negative")
        refresh = (
            self.correction is None
            or self.built_at_step is None
            or outer_step - self.built_at_step >= self.refresh_interval
        )
        if refresh:
            self.correction = build_signed_relative_residual(
                system_matvec,
                base_diagonal,
                seed=seed,
                rank=self.rank,
                minimum_relative_eigenvalue=self.minimum_relative_eigenvalue,
                storage_dtype=self.storage_dtype,
            )
            self.built_at_step = outer_step
        assert self.correction is not None
        return self.correction, refresh

    def update_from_secant(
        self,
        *,
        outer_step: int,
        base_diagonal: Tensor,
        seed: Tensor,
        system_seed: Tensor,
    ) -> tuple[LowRankResidualCorrection, float]:
        """Update a recycled projected residual from one accumulated ``A seed``.

        In whitened coordinates, ``x=M^{1/2} seed`` and
        ``y=M^{-1/2} A seed-x`` provide one secant observation ``R x=y`` for
        ``R=M^{-1/2} A M^{-1/2}-I``.  The method adds one orthogonal direction
        per outer step, retains a symmetric projected operator, and truncates
        to the configured rank by residual-eigenvalue magnitude.
        """
        if outer_step < 0:
            raise ValueError("outer_step must be non-negative")
        if base_diagonal.ndim != 1 or seed.shape != base_diagonal.shape:
            raise ValueError("base diagonal and seed must be equal flat vectors")
        if system_seed.shape != seed.shape:
            raise ValueError("system_seed has the wrong shape")
        if not torch.all(base_diagonal > 0):
            raise ValueError("base diagonal must be positive")
        square_root = base_diagonal.sqrt()
        inverse_square_root = base_diagonal.rsqrt()
        scale = (square_root * seed).norm().clamp_min(_positive_epsilon(seed))
        x = square_root * seed / scale
        y = inverse_square_root * system_seed / scale - x
        storage_dtype = seed.dtype if self.storage_dtype is None else self.storage_dtype

        if self.correction is None:
            basis = x[:, None].to(storage_dtype)
            projected = torch.dot(x, y).reshape(1, 1)
        else:
            old_basis = self.correction.basis.to(device=seed.device)
            old_projected = self.correction.projected_matrix.to(seed)
            old_projected = 0.5 * (old_projected + old_projected.T)
            basis_fp = old_basis.to(dtype=seed.dtype)
            coefficients = basis_fp.T @ x
            remainder = x - basis_fp @ coefficients
            remainder_norm = remainder.norm()
            threshold = 32 * _positive_epsilon(seed).sqrt()
            projected_response = basis_fp.T @ y
            if remainder_norm <= threshold:
                denominator = torch.dot(coefficients, coefficients).clamp_min(
                    _positive_epsilon(seed)
                )
                error = projected_response - old_projected @ coefficients
                cross = torch.dot(coefficients, error)
                update = (
                    torch.outer(error, coefficients)
                    + torch.outer(coefficients, error)
                ) / denominator
                update -= (
                    cross
                    * torch.outer(coefficients, coefficients)
                    / denominator.square()
                )
                basis = old_basis
                projected = old_projected + update
            else:
                new_basis_vector = remainder / remainder_norm
                cross = (
                    projected_response - old_projected @ coefficients
                ) / remainder_norm
                diagonal = (
                    torch.dot(new_basis_vector, y)
                    - torch.dot(cross, coefficients)
                ) / remainder_norm
                basis = torch.cat(
                    (old_basis, new_basis_vector[:, None].to(storage_dtype)), dim=1
                )
                projected = torch.empty(
                    (old_projected.shape[0] + 1, old_projected.shape[1] + 1),
                    device=seed.device,
                    dtype=seed.dtype,
                )
                projected[:-1, :-1] = old_projected
                projected[:-1, -1] = cross
                projected[-1, :-1] = cross
                projected[-1, -1] = diagonal

        if basis.shape[1] > self.rank:
            eigenvalues, eigenvectors = torch.linalg.eigh(
                0.5 * (projected + projected.T)
            )
            keep = torch.topk(eigenvalues.abs(), self.rank).indices
            rotation = eigenvectors[:, keep]
            basis = (basis.to(seed.dtype) @ rotation).to(storage_dtype)
            projected = torch.diag(eigenvalues[keep])
        eigenvalues, eigenvectors = torch.linalg.eigh(
            0.5 * (projected + projected.T)
        )
        eigenvalues = eigenvalues.clamp_min(
            -1.0 + self.minimum_relative_eigenvalue
        )
        projected = (eigenvectors * eigenvalues) @ eigenvectors.T
        correction = LowRankResidualCorrection(
            basis=basis,
            projected_matrix=projected,
            minimum_relative_eigenvalue=self.minimum_relative_eigenvalue,
        )
        basis_fp = basis.to(seed.dtype)
        predicted = basis_fp @ (projected @ (basis_fp.T @ x))
        fit_residual = float(
            (y - predicted).norm().div(y.norm().clamp_min(_positive_epsilon(y))).item()
        )
        self.correction = correction
        self.built_at_step = outer_step
        return correction, fit_residual


@dataclass(frozen=True)
class PCGResult:
    direction: Tensor
    iterations: int
    residual_norm: float
    relative_residual: float
    residual_history: tuple[float, ...]


def pcg_solve(
    system_matvec: Callable[[Tensor], Tensor],
    right_hand_side: Tensor,
    *,
    preconditioner: Callable[[Tensor], Tensor],
    maximum_iterations: int,
    relative_tolerance: float = 1.0e-3,
    initial_direction: Tensor | None = None,
) -> PCGResult:
    """Solve an SPD system using a small number of matrix-free PCG steps."""
    if right_hand_side.ndim != 1:
        raise ValueError("right_hand_side must be flat")
    if maximum_iterations <= 0:
        raise ValueError("maximum_iterations must be positive")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    direction = (
        torch.zeros_like(right_hand_side)
        if initial_direction is None
        else initial_direction.detach().to(right_hand_side).clone()
    )
    residual = right_hand_side - system_matvec(direction)
    rhs_norm = right_hand_side.norm().clamp_min(_positive_epsilon(right_hand_side))
    residual_norm = residual.norm()
    relative_residual = residual_norm / rhs_norm
    history = [float(relative_residual.item())]
    if relative_residual <= relative_tolerance:
        return PCGResult(
            direction,
            0,
            float(residual_norm.item()),
            float(relative_residual.item()),
            tuple(history),
        )
    preconditioned = preconditioner(residual)
    residual_inner = torch.dot(residual, preconditioned)
    if not torch.isfinite(residual_inner) or residual_inner <= 0:
        raise RuntimeError("Preconditioner must be positive definite")
    search = preconditioned.clone()
    for iteration in range(1, maximum_iterations + 1):
        system_search = system_matvec(search)
        denominator = torch.dot(search, system_search)
        if not torch.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("Gauss--Newton system must be positive definite")
        step_size = residual_inner / denominator
        direction = direction + step_size * search
        residual = residual - step_size * system_search
        residual_norm = residual.norm()
        relative_residual = residual_norm / rhs_norm
        history.append(float(relative_residual.item()))
        if not torch.isfinite(relative_residual):
            raise RuntimeError("PCG residual is non-finite")
        if relative_residual <= relative_tolerance:
            return PCGResult(
                direction,
                iteration,
                float(residual_norm.item()),
                float(relative_residual.item()),
                tuple(history),
            )
        next_preconditioned = preconditioner(residual)
        next_inner = torch.dot(residual, next_preconditioned)
        if not torch.isfinite(next_inner) or next_inner <= 0:
            raise RuntimeError("Preconditioner must be positive definite")
        search = next_preconditioned + (next_inner / residual_inner) * search
        residual_inner = next_inner
    return PCGResult(
        direction,
        maximum_iterations,
        float(residual_norm.item()),
        float(relative_residual.item()),
        tuple(history),
    )
