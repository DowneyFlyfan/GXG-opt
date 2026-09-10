from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .polar import newton_schulz_polar


def spectral_exponent(p: float) -> float:
    if p < 2.0 and not math.isinf(p):
        raise ValueError("the configured Schatten p must be at least two")
    return 0.0 if math.isinf(p) else 1.0 / (p - 1.0)


def schatten_direction(
    matrix: torch.Tensor, p: float, eps: float = 1e-8
) -> torch.Tensor:
    """Exact energy-matched Schatten-p steepest geometry direction."""
    if matrix.ndim != 2:
        raise ValueError("Schatten geometry requires a matrix")
    matrix = matrix.float()
    rank = min(matrix.shape)
    if float(torch.linalg.vector_norm(matrix)) <= eps:
        return torch.zeros_like(matrix)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    phi = (u * (singular_values + eps).pow(spectral_exponent(p))) @ vh
    return math.sqrt(rank) * phi / (torch.linalg.vector_norm(phi) + eps)


def reduced_schatten_direction(
    matrix: torch.Tensor,
    p: float,
    *,
    ns_steps: int = 8,
    polynomial_degree: int = 12,
    polynomial_floor: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    rank = min(matrix.shape)
    if p == 2.0:
        return math.sqrt(rank) * matrix.float() / (
            torch.linalg.vector_norm(matrix.float()) + eps
        )
    if math.isinf(p):
        direction, _ = newton_schulz_polar(matrix, steps=ns_steps, eps=eps)
        return math.sqrt(rank) * direction / (
            torch.linalg.vector_norm(direction) + eps
        )
    matrix = matrix.float()
    scale_squared = torch.sum(matrix.square()) + eps
    left_action = matrix.shape[0] <= matrix.shape[1]
    gram = (
        matrix @ matrix.T / scale_squared
        if left_action
        else matrix.T @ matrix / scale_squared
    )
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    mapped = 2.0 * gram - identity
    power = 0.5 * (spectral_exponent(p) - 1.0)
    sample_count = polynomial_degree + 1
    indices = torch.arange(sample_count, device=matrix.device, dtype=torch.float32)
    angles = math.pi * (indices + 0.5) / sample_count
    nodes = 0.5 * (torch.cos(angles) + 1.0)
    values = (nodes + polynomial_floor).pow(power)
    coefficients = []
    for degree in range(sample_count):
        coefficient = 2.0 * torch.sum(values * torch.cos(degree * angles)) / sample_count
        coefficients.append(0.5 * coefficient if degree == 0 else coefficient)
    previous = matrix
    current = mapped @ matrix if left_action else matrix @ mapped
    phi = coefficients[0] * previous
    if polynomial_degree >= 1:
        phi = phi + coefficients[1] * current
    for degree in range(2, polynomial_degree + 1):
        following = (
            2.0 * mapped @ current - previous
            if left_action
            else 2.0 * current @ mapped - previous
        )
        phi = phi + coefficients[degree] * following
        previous, current = current, following
    return math.sqrt(rank) * phi / (torch.linalg.vector_norm(phi) + eps)


class SharedSpectralBasis:
    def __init__(self, matrix: torch.Tensor, eps: float = 1e-8) -> None:
        self.u, self.singular_values, self.vh = torch.linalg.svd(
            matrix.float(), full_matrices=False
        )
        self.eps = eps

    def direction(self, matrix: torch.Tensor, p: float) -> tuple[torch.Tensor, float]:
        coordinates = self.u.T @ matrix.float() @ self.vh.T
        diagonal = torch.diagonal(coordinates)
        signs = torch.sign(diagonal)
        values = signs * (torch.abs(diagonal) + self.eps).pow(spectral_exponent(p))
        phi = (self.u * values) @ self.vh
        direction = math.sqrt(min(matrix.shape)) * phi / (
            torch.linalg.vector_norm(phi) + self.eps
        )
        diagonal_part = torch.diag(diagonal)
        drift = float(
            torch.linalg.vector_norm(coordinates - diagonal_part)
            / (torch.linalg.vector_norm(coordinates) + self.eps)
        )
        return direction, drift


def fit_projected_curvature(
    steps: Sequence[torch.Tensor],
    gradient_changes: Sequence[torch.Tensor],
    *,
    basis_rank: int,
    ridge: float,
    min_curvature: float,
    max_curvature: float,
    perpendicular_curvature: float | str,
) -> dict[str, torch.Tensor | float] | None:
    if not steps:
        return None
    flat_steps = torch.stack([value.float().reshape(-1) for value in steps], dim=1)
    flat_changes = torch.stack(
        [value.float().reshape(-1) for value in gradient_changes], dim=1
    )
    source = torch.cat((flat_steps, flat_changes), dim=1)
    basis = torch.linalg.qr(source, mode="reduced").Q[:, :basis_rank]
    projected_steps = basis.T @ flat_steps
    projected_changes = basis.T @ flat_changes
    gram = projected_steps @ projected_steps.T
    cross = projected_changes @ projected_steps.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    raw = torch.linalg.solve((gram + ridge * identity).T, cross.T).T
    symmetric = 0.5 * (raw + raw.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clipped = torch.clamp(eigenvalues, min_curvature, max_curvature)
    curvature = (eigenvectors * clipped) @ eigenvectors.T
    if perpendicular_curvature == "median_secant":
        rayleigh = []
        for step, change in zip(flat_steps.T, flat_changes.T):
            denominator = torch.sum(step.square())
            if float(denominator) > 1e-12:
                rayleigh.append(float(torch.sum(step * change) / denominator))
        perpendicular = max(float(torch.tensor(rayleigh).median()), 0.0) if rayleigh else 0.0
    else:
        perpendicular = max(float(perpendicular_curvature), 0.0)
    fitted = basis @ (curvature @ projected_steps)
    residual = float(
        torch.linalg.vector_norm(fitted - flat_changes)
        / (torch.linalg.vector_norm(flat_changes) + 1e-8)
    )
    explained = float(
        torch.linalg.vector_norm(basis.T @ flat_steps).square()
        / (torch.linalg.vector_norm(flat_steps).square() + 1e-8)
    )
    return {
        "basis": basis,
        "curvature": curvature,
        "perpendicular": perpendicular,
        "fit_residual": residual,
        "explained_energy": explained,
        "eigenvalues": clipped,
    }


def apply_projected_curvature(
    model: dict[str, torch.Tensor | float] | None, vector: torch.Tensor
) -> torch.Tensor:
    if model is None:
        return torch.zeros_like(vector, dtype=torch.float32)
    flat = vector.float().reshape(-1)
    basis = model["basis"]
    curvature = model["curvature"]
    projected = basis.T @ flat
    parallel = basis @ (curvature @ projected)
    perpendicular = float(model["perpendicular"])
    result = parallel + perpendicular * (flat - basis @ projected)
    return result.reshape_as(vector)


def local_loss_increment(
    gradient: torch.Tensor,
    displacement: torch.Tensor,
    curvature_model: dict[str, torch.Tensor | float] | None,
) -> float:
    curvature_displacement = apply_projected_curvature(
        curvature_model, displacement
    )
    return float(
        torch.sum(gradient.float() * displacement.float())
        + 0.5 * torch.sum(displacement.float() * curvature_displacement)
    )


def geometry_distance(left: float, right: float) -> float:
    return abs(spectral_exponent(left) - spectral_exponent(right))


def rollout_geometry_candidates(
    gradient: torch.Tensor,
    momentum: torch.Tensor,
    curvature_model: dict[str, torch.Tensor | float] | None,
    candidates: Sequence[float],
    *,
    learning_rates: Sequence[float],
    horizon_weights: Sequence[float],
    beta: float,
    matrix_scale: float,
    selected_p: float,
    switch_penalty: float,
    compute_penalties: dict[float, float] | None = None,
    backend: str = "exact_svd",
    ns_steps: int = 8,
    polynomial_degree: int = 12,
    polynomial_floor: float = 1e-4,
    eps: float = 1e-8,
) -> tuple[dict[float, float], dict[float, list[float]], dict[float, float]]:
    if len(learning_rates) != len(horizon_weights):
        raise ValueError("learning rates and horizon weights must match")
    compute_penalties = compute_penalties or {}
    shared = SharedSpectralBasis(momentum, eps) if backend == "shared_svd" else None
    scores = {}
    increments = {}
    basis_drift = {}
    for candidate in candidates:
        predicted_gradient = gradient.detach().float().clone()
        predicted_momentum = momentum.detach().float().clone()
        candidate_increments = []
        drift_values = []
        for lr, weight in zip(learning_rates, horizon_weights):
            if backend == "exact_svd":
                transform = schatten_direction(predicted_momentum, candidate, eps)
            elif backend == "shared_svd":
                transform, drift = shared.direction(predicted_momentum, candidate)
                drift_values.append(drift)
            elif backend == "reduced":
                transform = reduced_schatten_direction(
                    predicted_momentum,
                    candidate,
                    ns_steps=ns_steps,
                    polynomial_degree=polynomial_degree,
                    polynomial_floor=polynomial_floor,
                    eps=eps,
                )
            else:
                raise ValueError(f"unsupported spectral backend: {backend}")
            displacement = -lr * matrix_scale * transform
            increment = local_loss_increment(
                predicted_gradient, displacement, curvature_model
            )
            candidate_increments.append(increment)
            predicted_gradient = predicted_gradient + apply_projected_curvature(
                curvature_model, displacement
            )
            predicted_momentum = (
                beta * predicted_momentum + (1.0 - beta) * predicted_gradient
            )
        scores[candidate] = sum(
            weight * increment
            for weight, increment in zip(horizon_weights, candidate_increments)
        ) + switch_penalty * geometry_distance(candidate, selected_p) ** 2 + float(
            compute_penalties.get(candidate, 0.0)
        )
        increments[candidate] = candidate_increments
        basis_drift[candidate] = max(drift_values, default=0.0)
    return scores, increments, basis_drift
