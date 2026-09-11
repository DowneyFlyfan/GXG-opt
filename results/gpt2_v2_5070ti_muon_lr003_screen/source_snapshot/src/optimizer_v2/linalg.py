from __future__ import annotations

import torch


class ProbeFailure(RuntimeError):
    """An auxiliary numerical failure permits a baseline-only commit."""


def finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ProbeFailure("nonfinite auxiliary statistic")


def orthonormalize(vectors, tolerance=1e-10):
    basis = []
    for vector in vectors:
        residual = vector.double().clone()
        original = torch.linalg.vector_norm(residual)
        if float(original) == 0:
            continue
        for _ in range(2):
            for column in basis:
                residual -= torch.dot(column, residual) * column
        norm = torch.linalg.vector_norm(residual)
        if float(norm) > tolerance * float(original):
            basis.append(residual / norm)
    return basis


def psd(matrix):
    matrix = (matrix.double() + matrix.double().T) * 0.5
    finite(matrix)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    scale = max(float(eigenvalues.abs().max()), 1e-12)
    if float(eigenvalues.min()) < -1e-6 * scale:
        raise ValueError("materially indefinite joint metric")
    return matrix


def solve_spd(matrix, rhs):
    finite(matrix, rhs)
    factor, info = torch.linalg.cholesky_ex(matrix.double())
    if int(info) != 0:
        raise ProbeFailure("auxiliary Cholesky failed")
    result = torch.cholesky_solve(rhs.double().reshape(matrix.shape[0], -1), factor)
    finite(result)
    return result.reshape(rhs.shape)


def proximal_coefficients(gram, projected, rho=1.0, kappa=None):
    gram = psd(gram)
    largest = float(torch.linalg.eigvalsh(gram)[-1])
    if largest <= 0 or rho == 0:
        return torch.zeros_like(projected), 0.0
    strength = rho / largest if kappa is None else kappa
    system = torch.eye(len(gram), device=gram.device, dtype=torch.float64) + strength * gram
    coefficients = strength * solve_spd(system, projected)
    return coefficients, strength


def low_rank_prox(increment, columns, rho=1.0, kappa=None):
    """Woodbury action; only the tiny Gram and solve are promoted to FP64."""
    projected = torch.stack([(column * increment).sum() for column in columns]).double()
    gram = torch.stack([
        torch.stack([(left * right).sum() for right in columns]) for left in columns
    ]).double()
    coefficients, strength = proximal_coefficients(gram, projected, rho, kappa)
    result = increment.clone()
    for coefficient, column in zip(coefficients, columns):
        result.add_(column, alpha=-float(coefficient))
    return result, {"kappa": strength, "gram_eigenvalues": torch.linalg.eigvalsh(gram).tolist()}
