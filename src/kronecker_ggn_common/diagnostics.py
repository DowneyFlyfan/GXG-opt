from __future__ import annotations

import torch
from torch import Tensor


def tensor_bytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


def cosine_similarity(left: Tensor, right: Tensor, epsilon: float = 1.0e-12) -> float:
    denominator = float(left.norm().item() * right.norm().item()) + epsilon
    return float((left * right).sum().item()) / denominator


def relative_difference(
    value: Tensor, reference: Tensor, epsilon: float = 1.0e-12
) -> float:
    return float((value - reference).norm().item()) / (
        float(reference.norm().item()) + epsilon
    )


def quadratic_system_residual(
    matvec, direction: Tensor, gradient: Tensor, epsilon: float = 1.0e-12
) -> float:
    residual = matvec(direction) + gradient
    return float(residual.norm().item()) / (float(gradient.norm().item()) + epsilon)


def device_peak_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mb": 0.0, "peak_reserved_mb": 0.0}
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def require_finite(value: Tensor, label: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{label} contains non-finite values")
