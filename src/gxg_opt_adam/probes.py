from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _common_names(*maps: Mapping[str, torch.Tensor]) -> list[str]:
    if not maps:
        return []
    return sorted(set.intersection(*(set(value) for value in maps)))


def dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> torch.Tensor:
    names = _common_names(left, right)
    if not names:
        return torch.tensor(0.0)
    device = left[names[0]].device
    return sum((left[name].float() * right[name].float()).sum() for name in names).to(device)


def norm(value: Mapping[str, torch.Tensor]) -> float:
    squared = dot(value, value)
    return float(torch.sqrt(torch.clamp(squared, min=0)).item())


def cosine(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], eps: float = 1.0e-12) -> float:
    denominator = norm(left) * norm(right) + eps
    return float(dot(left, right).item() / denominator)


def negate(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: -tensor for name, tensor in value.items()}


def gradient_noise_ratio(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor], eps: float = 1.0e-12
) -> float:
    names = _common_names(first, second)
    difference = {name: first[name] - second[name] for name in names}
    mean = {name: 0.5 * (first[name] + second[name]) for name in names}
    return float(dot(difference, difference).item() / (2.0 * dot(mean, mean).item() + eps))


def mean_squared_gradients(gradients: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not gradients:
        return {}
    names = _common_names(*gradients)
    return {
        name: sum(gradient[name].float().square() for gradient in gradients) / len(gradients)
        for name in names
    }


def reduction_ratio(actual_reduction: float, predicted_reduction: float, eps: float = 1.0e-12) -> float:
    return actual_reduction / (predicted_reduction + eps)


def transfer_ratio(reference_reduction: float, curvature_reduction: float, eps: float = 1.0e-12) -> float:
    return reference_reduction / (curvature_reduction + eps)


def gain_per_second(reduction: float, seconds: float, eps: float = 1.0e-12) -> float:
    return reduction / (seconds + eps)
