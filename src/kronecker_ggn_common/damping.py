from __future__ import annotations

import torch
from torch import Tensor


def joint_damped_eigenvalues(
    activation_eigenvalues: Tensor,
    output_eigenvalues: Tensor,
    damping: float,
    floor: float,
) -> Tensor:
    if damping <= 0 or floor <= 0:
        raise ValueError("damping and floor must be positive")
    return torch.clamp(
        output_eigenvalues[:, None] * activation_eigenvalues[None, :] + damping,
        min=floor,
    )
