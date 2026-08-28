from __future__ import annotations

import torch
from torch import Tensor


def mse_output_hessian_matvec(
    vector: Tensor, *, reduction: str = "mean", output_numel: int | None = None
) -> Tensor:
    if reduction == "sum":
        return vector
    if reduction != "mean":
        raise ValueError("MSE reduction must be mean or sum")
    denominator = output_numel if output_numel is not None else vector.numel()
    return vector / max(denominator, 1)


def softmax_cross_entropy_hessian_matvec(
    logits: Tensor,
    vector: Tensor,
    *,
    mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    if logits.shape != vector.shape:
        raise ValueError("logits and vector must have identical shapes")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError("reduction must be none, sum, or mean")
    compute_dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    probability = logits.to(compute_dtype).softmax(dim=-1)
    tangent = vector.to(compute_dtype)
    result = probability * tangent - probability * (probability * tangent).sum(
        dim=-1, keepdim=True
    )
    if mask is not None:
        if mask.shape != logits.shape[:-1]:
            raise ValueError("mask must match every logits dimension except classes")
        result = result * mask.to(result).unsqueeze(-1)
    if reduction == "mean":
        count = (
            int(mask.sum().item())
            if mask is not None
            else logits.numel() // logits.shape[-1]
        )
        result = result / max(count, 1)
    return result.to(vector)
