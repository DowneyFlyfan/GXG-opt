from __future__ import annotations

from typing import Any

from torch import Tensor


def detached_cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detached_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detached_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [detached_cpu_tree(item) for item in value]
    return value
