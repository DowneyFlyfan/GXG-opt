from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def rng_state_dict() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch_cpu": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:  # pragma: no cover
        pass
    return state


def load_rng_state_dict(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:  # pragma: no cover
            pass


def checkpoint_payload(model: nn.Module, optimizer, *, scheduler=None, scaler=None, sampler=None, extra=None):
    return {
        "model": model.state_dict(),
        "gn_guided_adamw": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "sampler": sampler.state_dict() if sampler is not None else None,
        "rng": rng_state_dict(),
        "extra": dict(extra or {}),
    }


def save_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def restore_payload(payload, model: nn.Module, optimizer, *, scheduler=None, scaler=None, sampler=None):
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["gn_guided_adamw"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if sampler is not None and payload.get("sampler") is not None:
        sampler.load_state_dict(payload["sampler"])
    load_rng_state_dict(payload["rng"])
    return dict(payload.get("extra", {}))
