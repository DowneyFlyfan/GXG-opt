#!/usr/bin/env python3
"""Bounded microbenchmark of factor spectral and correction inverse actions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator
from low_rank_corrected_kronecker_ggn.correction import corrected_direction


def timed(function, repeats: int, synchronize) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    synchronize()
    return (time.perf_counter() - started) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--memory-budget-mb", type=float, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "kronecker_ggn" / "overhead.json",
    )
    args = parser.parse_args()
    if args.rows <= 0 or args.columns <= 0 or args.rank < 0 or args.repeats <= 0:
        raise ValueError(
            "matrix dimensions/repeats must be positive and rank nonnegative"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    estimated = (2 * args.rank + 3) * args.rows * args.columns * 4
    if estimated > args.memory_budget_mb * 1024**2:
        raise MemoryError(
            f"Estimated {estimated / 1024**2:.1f} MiB exceeds the explicit budget"
        )
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(42)
    activation = torch.eye(args.columns, device=device)
    output = torch.eye(args.rows, device=device)
    operator = KroneckerSpectralOperator(activation, output, 0.01)
    gradient = torch.randn(args.rows, args.columns, generator=generator, device=device)
    flat = torch.randn(
        args.rows * args.columns, args.rank, generator=generator, device=device
    )
    basis = torch.linalg.qr(flat, mode="reduced")[0].T.reshape(
        args.rank, args.rows, args.columns
    )
    eigenvalues = torch.linspace(-0.5, 1.0, args.rank, device=device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    synchronize()
    baseline_seconds = timed(
        lambda: operator.apply_inverse(gradient), args.repeats, synchronize
    )
    corrected_seconds = timed(
        lambda: corrected_direction(
            operator,
            gradient,
            basis,
            eigenvalues,
            eigenvalue_margin=0.1,
            absolute_eigenvalue_cap=100.0,
        ),
        args.repeats,
        synchronize,
    )
    report = {
        "rows": args.rows,
        "columns": args.columns,
        "rank": args.rank,
        "device": str(device),
        "dtype": "float32",
        "repeats": args.repeats,
        "baseline_inverse_seconds": baseline_seconds,
        "corrected_inverse_seconds": corrected_seconds,
        "corrected_over_baseline_ratio": corrected_seconds / baseline_seconds,
        "estimated_working_memory_mb": estimated / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0,
        "accelerator_hours": (baseline_seconds + corrected_seconds)
        * args.repeats
        / 3600
        if device.type == "cuda"
        else 0.0,
        "scope": "optimizer_inverse_action_microbenchmark_not_training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
