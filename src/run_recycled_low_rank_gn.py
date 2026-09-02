from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from recycled_low_rank_gn_experiment import run_kron_rpcg_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the recycled low-rank PCG Gauss--Newton experiment"
    )
    parser.add_argument("--physical-batch-size", type=int, default=4)
    parser.add_argument("--outer-effective-batch-size", type=int, default=3_904)
    parser.add_argument("--curvature-batch-size", type=int, default=3_904)
    parser.add_argument("--damping", type=float, default=0.03)
    parser.add_argument("--correction-rank", type=int, default=4)
    parser.add_argument("--correction-refresh-interval", type=int, default=4)
    parser.add_argument("--minimum-relative-eigenvalue", type=float, default=0.25)
    parser.add_argument("--maximum-pcg-iterations", type=int, default=6)
    parser.add_argument("--relative-pcg-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--solver-mode", choices=("secant_direct", "pcg"), default="secant_direct"
    )
    parser.add_argument("--line-search-range", type=int, default=5)
    parser.add_argument("--initial-step-scale", type=float, default=1.0)
    parser.add_argument("--line-search-screening-sequences", type=int)
    parser.add_argument("--line-search-finalists", type=int, default=2)
    parser.add_argument(
        "--preconditioner-statistic",
        choices=("mean_gradient_square", "batch_second_moment"),
        default="mean_gradient_square",
    )
    parser.add_argument("--maximum-outer-steps", type=int, default=10_000)
    parser.add_argument("--maximum-seconds", type=float, default=13_189.759907)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--paper-full-gn-label",
        default="paper_full_gn_template_n122_b32_lr0003",
    )
    parser.add_argument("--basis-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    result = run_kron_rpcg_trial(
        Path(__file__).resolve().parents[1],
        physical_batch_size=args.physical_batch_size,
        outer_effective_batch_size=args.outer_effective_batch_size,
        curvature_batch_size=args.curvature_batch_size,
        damping=args.damping,
        correction_rank=args.correction_rank,
        correction_refresh_interval=args.correction_refresh_interval,
        minimum_relative_eigenvalue=args.minimum_relative_eigenvalue,
        maximum_pcg_iterations=args.maximum_pcg_iterations,
        relative_pcg_tolerance=args.relative_pcg_tolerance,
        solver_mode=args.solver_mode,
        line_search_range=args.line_search_range,
        initial_step_scale=args.initial_step_scale,
        line_search_screening_sequences=args.line_search_screening_sequences,
        line_search_finalists=args.line_search_finalists,
        preconditioner_statistic=args.preconditioner_statistic,
        maximum_outer_steps=args.maximum_outer_steps,
        maximum_seconds=args.maximum_seconds,
        workers=args.workers,
        seed=args.seed,
        label=args.label,
        paper_full_gn_label=args.paper_full_gn_label,
        basis_dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[
            args.basis_dtype
        ],
        fresh=args.fresh,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
