# Kronecker GGN implementation qualification

Date: 2026-08-27

## Scope

This record qualifies the repository integration and dense mathematical reference implementation. It is not a compute-normalized training claim. No 10M–200M formal model experiment was started because `ssh ABA` could not resolve from this environment, the local test environments do not contain the full formal dataset stack, and the local OWSM checkpoint is absent. The only GPU work was the tiny FP32 spectral smoke test on the single local GPU; no OOM retry occurred.

## Correctness

- New optimizer suite: 23 passed, comprising CPU FP64 mathematical checks, a tiny per-token cross-entropy Transformer check, and a CUDA FP32 smoke test.
- Existing accessible suite plus the new optimizer tests: 109 passed when tests requiring the unavailable `datasets` package and local OWSM checkpoint were excluded.
- Full-suite collection is currently blocked by `ModuleNotFoundError: datasets`; the OWSM adapter test is separately blocked by the missing `.cache/huggingface/models/owsm_v3.1_ebf_base` checkpoint.
- Ruff check passes for all new implementation, script, registry, and optimizer test files.
- Baseline inverse, inverse square root, and square root actions agree with dense FP64 eigendecompositions at `1e-11` tolerance.
- Matrix-free exact GGN products agree with the double-autograd reference and explicit dense matrix at `1e-10` tolerance.
- Rank-zero corrected optimization matches the baseline trajectory exactly in the deterministic test.
- Exact rank-one relative mismatch recovers the dense damped GN direction within `1e-9`.
- Checkpoint round-trip produces bit-identical next baseline and corrected directions, including AdamW fallback and low-rank correction state.

## Residual-spectrum reference diagnostic

Artifact: `results/kronecker_ggn/residual_spectrum.json` and `results/kronecker_ggn/residual_spectrum.png`.

The deterministic synthetic exact-GGN fixture has signed mismatch eigenvalues `1.40, -0.65, 0.35, -0.18, 0.08, -0.03`. The baseline direction cosine to the exact damped-GN direction is `0.98025`. Corrected cosine by rank is:

| Rank | Oracle cosine | Relative direction error |
|---:|---:|---:|
| 0 | 0.98025 | 0.19825 |
| 1 | 0.98897 | 0.15070 |
| 2 | 0.99870 | 0.05181 |
| 4 | 0.99998 | 0.00609 |
| 8 | 1.00000 | approximately zero |

This confirms the implementation on a deliberately low-rank synthetic system. It does not satisfy the scientific go gate for real checkpoints; that gate still requires early/middle/late baseline checkpoints and independent curvature batches.

## Bounded overhead reference

Artifact: `results/kronecker_ggn/overhead.json`.

On CPU for a `64 x 64` matrix-shaped block, rank 4, and five repetitions:

- Baseline inverse action: `0.0617 ms` average.
- Corrected inverse action: `0.2328 ms` average.
- Corrected/baseline action ratio: `3.78x`.
- Estimated working memory: `0.172 MiB`.
- Accelerator-hours and accelerator peak memory: zero because this was a CPU microbenchmark.

These timings cover inverse actions only. They do not include model forward/backward, factor capture, an exact GGN product, correction construction, evaluation, or data loading, so they must not be presented as training-time savings.

## Formal experiment gate

Before any optimizer-quality claim, run separately tuned AdamW and Muon baselines plus both Kronecker variants on the same 10M–200M model, seed, batches, precision, and single GPU. Use the checked-in comparison script to produce the required combined metric-vs-step PNG and report total wall-clock, accelerator-hours, peak memory, correction-build time, and held-out quality. Stop or change direction if real residual spectra are flat, batch-specific, or fail to improve realized held-out decrease at ranks up to 8.
