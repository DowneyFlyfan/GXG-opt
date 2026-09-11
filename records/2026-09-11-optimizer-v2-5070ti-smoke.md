# Optimizer-v2 RTX 5070 Ti smoke

## Scope

The six optimizer-v2 ideas are now sourced from `records/ideas/` and implemented
by `src/optimizer_v2/`. This local test executes the two baselines plus all six
proposals in one single-seed, fixed-step smoke comparison before any full
five-epoch claim.

## Local profile

`configs/experiments/gpt2_v2_5070ti_smoke.yaml` preserves random GPT-2 (124M),
WikiText-103 blocks of length 512, BF16, five configured epochs, seed 0, and the
eight-method matrix. To fit the 16 GiB RTX 5070 Ti conservatively, it uses
physical batch 4 and accumulation 8, preserving the source profile's effective
batch 32. The smoke cap is supplied by `--steps 33`; it does not represent a
completed epoch or a final optimizer comparison.

The profile's manifest reports `NVIDIA GeForce RTX 5070 Ti` and a 16 GiB minimum
instead of the imported 5090/32 GiB profile. The configured output keeps metrics,
diagnostics, manifests, and plots under `results/gpt2_v2_5070ti_smoke`; all
resumable checkpoints are redirected beneath `.cache/gpt2-v2/checkpoints/`.

## Runtime boundary

The repository's local environment provides CUDA Torch **2.13.0+cu130**,
Transformers **4.57.6**, Datasets **5.0.1**, and Matplotlib **3.10.8**. The
imported source validation record pins Torch 2.11.0+cu130 and Transformers 5.7.0;
the exact `2.11.0+cu130` package cannot be resolved from this host's package
index. This smoke is therefore a Torch-2.13 compatibility result, not a
reproduction of the source runtime's PyTorch-2.11 parity claim.

The 2.13 Muon constructor plus the `_adjust_lr` and
`_zeropower_via_newtonschulz` helper signatures match the adapter call sites.
The adapter now explicitly accepts only 2.11.0 and 2.13.0, records the exact
Torch revision and Muon source SHA-256 in its manifest, and rejects every other
version. Focused profile/math tests passed **28 tests**; the existing
end-to-end integration suite passed **31 tests** (one pre-existing skip) under
Torch 2.13, including its tiny-GPU run of all eight methods.

The first launch exposed one stale execution-only RTX 5090 guard after
configuration validation had already accepted the 5070 Ti profile. The guard
now compares the actual device to `hardware.gpu`, and a regression test covers
the configured-5070-Ti runtime path. The failed launch performed no training,
produced no metrics, and retained no checkpoint.

## Execution gate

Before launch, prepare the random GPT-2 and WikiText-103 assets only under
`.cache/gpt2-v2`. Then run exactly one 33-step job matrix. Inspect each method's
summary, metric logs, checkpoint location, and generated step/time PNGs before
reporting any result.

## Completed Torch-2.13 compatibility smoke

The bounded screen completed all eight methods, with the identical random
initialization hash `811cbf5668684ccd17f492caec21cc986a42b984f4f836314fb8bd20ba5cdafe`.
Each method committed 33 updates (540,672 input tokens) and evaluated the same
248,346 validation prediction tokens at updates 0, 16, 32, and 33. The screen
did not complete an epoch; `epochs_completed = 0` is expected because the
prepared training split has 232,000 blocks.

| Method | Validation NLL | Wall s | Peak GiB | Fallbacks |
|---|---:|---:|---:|---:|
| AdamW | 9.641650 | 35.61 | 5.826 | 0 |
| Muon | 10.842623 | 52.04 | 5.567 | 0 |
| Query-key correction | 10.842623 | 34.45 | 5.530 | 0 |
| Sparse attention curvature | 10.842601 | 34.20 | 5.530 | 0 |
| Tied embedding curvature | 10.842613 | 36.52 | 5.818 | 0 |
| Resonance filtering | 10.842623 | 35.36 | 5.742 | 0 |
| Feature-remap cohort | 10.842557 | 53.28 | 5.807 | 0 |
| LayerNorm response | 10.842497 | 34.78 | 5.530 | 0 |

The complete evidence is in `results/gpt2_v2_5070ti_smoke_t213/`, including
the step and time figures, immutable source snapshot, manifests, metric logs,
diagnostics, and `final_results.csv`. All eight resumable checkpoints were
verified under `.cache/gpt2-v2/checkpoints/results/gpt2_v2_5070ti_smoke_t213/`;
none were placed under `results/nlp`. The figures were visually inspected.

This is an execution and compatibility result, not a five-epoch result and not
evidence of optimizer superiority. AdamW is substantially lower at this
un-tuned fixed setting, and the largest proposal-versus-Muon difference is only
about `1.25e-4` NLL. A future final comparison must tune the baselines and each
candidate with comparable budgets, use the required five epochs, and use
matched repeated seeds.

## Aggressive Muon learning-rate screen

The original `3.0e-4` Muon rate produced corrections whose observed impact was
near the validation rounding floor. With the same 33 updates, random
initialization, batch protocol, and validation set, a direct Muon screen found:

| Muon LR | Validation NLL | Peak GiB | Fallbacks |
|---:|---:|---:|---:|
| 3e-4 | 10.842623 | 5.567 | 0 |
| 3e-3 | 9.752750 | 5.567 | 0 |
| 1e-2 | 8.072368 | 5.567 | 0 |

The `1e-2` screen was stable, so each proposal was run once at that matched
rate. All reached 33 updates with no fallback. Feature-remap had the largest
screen improvement over Muon, but only `9.16e-4` NLL and with 52.76 seconds
versus Muon's 51.71 seconds; it is a continuation candidate, not a winning
result.

| Proposal at Muon LR 1e-2 | Validation NLL | Wall s | Delta vs Muon |
|---|---:|---:|---:|
| Query-key correction | 8.072368 | 34.64 | 0.000000 |
| Sparse attention curvature | 8.072426 | 34.13 | +0.000058 |
| Tied embedding curvature | 8.072309 | 37.29 | -0.000059 |
| Resonance filtering | 8.072368 | 35.28 | 0.000000 |
| Feature-remap cohort | 8.071451 | 52.76 | -0.000916 |
| LayerNorm response | 8.071860 | 34.51 | -0.000508 |

The tuning artifacts are `results/gpt2_v2_5070ti_muon_lr003_screen/`,
`results/gpt2_v2_5070ti_muon_lr01_screen/`, and
`results/gpt2_v2_5070ti_proposals_lr01_screen/`; their checkpoints remain only
under the matching `.cache/gpt2-v2/checkpoints/results/` paths.
