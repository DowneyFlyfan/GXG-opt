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
imported source validation record pins Torch 2.11.0+cu130 and Transformers 5.7.0,
which are unavailable on this host. The smoke is therefore a local execution
compatibility result, not a reproduction of the source runtime's parity claim.
The independent FP64 optimizer-v2 mathematics suite and the new profile/cache
tests passed **26 tests** under the local runtime; the PyTorch-2.11-only
integration suite remains skipped by its existing version guard.

## Execution gate

Before launch, prepare the random GPT-2 and WikiText-103 assets only under
`.cache/gpt2-v2`. Then run exactly one 33-step job matrix. Inspect each method's
summary, metric logs, checkpoint location, and generated step/time PNGs before
reporting any result.
