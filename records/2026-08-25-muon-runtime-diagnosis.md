# Muon Runtime Diagnosis

## Question

Why did the rebuilt baseline estimate substantially longer wall-clock time for
Muon than for AdamW (Adam with decoupled weight decay), despite published Muon
claims of better training efficiency?

## Reproduction

- Hardware: NVIDIA GeForce RTX 5070 Ti with 16,303 MiB memory; 22 MiB was in
  use before the measurement.
- Model: torchvision Vision Transformer with 12 blocks, width 768, and
  85,152,010 trainable parameters.
- Input: batch of 32 CIFAR-10-shaped images with bfloat16 autocast forward and
  backward pass.
- Timing boundary: synchronized optimizer step only; ten post-warm-up samples;
  median reported.

## Measurements

| Optimizer-step implementation | Median seconds | Relative to AdamW |
|---|---:|---:|
| AdamW on matrix parameters | 0.009023 | 1.00 |
| Current Muon, uncompiled float32 | 0.119510 | 13.25 |
| Naive shape-stacked Muon, float32 | 0.191740 | 21.25 |
| Reference-style compiled bfloat16 Muon kernel | 0.053947 | 5.98 |

## Complete-step measurements after GPU access was restored

The table above isolates optimizer work on a 32-pixel Vision Transformer. The
following measurements include forward pass, backward pass, and all optimizer
updates. Each number is the median after warm-up on the same RTX 5070 Ti.

| Workload | Optimizer | Complete step | Optimizer portion | Muon/AdamW ratio |
|---|---|---:|---:|---:|
| 85.15M Vision Transformer, 32x32, batch 32 | AdamW | 28.99 ms | 9.17 ms | 1.00 |
| 85.15M Vision Transformer, 32x32, batch 32 | Current float32 Muon | 139.19 ms | 118.93 ms | 4.80 |
| 85.15M Vision Transformer, 32x32, batch 32 | Compiled bfloat16 Muon | 47.32 ms | 27.24 ms | 1.63 |
| 54.68M decoder Transformer, 1,024 tokens, batch 4 | AdamW | 32.38 ms | 5.95 ms | 1.00 |
| 54.68M decoder Transformer, 1,024 tokens, batch 4 | Compiled bfloat16 Muon | 35.74 ms | 9.29 ms | 1.10 |

The decoder Transformer uses 12 blocks of width 512, eight attention heads,
and a 32,000-token vocabulary. Its peak allocated memory was 3,185 MiB for
AdamW and 3,132 MiB for Muon.

The model contains 52 eligible matrices. The four dominant shapes occur 12
times each: `(768, 768)`, `(768, 3072)`, `(2304, 768)`, and `(3072, 768)`.

## Causal analysis

The current Muon code executes five Newton--Schulz iterations in float32 for
each eligible matrix. Each iteration evaluates a Gram matrix, two matrix
products involving it, and several elementwise operations. AdamW instead uses
elementwise first- and second-moment updates, so its optimizer step is much
cheaper.

The local Moonlight reference has two execution differences from the current
baseline: it decorates the Newton--Schulz kernel with `torch.compile` and casts
the iteration matrix to bfloat16. Reproducing those two properties reduces the
orthogonalization pass from 117.6 ms to 53.9 ms. This directly establishes that
the original 2--4 hour estimates were based on an inefficient implementation.

The attempted three-dimensional batching of equal-shaped matrices was slower,
not faster, on this GPU. Therefore that change is rejected rather than adopted.

## Interpretation

Muon being more computationally or sample efficient does not imply that one
optimizer step is faster. Muon may reach a target validation metric in fewer
updates or with less total training computation while paying a more expensive
orthogonalization cost per update. A wall-clock superiority claim requires both
an efficient implementation and an end-to-end metric-versus-time measurement.

## Conclusion and next action

The current uncompiled float32 Muon timing is not an acceptable formal
baseline. The formal benchmark is revised from low-compute 32-pixel Vision
Transformers to long-sequence decoder-only language Transformers. The
compiled bfloat16 path closes the end-to-end gap from 4.80 times to 1.10 times
on the representative long-sequence prototype. The next implementation step is
to place that reference-conformant path in `src/optimizers.py`, test it against
the reference calculation, and profile the four formal candidates before any
epoch budget is assigned.
