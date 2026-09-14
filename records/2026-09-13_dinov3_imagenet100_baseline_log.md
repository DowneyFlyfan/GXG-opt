# DINOv3 ImageNet-100 baseline log

## Contract

- Model: DINOv3 Vision Transformer Base/16 classifier, 85,737,316 parameters.
- Task: ImageNet-100 classification with 100 classes.
- Data: 126,689 training images and 5,000 validation images.
- Metric: full-validation top-1 accuracy.
- Optimizers: AdamW, Muon, and Muown. Matrix optimization is restricted to trainable interior transformer-block matrices; patch embedding, final transformer block, normalization, classifier, and all non-matrix parameters use AdamW.
- Weight decay: zero because the model is below the 200 million parameter threshold.
- Checkpoints and dataset caches: project `.cache` only.

## Data staging

- Completed locally on 2026-09-13. The Hugging Face cache and dataset conversion cache are both under `.cache`.
- A100 execution was unavailable for this dataset because ABA had 369 MB free on its root filesystem and 44 MB free on `/data`; the 8.41 GB dataset cannot fit there. The fallback machine has 338 GB free and an idle 16 GB NVIDIA GeForce RTX 5070 Ti.

## Active screen

| Field | Value |
|---|---:|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Micro-batch size | 64 |
| Gradient accumulation | 4 |
| Effective batch size | 256 |
| Epochs | 1 |
| Purpose | Establish a safe throughput and accuracy screen before final five-epoch tuning |

| Result | Value |
|---|---:|
| Full-validation top-1 accuracy | 94.80% |
| Optimizer updates | 495 |
| Wall-clock time | 271.95 s |
| Peak allocated GPU memory | 3,323.50 MiB |

The local screen completed successfully. It establishes a working AdamW reference but does not yet establish the final five-epoch tuned baseline.

## ABA transition

- The processed cache was synchronized and validated offline on ABA: 126,689 training and 5,000 validation examples.
- ABA uses Transformers 5.16.1, whose DINOv3 encoder exposes `backbone.model.layer` instead of the local Transformers 4.57.6 layout `backbone.layer`. The training adapter and matrix-parameter routing now support both layouts, with a regression test. This preserves the boundary-layer AdamW policy on both hosts.

## ABA batch-size admission

| Probe | Micro-batch | Optimizer | Peak allocated memory | Outcome |
|---|---:|---|---:|---|
| One update | 1,024 | AdamW | 41,501.3 MiB | Fits |
| One update | 1,280 | Muon | 51,753.2 MiB | Fits |
| One update | 1,536 | Muon | 62,023.2 MiB | Fits |
| One epoch | 1,792 | Muon | 77,468 MiB total in use | Out of memory; stopped before any metric/checkpoint |

The active Muon screen therefore uses micro-batch 1,536 and no accumulation. It has enough headroom for the A100 80 GB device while using substantially more memory than the 1,280 fit point. Its configuration is learning rate 1e-3, auxiliary AdamW learning rate 1e-4, zero weight decay, and one epoch.

## Completed screens

| Optimizer | Host | Micro-batch / accumulation | Updates | Top-1 accuracy | Time | Peak memory |
|---|---|---:|---:|---:|---:|---:|
| AdamW | local RTX 5070 Ti | 64 / 4 | 495 | 94.80% | 271.95 s | 3,323.50 MiB |
| Muon | ABA A100 80 GB | 1,536 / 1 | 83 | 94.14% | 206.44 s | 62,191.13 MiB |

These are one-epoch parameter screens, not the final five-epoch comparison. They intentionally have optimizer-specific batch settings, so their accuracy values are screening evidence rather than a matched winner declaration.

## Active screen

Muown is running on ABA A100 GPU 1 with direction learning rate 1e-3, gain learning rate 1e-4, micro-batch 1,536, no accumulation, and zero weight decay. Its one-step memory admission matched the 62,023.2 MiB Muon probe.

The first Muown invocation stopped before its first batch because the two-rate optimizer did not expose a scheduler-compatible `lr` field. The fix maps that field to the scheduled direction rate while retaining the independent gain rate; it is covered by a regression test. No metric or checkpoint was produced by the failed invocation.
