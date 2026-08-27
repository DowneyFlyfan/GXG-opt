# Transformer Baseline Profiling Record

## Hardware

- GPU: NVIDIA GeForce RTX 5070 Ti.
- Available GPU memory before profiling: 15.80 GiB.
- Profiling input: CIFAR-10 batches and bfloat16 autocast.

## Rejected candidates

The following candidates were rejected because their parameter counts exceeded
the required 10M–200M interval: 24-layer width-1024 Vision Transformer
(302.44M), 36-layer width-768 Vision Transformer (255.26M), 20-layer
width-1280 Vision Transformer (393.71M), and 32-layer width-1024 Vision
Transformer (403.21M).

## Accepted candidate measurements

| Model | Parameters | Batch | AdamW epoch min | Muon epoch min | AdamW peak MiB | Muon peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| ViT-Base-12x768 | 85,152,010 | 64 | 0.61 | 2.04 | 2611.2 | 2282.4 |
| ViT-Deep-20x768 | 141,854,986 | 48 | 1.06 | 4.24 | 3700.1 | 3154.8 |
| ViT-Large-14x1024 | 176,477,194 | 32 | 1.40 | 7.97 | 3513.2 | 3430.8 |
| Swin-Base | 86,753,474 | 32 | 2.55 | 5.24 | 5833.3 | 5495.4 |

The formal epoch budgets in the accompanying design exceed two hours for every
AdamW run based on these measurements.
