# DINOv3 ViT-B/16 formal AdamW and Muon baseline

## Scope

- Task: CIFAR-100 top-1 accuracy using the cached offline
  `facebook/dinov3-vitb16-pretrain-lvd1689m` checkpoint.
- Model: 85,737,316 parameters. The patch embedding and first eight of the
  twelve Transformer blocks were frozen. The classifier uses AdamW.
- Both formal runs used microbatch 8, gradient accumulation 8, 75 epochs,
  cached data, and the same seed.
- Muon applies only to eligible matrix parameters in the last four Transformer
  blocks. The first convolution, embeddings, normalizations, and classifier
  are excluded according to `AGENTS.md`.

## Tuning decision

The initially qualified Muon matrix learning rate of 0.0003 was rejected
before completion: after 50 epochs its validation accuracy was 0.888672,
against AdamW's 0.914062 at the same epoch. Its partial metric trace and
checkpoint are retained with the `muon_lr3e-4_rejected` suffix.

An isolated 10-epoch Muon pilot at matrix learning rate 0.0001 and auxiliary
AdamW learning rate 0.0001 achieved 0.931641 at epoch 10. AdamW achieved
0.898438 at epoch 10. Therefore the final Muon run uses 0.0001 for both.

## Formal result

| Optimizer | Matrix LR | Final accuracy | Best accuracy | Wall-clock | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.0001 | 0.916016 | 0.923828 | 2h 03m 51s | 1,113.29 MiB |
| Muon | 0.0001 | 0.916016 | 0.929688 | 2h 16m 11s | 1,004.10 MiB |

Muon divided by AdamW wall-clock time is 1.0996, within the required 1.20
limit. Muon ties the final fixed-subset accuracy and reaches a higher peak
accuracy. The paired epoch-metric figure is
`results/cv/cv_dinov3_vitb16_metric_steps.png`; its title states both measured
durations.
