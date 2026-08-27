# DINOv3 ViT-B/16 qualification

## Model and data

- Backbone: `facebook/dinov3-vitb16-pretrain-lvd1689m`, downloaded to
  `.cache/huggingface/models/DINOv3-ViT-B` and loaded offline.
- Task: CIFAR-100 classification with the checkpoint's 224-pixel ImageNet
  normalization and a 100-class linear classifier.
- Model parameters: 85,737,316.  The patch embedding and first eight of twelve
  Transformer blocks are frozen; 28,432,996 parameters remain trainable.
- The frozen patch projection is the first convolution and therefore cannot
  receive Muon.  The classifier is AdamW-only.  Muon applies to eligible
  matrices in the final four trainable Transformer blocks.

## Capacity and speed gate

ABA had 1,547 MiB allocated on one device, so it was not fully available. The
qualification ran on the local Graphics Processing Unit after its preflight.

| Optimizer | Microbatch | Accumulation | Complete update | Optimizer step | Peak memory |
|---|---:|---:|---:|---:|---:|
| AdamW | 8 | 8 | 0.12355 s | 0.00352 s | 1,112.70 MiB |
| Muon | 8 | 8 | 0.13153 s | 0.01095 s | 1,005.73 MiB |

Muon divided by AdamW complete-update time is 1.0646.  A full data epoch took
97.60 seconds for AdamW and 104.33 seconds for Muon.  Seventy-five epochs are
therefore estimated at 2.03 and 2.17 hours respectively, within the two-to-four
hour target and well below the four-hour per-experiment limit.

## Learning-rate tuning

All pilots used the same seed, cached CIFAR-100 order, augmentation, batch
shape, and 75-epoch cosine schedule.  The table reports validation top-1
accuracy after complete epochs.

| Optimizer | Matrix learning rate | Auxiliary AdamW learning rate | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.0001 | not applicable | 0.882812 | 0.896484 | 0.898438 |
| Muon | 0.0003 | 0.0001 | 0.876953 | 0.898438 | 0.890625 |
| Muon | 0.0004 | 0.0001 | 0.867188 | 0.880859 | 0.884766 |
| Muon | 0.0005 | 0.0001 | 0.859375 | 0.873047 | 0.884766 |
| Muon | 0.0007 | 0.0001 | 0.851562 | 0.851562 | 0.859375 |

Full-backbone Muon was unstable for learning rates from 0.00000001 through
0.00002, and cannot be used for a pretrained DINOv3 model.  Freezing the early
feature hierarchy was necessary.  Among the valid partial-fine-tuning trials,
Muon 0.0003 matched and slightly exceeded AdamW at epoch two; it is the chosen
Muon setting for the formal pair.  The formal curve, rather than this short
pilot, remains the deciding result.

## Reliability and outputs

The runner now writes a model, optimizer, scheduler, completed-epoch, and
elapsed-time checkpoint after every metric.  An interrupted run resumes from
that checkpoint instead of resetting its recorded curve.  Finished metric PNGs
include both measured wall-clock durations in their title.
