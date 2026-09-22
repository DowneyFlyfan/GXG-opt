# Qwen3-0.6B AWS A100 500M-token baseline plan

## Scope

- Start each baseline from a fresh Qwen3-0.6B initialization.
- Use 500,039,680 training-token exposures per baseline: five epochs of
  100,007,936 committed tokens.
- Tune AdamW, Muon, and Muown with short matched 16,056,320-token screens,
  then run the lowest-screen-perplexity configuration for each optimizer.

## Configured launch

- Region: `us-east-1`.
- Instance request: one `p4de.24xlarge`, eight NVIDIA A100 80 GiB GPUs,
  on-demand.
- Console price shown at configuration time: $27.44705 per hour for Linux.
- Instance tag: `owner=yufanwang`, applied to the Instance resource type.
- Root volume: 128 GiB gp3 SSD.
- SSH: the existing `downeyflyfan-g7e-20260914` key; ingress restricted to the
  launching machine's public address.

## Training geometry

- Eight data-parallel ranks.
- Per-rank micro-batch: 8 sequences of 1,024 tokens.
- Gradient accumulation: 1.
- Global effective batch: 64 sequences, equal across all baselines.
- Formal updates: 1,526 updates per epoch and 7,630 updates over five epochs.
- Validation: 12 held-out batches at initialization and after each epoch.

## Budget and timing estimate

The prior single-A100 runs processed approximately 13,528 tokens per second.
With eight A100 ranks, a full 500M-token baseline has an ideal compute time of
about 1.3 hours. Allowing for communication, validation, the seven tuning
screens, setup, and recovery, the planned job is 5 to 8 instance-hours for all
three formal baselines. At $27.44705 per hour, that is approximately $137 to
$220. The first 100 updates will replace this estimate with measured
throughput before the formal runs continue.
