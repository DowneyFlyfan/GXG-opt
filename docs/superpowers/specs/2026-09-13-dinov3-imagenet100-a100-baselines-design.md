# DINOv3 ImageNet-100 A100 Baselines

## Purpose

Replace the current DINOv3 ViT-B CIFAR-100 comparison with a stable, image-domain-matched
baseline suite that can complete five matched epochs within four hours on one A100 80 GB
accelerator. The suite compares AdamW, Muon, and Muown without reusing the old CIFAR-100
artifacts.

## Experiment contract

- Model: the existing DINOv3 ViT-B classifier (85,737,316 parameters), with its 100-way
  classifier head and the current frozen embeddings plus first eight transformer blocks.
- DINOv3 is an encoder-only Vision Transformer, not a variational autoencoder (VAE). It has a
  16-by-16 patch projection, twelve transformer blocks, feature normalization, and a downstream
  classifier; it has no encoder-decoder latent-variable path.
- Dataset: `clane9/imagenet-100`, which has 126,689 training images and 5,000 validation
  images in two ImageNet-format splits. Its fixed `ClassLabel` mapping has exactly 100 labels,
  so the classifier head remains compatible.
- Data cache: Hugging Face data, decoded dataset metadata, and any staged image cache are under
  `.cache/cv/imagenet100`; checkpoints remain under `.cache/cv/checkpoints`. No checkpoints go
  in `nlp/`.
- Augmentation: ImageNet normalization, random-resized-crop plus horizontal flip for training,
  and resize-plus-center-crop for validation at 224 by 224 pixels.
- Comparison: one fixed seed, full 5,000-image validation at each epoch end, five epochs per
  final trial, no weight decay unless a screen supplies evidence that it is needed.
- Budget: each final trial must finish in at most four hours. A GPU-memory profile selects the
  largest safe microbatch before final launch; gradient accumulation then fixes the matched
  effective batch.

## Optimizer contract

Each optimizer is tuned independently in one-epoch screens before its five-epoch final run.
The screens start aggressively and use the same data order, model initialization, effective
batch, epoch count, and validation definition as the final trials.

- AdamW has its own learning-rate candidates.
- Muon has an independent matrix learning-rate sweep and an independent AdamW auxiliary
  learning-rate sweep for excluded parameters such as the classifier.
- Muown exposes two independent rates: `direction_lr` for its tangent Muon update and `gain_lr`
  for the Adam row-gain update. Its auxiliary AdamW rate is independently configured as well.
  A single shared rate is prohibited.
- No matrix-based optimizer is used on a first or last neural-network layer. The frozen patch
  projection remains untouched; all trainable matrices in the final transformer block, the final
  feature normalization, and the classifier use AdamW. Muon and Muown operate only on eligible
  interior matrices of trainable blocks 8 through 10. Existing one-dimensional parameters also
  remain in AdamW.

The selected settings are those that remain numerically finite and maximize full-validation
top-1 accuracy after their matched screen. Every screen is retained as metrics and a record, but
only selected final runs receive comparison figures.

## Measurement and artifacts

Every metric record contains `epoch`, completed optimizer `step`, measured
`elapsed_seconds`, and full-validation `metric`. Figures use those stored values directly:

- `results/cv/cv_dinov3_vitb16_imagenet100_baselines_metric_steps.png`
- `results/cv/cv_dinov3_vitb16_imagenet100_baselines_metric_time.png`

The step figure has completed optimizer step on its horizontal axis; the time figure has actual
wall-clock timestamps, never epoch interpolation. Per-optimizer JSONL metric streams and JSON
results record learning rates, batch settings, runtime, peak memory, status, and accelerator.
The old CIFAR-100 files remain historical evidence and are not overwritten.

## Execution topology

Local code is tested before synchronization. Two idle ABA A100 80 GB accelerators run AdamW and
Muon in parallel. The user-owned Nautilus deployment has been reduced from an unschedulable
four-A100/64-GiB-scratch request to one A100/20-GiB-scratch request for Muown. If it schedules,
Muown starts there concurrently. If it remains unavailable, Muown starts on the first released
ABA A100 and is labeled as a delayed, otherwise matched run; no unrelated training is stopped.

Before a final launch, the updated code and `.cache` directory layout are synchronized to the
target. Large data are downloaded on the target into its `.cache`, not copied into source control.
Each final run is invoked with `nohup`, has a unique run label and checkpoint, and is monitored to
completion. Results, records, and figures are then committed and pushed without staging unrelated
worktree changes.

## Verification

Tests cover the ImageNet-100 cached loader and its 100 labels, full-validation behavior,
step/timestamp metric serialization, three-way plotting, and separate Muown direction/gain
rates. A short A100 profile verifies model loading, data decoding, safe batch size, and absence
of out-of-memory failure before any full run. Final acceptance requires all three completed JSON
results, their full-validation traces, both PNG figures, and a record containing the selected
settings and runtimes.
