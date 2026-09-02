# Stiefel-Muon on GPT-12x512

## Question

Evaluate the square closed-form update in [article 11215](https://kexue.fm/archives/11215)
and the rectangular closed-form update in
[article 11864](https://kexue.fm/archives/11864) on the retained WikiText-103
GPT-12x512 baseline, then compare it with retained AdamW and ordinary Muon.

## Method audited before training

The premise that only the first and final layers are rectangular is false for
this Transformer.  The baseline's Muon routing selects 48 matrices:

| Logical matrix geometry | Count | Formula used |
|---|---:|---|
| 512 by 512 projection | 12 | square orthogonal update from article 11215 |
| 1536 by 512 QKV projection | 12 | column-Stiefel update from article 11864 |
| 2048 by 512 feed-forward expansion | 12 | column-Stiefel update from article 11864 |
| 512 by 2048 feed-forward contraction | 12 | transpose, apply the column-Stiefel update, transpose back |

Embedding/tied output, position embedding, biases, and normalization parameters
remain in the AdamW auxiliary optimizer, exactly as in the ordinary Muon
baseline.

For an initial selected matrix \(W_0\), the implementation preserves its RMS
scale \(s=\lVert W_0\rVert_F/\sqrt{d}\) and constrains
\(Q=W/s\), not the physical tensor directly.  Thus \(Q^\top Q=I\) for tall
matrices (or \(QQ^\top=I\) for stored-wide matrices), while the model retains
the initialization's overall scale.  This avoids the severe logit-scale change
caused by forcing every physical weight to unit Stiefel scale.

The rectangular fast path was tested directly against the full article
generator \(\operatorname{msign}(GW^\top-WG^\top)W\); it is its QR-reduced,
mathematically equivalent form.

## Tuning and resource selection

The local NVIDIA GeForce RTX 5070 Ti has 16,303 MiB.  Physical batch 14 was
unsafe once gradient accumulation and optimizer workspaces were enabled;
physical batch 12 with accumulation 4 was stable and used 12,295 MiB peak
allocated memory.  It gives effective batch 48.

| Learning rate | Probe steps | Validation accuracy |
|---:|---:|---:|
| 0.0003 | 12 | 0.189693 |
| 0.001 | 12 | 0.229403 |
| 0.003 | 128 | 0.292706 |
| 0.01 | 12 | 0.273836 |

The formal configuration was therefore learning rate 0.003, momentum 0.95,
five Newton-Schulz iterations, physical batch 12, accumulation 4, deterministic
seed 1337, and the same five WikiText-103 epochs/validation metric as the
retained baseline.  It completed 10,170 optimizer steps in 13,768.6 seconds,
below the four-hour cap.

## Final result

| Optimizer | Final validation next-token accuracy | Time | Peak allocated GPU memory |
|---|---:|---:|---:|
| AdamW (retained) | 0.711754 | 2.787 h | 4,824 MiB |
| Muon (retained) | **0.752193** | **2.894 h** | 4,688 MiB |
| Stiefel-Muon | 0.722632 | 3.825 h | 12,295 MiB |

Stiefel-Muon improves on the retained AdamW result by 0.010878 absolute
accuracy, but it is 0.029561 below ordinary Muon and takes 32.2% longer.  It
therefore does **not** beat ordinary Muon under this configuration.

The final checkpoint contains all 48 constrained matrices.  The normalized
Stiefel Gram-error range is \(4.09\times10^{-5}\) to
\(2.43\times10^{-3}\) in Frobenius norm; this is finite-precision
Newton-Schulz retraction error, not unconstrained drift.

## Interpretation and comparison limit

This is a genuine hard-manifold optimizer evaluation, not evidence that hard
orthogonality is generally beneficial.  Hard Stiefel initialization changes the
selected matrices' singular spectra, while the historical AdamW/Muon traces
start from the ordinary Gaussian GPT initialization.  Consequently, their
reported comparison is useful for the requested retained baseline, but it is
not a fully initialization-matched causal ablation.  A strict follow-up should
run ordinary Muon from the identical scaled-Stiefel initial state and same
effective batch before attributing the gap solely to the optimizer update.

## Artifacts

- Metrics: `metrics/nlp/nlp_gpt_12x512__stiefel_muon_b12_a4_lr003_formal.jsonl`
- Result: `results/nlp/nlp_gpt_12x512__stiefel_muon_b12_a4_lr003_formal.json`
- Step plot: `results/nlp/stiefel_muon_b12_a4_lr003_formal_metric_steps.png`
- Time plot: `results/nlp/stiefel_muon_b12_a4_lr003_formal_metric_time.png`
- Checkpoint: `.cache/nlp/checkpoints/nlp_gpt_12x512__stiefel_muon_b12_a4_lr003_formal.checkpoint.pt`
