# Recycled low-rank Gauss--Newton: equal-batch formal result

## Question

Can a low-rank approximation retain the optimisation benefit of the
paper-faithful Full Gauss--Newton (GN) baseline while avoiding its repeated
full-batch curvature scans?

## Controlled task

- Task: cached WikiText-103 next-token prediction with GPT-12x512.
- Parameters: 54,682,624.
- Shared start: the completed common AdamW warmup checkpoint, after 54,685,696
  warmup tokens.
- Sequence length: 1,024.
- Validation: unchanged held-out next-token accuracy.
- Formal post-warmup budget: 13,189.76 seconds.  Step 29 was already in
  progress at the budget boundary, so it was finished and evaluated as required;
  the reported post-warmup time is 13,492.53 seconds.

## Method and exact batch contract

The update approximates

\[
  (G + 0.1I)^{-1}(-g), \qquad G = J^\top H_L J,
\]

with a factored diagonal Kronecker base plus a rank-four signed, whitened
secant residual correction.  It never constructs either \(G\) or \(G^{-1}\).
For a normalized gradient direction \(s\), one exact matrix-free product
\(Gs\) calibrates the base and contributes a secant pair.  The inverse action
is then direct rather than a preconditioned conjugate-gradient loop.

| Quantity | Value |
|---|---:|
| Physical microbatch | 2 sequences |
| Gradient effective batch | 3,904 = 1,952 microbatches |
| Curvature effective batch | 3,904 = 1,952 microbatches |
| Curvature construction | all gradient-batch samples, sample weighted |
| Curvature products | 1 per outer update; 29 total |
| Residual rank / storage | 4 / bfloat16 |
| Spectral floor | relative eigenvalue \(\geq 0.25\) |
| Held-out line-search scales | 0, 0.025, 0.0354, 0.05, 0.0707, 0.1 |
| Peak allocated local GPU memory | 14,213.60 MiB |

The gradient and curvature statistics therefore use the identical effective
batch.  Gradient accumulation is not being used only for the gradient: the
same 1,952-way accumulation is applied to the Generalized Gauss--Newton (GGN)
operator.

## Result

| Optimizer | Final validation accuracy | Recorded wall time | Notes |
|---|---:|---:|---|
| AdamW | 0.711754 | 10,034.52 s | retained baseline, five epochs |
| Muon | 0.752193 | 10,418.21 s | retained separately tuned baseline, five epochs |
| Paper Full GN | 0.715515 | 14,296.38 s | same warmup and 13,189.76 s post-warmup budget |
| Recycled low-rank GN | 0.666931 | 14,599.15 s | 29 outer updates; last begun update completed |

The low-rank method is numerically stable: all 29 directions were finite and
descending, and its metric rose from 0.661888 after the first update to
0.666931.  It did **not** retain the performance of Full GN, AdamW, or Muon in
this controlled run.  The result is therefore a negative accuracy result, not
evidence that the low-rank replacement is ready.

## Interpretation

One full-batch secant pair per update is too weak to approximate the important
curvature subspace of this 54.7M-parameter language model.  The rank-four
correction needs a spectral floor to remain positive definite; that safeguard
limits harmful inverse amplification but also makes the update close to its
diagonal base.  The low-rank method has reduced the number of curvature scans
per update, but the remaining single exact \(Gv\) still takes about 283 seconds
at the required equal batch.  Its present bottleneck is approximation quality,
not merely matrix inversion cost.

## Artifacts

- Metrics: `metrics/nlp/nlp_gpt_12x512__lowrank_secant_equalbatch3904_b2_s01_floor025_formal.jsonl`.
- Result: `results/nlp/nlp_gpt_12x512__lowrank_secant_equalbatch3904_b2_s01_floor025_formal.json`.
- Plots: `results/nlp/lowrank_secant_equalbatch3904_b2_s01_floor025_formal_metric_steps.png` and
  `results/nlp/lowrank_secant_equalbatch3904_b2_s01_floor025_formal_metric_time.png`.
- Checkpoint and stored rank-four bases: `.cache/nlp/checkpoints/`.

## Next research decision

Do not scale this exact one-pair method to more tasks yet.  The next candidate
should collect several independent secant directions per curvature batch (or a
blockwise low-rank approximation) and measure its inverse-action residual
against the Full GN direction before spending another equal-duration run.
