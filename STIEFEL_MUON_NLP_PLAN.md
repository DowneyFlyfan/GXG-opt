# Stiefel-Muon on the retained NLP baseline

## Objective

Evaluate the closed-form orthogonality-constrained Muon updates from the two
specified articles on the retained GPT-12x512 WikiText-103 baseline, and compare
their validation accuracy to the existing, separately tuned Muon trace.

## Shape audit

The original assumption is false for this model.  The 48 matrix parameters that
the baseline routes to Muon comprise 12 square projection matrices and 36
rectangular matrices: QKV projections, feed-forward expansion matrices, and
feed-forward contraction matrices in every Transformer block.  The tied token
embedding/output matrix remains on AdamW, exactly as in the retained Muon
baseline.

| Matrix shape in storage | Count | Update route |
|---|---:|---|
| 512 by 512 | 12 | square orthogonal closed form from article 11215 |
| 1536 by 512 | 12 | column-Stiefel closed form from article 11864 |
| 2048 by 512 | 12 | column-Stiefel closed form from article 11864 |
| 512 by 2048 | 12 | transpose, column-Stiefel update, transpose back |

## Experiment contract

- Model/data/metric/seed: unchanged GPT-12x512, cached WikiText-103,
  deterministic seed 1337, and validation next-token accuracy.
- Stiefel initialization: every Stiefel-Muon matrix is projected to the correct
  column-orthonormal manifold before the first update; this is required by both
  article derivations.
- Optimizer: exponential moving average of the gradient, Newton-Schulz matrix
  sign approximation, exact article direction, and manifold retraction after
  each update.  Non-matrix, embedding, and output-head parameters remain AdamW.
- Batch: profile physical batch 4 and accumulation 2 first; only increase it
  after a no-OOM measurement.
- Tuning: use short, checkpoint-isolated local-GPU probes to select base learning
  rate and momentum; a formal run is limited to four hours and finishes the
  final started optimizer step.
- Comparators: retained AdamW and original Muon metrics; plots show metric versus
  optimizer step and wall-clock time.

## Completion gates

1. Algebraic unit tests prove square and rectangular updates remain on their
   respective manifolds, including row-Stiefel transposition.
2. A GPU profile establishes the largest safe batch and records peak memory.
3. At least two short learning-rate probes are run before the formal candidate.
4. The formal run writes metric JSONL, JSON result, both PNG plots, checkpoint,
   and a detailed record in `records`.
5. Tests pass and only method-specific files are committed and pushed.
