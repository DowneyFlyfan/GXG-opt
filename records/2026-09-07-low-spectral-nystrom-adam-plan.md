# Low-Spectral-Variance and Nyström-to-AdamW Plan

## Objective

Evaluate the user-provided Low-Spectral-Variance method on cached GPT-2 12x512
Natural Language Processing data, with parameter tuning and final plots against
the selected AdamW and Muon baselines.  Then run a two-stage Nyström
Generalized Gauss-Newton (Nyström-GGN) update followed by AdamW on the same
model and dataset.

## Fixed references

- Model: GPT-2 12x512, 54,682,624 parameters.
- Data: cached WikiText-103 task stream.
- Baselines: five-epoch AdamW at learning rate `1.5e-4`; five-epoch Muon at
  learning rate `2.5e-3`, both micro-batch 12 / accumulation 4.
- Per-run time cap: 14,400 seconds; local RTX 5070 Ti only.

## Low-Spectral-Variance

1. Implement the dual tangent direction and the source's low-rank direct
   Cayley feasible curve.  Pass matrix geometry, artifact-contract, and
   optimizer-routing tests.
2. Retain preflights for the SVD polar, Newton-Schulz polar, and direct-Cayley
   variants.  Select the direct curve because it achieves 111 updates in
   187.70 seconds at `c=8`, versus 84 with the projection-based variant.
3. Run LR screens at `c=8`, one dual iteration, micro-batch 12, and gradient
   accumulation 12.  The larger effective batch is disclosed because the
   direct method would exceed four hours at accumulation 4.
4. Select a stable LR from screen metrics, run the five-epoch formal result,
   render step/time PNG comparisons against AdamW and Muon, and inspect JSON,
   JSONL, PNG signatures, and the rendered plots.

## Nyström-GGN to AdamW

1. Run a rank-4, damping-0.1, curvature-batch-64 Nyström first stage from the
   established common warmup.  Use a small initial held-out line-search scale
   (`1.5e-5` candidate) because scale 1.0 previously produced only zero steps.
2. Require every selected first-stage line-search step to be nonzero.  A zero
   step is a rejected configuration, not a handoff.
3. Load the accepted Nyström model checkpoint and initialize a new AdamW
   optimizer at `1.5e-4`; do not transfer non-existent AdamW moments.
4. Complete AdamW training under the same cached GPT/data task, render the
   combined metric-step and metric-time comparisons, then record the stage
   scale, selected curvature step, AdamW settings, time, memory, and final
   metric.

## Completion evidence

Completion requires completed result JSON files, metric JSONL traces, two valid
PNG plots per final optimizer, stable test evidence, records describing any
batch/configuration differences, and committed/pushed artifacts.  A preflight,
screen, zero Nyström step, or code-only implementation is insufficient.
