# Full GGN with gradient batch 60 and curvature batch 60

## Objective

Test exact matrix-free Generalized Gauss--Newton (GGN) on the retained
54.68M-parameter `nlp_gpt_12x512` task while increasing both the gradient and
curvature effective batch sizes to 60.  The task, model, tokenizer, data order,
context length, and validation metric are unchanged from the AdamW and Muon
baselines.

## Batch and curvature construction

- Physical micro-batch: 1 sequence.
- Sequence length: 1024 tokens.
- Gradient accumulation: 60 micro-batches, effective batch size 60.
- Curvature accumulation: the same fixed 60 micro-batches, effective curvature
  batch size 60.
- Each GGN-vector product is computed without materializing the curvature
  matrix:

  \[
  Gv=\frac{1}{60}\sum_{b=1}^{60}J_b^\top H_{L,b}J_bv.
  \]

  The same 60 batches are reused within every Conjugate Gradient (CG) solve.
- Micro-batch 2 at sequence length 1024 was previously measured to run out of
  the 16 GB local GPU memory, so accumulation is the safe way to reach batch 60.

## Correctness repairs made before the formal comparison

1. A characterization test verifies that all accumulated curvature operators
   are averaged; the hand-derived two-batch product is 1.25.
2. Candidate evaluation left the model at a candidate point before computing
   the predicted reduction.  This mixed the original gradient with curvature
   evaluated at new parameters and produced negative predicted reductions and
   invalid damping ratios.  A failing scalar regression test reproduced
   `-0.25` instead of the hand-derived `0.375`.  Restoring the original
   parameters before the curvature product fixed the test.
3. Labeled Full GGN trials were absent from the comparison plots.  Plot routing
   now accepts the selected Full GGN run label, with a regression test covering
   both Metric--Steps and Metric--Time outputs.

## Controlled results

All GGN rows use initial damping `0.001`, minimum damping `0.0001`, initial
step scale `1`, no CG warm start, and the batch construction above.  Each
formal comparison was run for approximately 1200 seconds on the local RTX 5070
Ti.  The step-scale-4 diagnostic used 600 seconds and is retained only as a
negative control.

| Run | CG iterations | Steps | Best accuracy | Best step | Time at best | Final accuracy | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed Full GGN | 1 | 65 | 0.243561 | 64 | 1188.48 s | 0.241104 | 9883 MB |
| Fixed Full GGN | 2 | 50 | **0.253296** | 48 | 1153.26 s | 0.252777 | 10092 MB |
| Step-scale-4 diagnostic, pre-fix process | 1 | 31 | 0.191254 | 8 | 153.77 s | 0.191254 | not used for conclusions |
| AdamW baseline | n/a | 5 epochs | 0.711754 | epoch 5 | 10034.52 s total | 0.711754 | existing baseline |
| Muon baseline | n/a | 5 epochs | **0.752193** | epoch 5 | 10418.21 s total | 0.752193 | existing baseline |

At comparable GGN metric levels, CG=2 reached 0.226730 in 583.60 seconds,
whereas CG=1 reached 0.225662 in 896.98 seconds.  CG=2 therefore reduced the
observed time-to-about-0.226 by 34.9%.  It also exceeded the entire CG=1 run's
best accuracy in 775.84 seconds (0.248566).

## Interpretation

Curvature accumulation over 60 samples is operational and stable, but it is
not sufficient by itself.  With a zero initial direction, one CG iteration
produces a direction of the form `-alpha * gradient`; curvature only determines
one global scalar.  Two CG iterations are the first tested configuration that
can change the direction using GGN anisotropy, and it improves both the best
metric and time-to-metric over CG=1.

The result is still not competitive with the retained AdamW or Muon baselines.
The experiment supports further damping/CG tuning; it does not support a claim
that Full GGN currently beats either baseline.

## Artifacts

- CG=1 metrics: `metrics/nlp/nlp_gpt_12x512__full_ggn_curv60_b1_s1024_grad60_d001_floor0001_cg1_fixed.jsonl`
- CG=2 metrics: `metrics/nlp/nlp_gpt_12x512__full_ggn_curv60_b1_s1024_grad60_d001_floor0001_cg2_fixed.jsonl`
- CG=1 result: `results/nlp/nlp_gpt_12x512__full_ggn_curv60_b1_s1024_grad60_d001_floor0001_cg1_fixed.json`
- CG=2 result: `results/nlp/nlp_gpt_12x512__full_ggn_curv60_b1_s1024_grad60_d001_floor0001_cg2_fixed.json`
- Metric--Steps: `results/nlp/nlp_gpt_12x512_gn_metric_steps.png`
- Metric--Time: `results/nlp/nlp_gpt_12x512_gn_metric_time.png`
- Checkpoints: `.cache/nlp/checkpoints/`

## Verification

The full repository suite passed after the implementation and plotting fixes:
`189 passed`.
