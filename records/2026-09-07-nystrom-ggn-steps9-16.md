# Nyström-GGN extension: outer steps 9--16

## Objective

Extend the completed GPT-2 12x512 Nyström Generalized Gauss-Newton trace from
outer step 8 to outer step 16 without changing the experimental contract:
physical batch 1, outer effective batch 3,904, curvature batch 64, four
sampled columns, damping 0.1, and factor refresh every four outer steps.

## Exact resumption

The checkpoint for `nystrom_ggn_formal_r4_c64_d01_s4` was inspected before
resumption.  It contained the model at step 8, stream position 62,464,
post-warmup elapsed time 1,322.381 s, and a three-column effective factor.
Steps 9--12 resumed successfully and preserved the metric 0.670380.  The
line search selected step size zero on every one of those steps.

## Step-13 failure and root cause

At the scheduled step-13 refresh, the old code raised `ValueError: Nyström
sampled intersection has no positive eigenvalue`.  A diagnostic replay from
the saved step-12 state used the same deterministic stream position and
sampled indices.  The finite symmetric sampled intersection had eigenvalues

```
[6.2706806e-10, 1.3228314e-09, 3.5151537e-08, 6.7870062e-08].
```

Thus the intersection was strictly positive.  The failure arose because the
old threshold was `float32_epsilon * max(max_abs_eigenvalue, 1)`, namely
approximately `1.19e-07`, which exceeded every legitimate positive
eigenvalue.  The implementation now uses the relative threshold
`float32_epsilon * max_abs_eigenvalue`.  The regression test
`test_nystrom_retains_strictly_positive_small_curvature_intersection` failed
before this single change and passes afterward.

## Extended result

The retry resumed from the unchanged step-12 checkpoint and completed steps
13--16.  Step 13 retained all four sampled spectral directions, crossing the
previous failure point.  The run completed with:

| Quantity | Value |
| --- | ---: |
| Outer steps | 16 |
| Training sequences consumed | 124,928 |
| Final validation next-token accuracy | 0.670380 |
| Post-warmup time | 2,668.60 s |
| Total time including shared warmup | 3,776.39 s |
| Peak GPU allocation | 8,713.90 MiB |

Every held-out search through step 16 selected a zero step, so the extension
is evidence of stable continuation and repaired positive-spectrum handling,
not of improved model quality.  The plot and the full 16-row metric trace are
updated in place.

## Evidence

- `records/2026-09-07-nystrom-ggn-continue-steps9-16.log`
- `records/2026-09-07-nystrom-ggn-step13-spectrum.log`
- `records/2026-09-07-nystrom-ggn-retry-steps13-16.log`
- `metrics/nlp/nlp_gpt_12x512__nystrom_ggn_formal_r4_c64_d01_s4.jsonl`
- `results/nlp/nlp_gpt_12x512__nystrom_ggn_formal_r4_c64_d01_s4.json`
