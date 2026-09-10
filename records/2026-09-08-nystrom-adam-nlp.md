# Nyström-GGN Then AdamW on GPT-2 12x512

## Scope

This is the requested sequential experiment on the same cached GPT-2 12x512
NLP task (54,682,624 parameters).  Stage 1 performs one Nyström-approximated
Generalized Gauss-Newton (Nyström-GGN) update; stage 2 loads only the resulting
model weights into a newly initialized AdamW optimizer.  It does not transfer
or invent first- or second-moment state.

## Stage 1: Nyström-GGN

The earlier zero-update behavior was not a rank issue.  Its line-search scale
was too large relative to the damped inverse-curvature action.  The stage used
rank 4, 4 Nyström columns, curvature batch 64, damping 0.1, refresh interval
4, and an initial scale of `1.5e-5`.

| Quantity | Value |
| --- | ---: |
| Selected line-search step | `1.5e-5` (nonzero) |
| Stage metric | 0.670654 |
| Curvature-stage total seconds | 1,285.66 |
| Post-warmup stage seconds | 177.87 |
| Peak stage memory | 8,505.31 MiB |

The handoff is guarded: it would have raised instead of starting AdamW if any
selected Nyström line-search step were zero.

## Stage 2: fresh AdamW

AdamW used the tuned baseline learning rate `1.5e-4`, micro-batch 12, gradient
accumulation 4 (effective batch 48), and five epochs.  The combined trace
contains the stage-1 point followed by AdamW epochs.

| AdamW epoch | Metric | Combined reported seconds |
| ---: | ---: | ---: |
| Nyström stage | 0.670654 | 177.87 |
| 1 | 0.695802 | 2,133.20 |
| 2 | 0.706610 | 4,088.85 |
| 3 | 0.712943 | 6,044.42 |
| 4 | 0.717798 | 7,999.95 |
| 5 | 0.720018 | 9,955.43 |

`seconds` in the combined result deliberately starts from the post-warmup
Nyström time, so the stage's 1,107.79 seconds of warmup are not represented on
the combined time axis.  The full measured wall-clock accounting is therefore
about 11,063.22 seconds (`9,955.43 + 1,285.66 - 177.87`).

## Comparison and conclusion

| Optimizer | Final metric | Reported training seconds |
| --- | ---: | ---: |
| AdamW, tuned `1.5e-4` | 0.723747 | 9,775.8 |
| Muon, tuned `2.5e-3` | 0.755032 | 9,886.2 |
| Nyström-GGN → fresh AdamW | 0.720018 | 9,955.43 (11,063.22 including all stage time) |

The nonzero Nyström update makes the requested two-stage method genuine, but
its final metric is slightly below tuned AdamW and materially below tuned
Muon.  It also does not improve total elapsed time after including curvature
warmup.  The final result JSON, stage JSON, metric JSONL files, and both
comparison PNGs are retained under `results/nlp` and `metrics/nlp`.
