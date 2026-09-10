# Low-Spectral-Variance on GPT-2 12x512

## Scope

This record evaluates the user-provided `Low-Spectral-Variance.md` method on
the repository's matched NLP task: GPT-2 12x512 (54,682,624 parameters), the
cached language data, micro-batch size 12, and gradient accumulation 4.  Its
final comparison will use the already tuned baselines: AdamW learning rate
`1.5e-4` and Muon learning rate `2.5e-3`.

## Implemented numerical method

For each already-feasible Muon-eligible matrix, the implementation normalizes
the largest singular value to one, approximately solves the paper's dual
spectral-norm direction, removes the top-singular tangent component, and takes
the source's low-rank direct Cayley finite curve.  When an extreme singular
value is repeated, the source's singular-vector velocity is not uniquely
defined; only for that degenerate update, the code takes the finite tangent
step and exactly retracts its spectrum to
`sigma_1 = 1, sigma_min >= 1 / c`.  Matrices not feasible at initialization
remain in the AdamW auxiliary group: the source derivation assumes a feasible,
full-column-rank starting point and this experiment does not silently project
the GPT initialization.

The dual polar factor uses the source's finite Newton-Schulz iteration,
`X <- X (3I - X^T X) / 2`, after Frobenius normalization.  This is a finite
numerical approximation to the dual solve, and the degenerate-value fallback
means the experiment is not a claim that every finite iterate follows an exact
Cayley curve.

## Feasibility and speed preflights

At condition cap `c=8`, 36 of 48 Muon-eligible GPT matrices are initially
feasible; they contain 20,079,616 parameters.  The remaining parameters use
AdamW auxiliary updates.

| Run | Dual iterations | Updates | Measured seconds | Outcome |
| --- | ---: | ---: | ---: | --- |
| `preflight_c8_lr0003` | 8, SVD polar | 24 | 191.38 | Too slow for a five-epoch run |
| `preflight_ns_c8_lr0003` | 8, Newton-Schulz polar | 76 | 187.28 | 3.2x more updates; dual-count sweep required |
| `preflight_ns1_c8_lr0003` | 1, Newton-Schulz polar | 84 | 188.42 | Projection remained the dominant cost |
| `preflight_direct_c8_lr0003` | 1, direct Cayley curve | 111 | 187.70 | 1.46x faster than Newton-Schulz plus projection |

The time-limited preflights both emitted one held-out metric because neither
finished an epoch.  Their identical metric (`0.18969345`) is not evidence of
optimizer equivalence or a final quality comparison.

## Next decision

The direct configuration projects to about 4.8 hours at effective batch 48,
which exceeds the per-experiment four-hour cap.  The next LR screens therefore
use the same micro-batch 12 and cached examples but gradient accumulation 12
(effective batch 144); this reduces the expensive feasible update frequency
and has an approximately 3.4-hour five-epoch estimate.  The final comparison
will disclose that AdamW and Muon reference curves use the existing tuned
effective batch 48 configuration.

## Learning-rate screen

| Label | Cap | LR | Effective batch | Epochs / updates | Metric | Seconds | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tune_direct_c8_lr0003_b12_a12` | 8 | 0.0003 | 144 | 1 / 678 | 0.292038 | 2,401.45 | Stable but far below the first-epoch AdamW (0.682594) and Muon (0.714909) references; screen a materially larger LSV LR. |
| `tune_direct_c8_lr003_b12_a12` | 8 | 0.003 | 144 | 1 / 678 | 0.299140 | 2,401.96 | Best of the two learning-rate screens; promote to the capped five-epoch run. |

The lower first-epoch metric is empirical evidence, not an assertion that the
method has converged or that batch-144 traces can be treated as a strict
optimizer-only comparison to the retained batch-48 baselines.

## Formal five-epoch result

The selected `c=8`, `lr=0.003`, one-dual-step configuration completed all five
epochs under the 14,400-second limit.  A first attempt stopped in epoch 3 on a
repeated extreme singular value; the direct-Cayley implementation was then
made robust with the explicitly documented exact-spectrum fallback, covered by
a regression test, and the formal run was restarted from a fresh checkpoint.

| Optimizer | Effective batch | Final metric | Training seconds | Status |
| --- | ---: | ---: | ---: | --- |
| AdamW, tuned `1.5e-4` | 48 | 0.723747 | 9,775.8 | retained reference |
| Muon, tuned `2.5e-3` | 48 | 0.755032 | 9,886.2 | retained reference |
| Low-Spectral-Variance, `c=8`, `lr=0.003` | 144 | 0.548587 | 12,051.9 | completed |

The final LSV curve is lower than both retained tuned baselines, and it takes
longer despite the larger effective batch.  It is a valid same-model,
same-cached-data comparison, but not a strict optimizer-only batch-matched
comparison because the LSV run uses accumulation 12 to fit the four-hour cap.
The result JSON and both metric-vs-step and metric-vs-time plots are retained
under `results/nlp`.
