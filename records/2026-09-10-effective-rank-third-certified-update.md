# Effective-Rank-Third certified update

## Motivation

The effective-rank-half candidate completed five matched GPT2-12x512 epochs at
0.75524139 with zero weight decay. Its subsequent 0.01-decay first-epoch
screen reached 0.70972061, but the decay is not required for the next study.
The incomplete five-epoch decay confirmation was deliberately terminated before
an epoch completed, at the request to test a different constraint; it created
no metric, result, or checkpoint artifact.

## Constraint and implementation

For a nonzero matrix \(W\in\mathbb{R}^{m\times n}\), let
\(r=\min(m,n)\) and define normalized Frobenius effective rank

\[
 e(W)=\frac{\lVert W\rVert_F^4}{r\lVert W^\top W\rVert_F^2}.
\]

The new method requires \(e(W)\geq 1/3\), replacing the old \(e(W)\geq
1/2\) lower bound. Every matrix update is a partial-polar direction with
spectral norm at most one. The scalar multiplier is accepted directly when the
finite candidate is feasible; otherwise a 48-step bisection returns the largest
feasible scalar. The numerical feasibility margin remains 1e-6 for float32 and
1e-10 for float64, and the zero-decay study does not invoke decoupled decay.

The implementation retains the historical `effective_rank_half` optimizer and
adds a separate `effective_rank_third` registry entry, runner, and four-way
plotter. This prevents the completed half-rank evidence from being relabeled as
one-third evidence.

## Verification before training

The red tests first established that neither the one-third API, registry, nor
runner existed. The final targeted suite passed **22 tests**: it verifies that
a matrix with \(e(W)=0.4\) is rejected at the half bound but accepted and kept
above \(1/3\) at the new bound; verifies a real optimizer step and model
builder; verifies runner arguments; and writes both step and time comparison
plots against AdamW, Muon, and Muown.

## Matched screen

The fresh initial screen uses the previously selected aggressive learning rate
0.00125, zero weight decay, micro-batch 8, gradient accumulation 6, four data
workers, one epoch, and the local GPU. It is a tuning screen only. A fresh
five-epoch run will be promoted only if its completed one-epoch metric warrants
it; all checkpoints remain confined to `.cache/nlp/checkpoints`.

## Screen result

`lr000125_b8_a6_screen` completed one fresh matched epoch with zero weight
decay. It reached validation next-token accuracy **0.71140671** in 3,625.27
seconds with 8,496.83 MiB peak allocated memory. This is **+0.00184631** above
the matched effective-rank-half zero-decay epoch-one value (0.70956039), so the
one-third constraint is a promising candidate for a later independent
five-epoch confirmation. Its checkpoint was automatically removed from
`.cache`, and no checkpoint was written under `results/nlp`.
