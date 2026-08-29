# Generalized Gauss--Newton reference retuning

## Scope

This record replaces an incorrect K-FAC interpretation. The retrained NLP
baseline is the 54,682,624-parameter `gpt_12x512` causal language model on the
cached WikiText-103 task. The earlier layer-wise Kronecker approximation is an
ablation, not the original Gauss--Newton solver; the active reference path is
matrix-free full generalized Gauss--Newton (GGN).

## Sources used

- Nicol Schraudolph, *Fast Curvature Matrix-Vector Products for Second-Order
  Gradient Descent* (2002): the positive-semidefinite generalized GGN used for
  classification losses.
- James Martens, *Deep Learning via Hessian-Free Optimization* (2010): damped
  curvature solves and Levenberg--Marquardt actual-versus-predicted reduction.
- `ltatzel/PyTorchHessianFree`, a public Martens-style reference implementation:
  adaptive damping, CG warm starts, backtracking, and line search.

The original sources do **not** prescribe K-FAC's factored damping or its
gradient-space momentum. Those changes are excluded from the reference path.

## Exact full-GGN feasibility and initialization check

An all-parameter matrix-free operator now computes \(Gv = J^T H_\ell Jv\)
with forward- and reverse-mode automatic differentiation. Its small linear
least-squares test matches the analytic Hessian exactly. On the 54.68M model,
a one-sequence, 128-token curvature batch completed one exact product in
0.453 seconds with 1,857 MB peak allocated GPU memory, so an exact
Hessian-free implementation is feasible on the local 16 GB GPU.

The same probe found that the current decoder's default initialization gives
logit standard deviation 22.784 and an exactly zero numerical GGN product for
a random full-parameter direction: the softmax Hessian is saturated at the
start. This is a property of the current baseline initialization, not an
out-of-memory limitation. The next qualification must therefore test a
properly damped first step and its actual reduction before committing to a
four-hour run. Kronecker tuning results below are not used to set full-GGN CG
parameters.

## Batch-size qualification

All candidates use eight sequences per optimizer update.

| Micro batch | Accumulation | Peak allocated memory | Steps/s | Metric at step 512 |
|---:|---:|---:|---:|---:|
| 2 | 4 | 4,974 MB | 4.472 | 0.172356 |
| 4 | 2 | 7,517 MB | 4.538 | 0.250790 |
| 8 | 1 | 12,573 MB | 4.407 | 0.247593 |

Micro batch four with accumulation two was selected before all hyperparameter
sweeps: it was fastest, had the best fixed-step metric, and retains more than
8 GB of GPU-memory headroom.

## Fixed-damping pilots

| Step scale | Initial damping | Step 512 | Final pilot metric | Decision |
|---:|---:|---:|---:|---|
| 0.03 | 0.03 | 0.250790 | 0.198177 at 1,362 steps | superseded |
| 0.10 | 0.03 | 0.222050 | 0.262875 at 1,272 steps | selected for full run |
| 0.10 | 0.10 | 0.149731 | 0.225246 at 1,359 steps | rejected |

The Martens-scale initial damping of 0.1 was materially worse here.  This does
not contradict the reference method: its exact GGN plus a full linear solve is
not the same curvature model as the local Kronecker approximation.

## Actual-versus-predicted reduction control

The baseline now supports an optional Levenberg--Marquardt damping update.  At
each requested interval, it evaluates the pre-update loss and the candidate
loss over the same accumulated training micro-batches, computes the damped
quadratic prediction, and updates every layer's spectral operator after a
damping change.  Unit coverage verifies that an agreeing scalar GGN quadratic
reduces damping.

The first language-model pilot at step scale 0.10 and initial damping 0.03 was
rejected after 512 steps: predicted reduction was 12.2773, actual reduction
was 0.7623, ratio 0.0621, and the repeated damping increases reduced validation
metric to 0.156555.  The factorized local model is too miscalibrated for this
adaptive rule at that cadence, so it is not used for the final fixed-damping
run.

## Active run

The retraining run started from a fresh model state with micro batch four,
accumulation two, step scale 0.10, damping 0.03, and relative per-parameter
trust cap 0.02.  It has a 14,400-second training limit and writes checkpoints
to `.cache/nlp/checkpoints`, metrics to `metrics/nlp`, and the required PNG
plots to `results/nlp`.

## Sources

- https://nic.schraudolph.org/pubs/Schraudolph02.pdf
- https://www.cs.utoronto.ca/~jmartens/docs/Deep_HessianFree.pdf
- https://github.com/ltatzel/PyTorchHessianFree
