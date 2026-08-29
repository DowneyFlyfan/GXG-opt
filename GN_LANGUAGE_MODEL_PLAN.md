# Language-model GN experiment plan

Run the 54.68M GPT-12x512 language model on cached WikiText-103 with two
curvature optimizers: Kronecker generalized Gauss–Newton (GN) and its signed
low-rank residual correction (DG-GN).  Existing AdamW and Muon runs remain
comparison baselines.

| Phase | Deliverable | Gate |
|---|---|---|
| Integration | Cached NLP curvature closure and resumable GN runner | MC-GN factors and correction operators are exercised, not fallback-only |
| Qualification | One-step GPU memory/time measurements for both optimizers | no OOM; projected run fits four hours |
| Runs | GN and DG-GN metric records, checkpoints, and result JSON | completed or safely checkpointed at the four-hour boundary |
| Reporting | Metric-step and metric-time PNGs plus a `records` report | all four optimizer traces are identified and reproducible |

The first implementation test will assert that a causal language-model batch
produces factors for supported linear layers and an exact residual operator
only when the low-rank correction requests it.  The production runs use only
data already under `.cache`.

## Gauss--Newton tuning basis

The baseline is a Kronecker approximation to the generalized
Gauss--Newton (GGN) matrix, not K-FAC.  Its tuning and acceptance procedure
therefore follows the GGN/Hessian-free literature:

- Schraudolph (2002) supplies the positive-semidefinite generalized
  Gauss--Newton construction used for classification losses.
- Martens (2010) supplies the large-model optimization procedure: damp the
  curvature system, solve a local quadratic model, compare predicted against
  actual reduction, and adapt damping with a Levenberg--Marquardt rule.
- The reference Hessian-free implementation additionally warms starts its
  linear solve and performs curvature-step backtracking plus an Armijo line
  search.  The Kronecker baseline has a direct spectral solve, so it has no
  conjugate-gradient iteration to warm start; it still needs the damping and
  actual-reduction controls.

The first sweep fixed the effective batch at eight sequences and selected
micro-batch four with accumulation two.  The next sweeps tune damping and the
outer step scale together, then add the Martens-style actual-versus-predicted
reduction control before the four-hour final run.  K-FAC-specific momentum and
factorized-damping settings are explicitly out of scope for this GN baseline.

Sources: [Schraudolph 2002](https://nic.schraudolph.org/pubs/Schraudolph02.pdf),
[Martens 2010](https://www.cs.utoronto.ca/~jmartens/docs/Deep_HessianFree.pdf),
and the [PyTorch Hessian-free reference implementation](https://github.com/ltatzel/PyTorchHessianFree).
