# Language-model GN experiment plan

Run the 54.68M GPT-12x512 language model on cached WikiText-103 with an exact,
matrix-free generalized Gauss–Newton (GGN) baseline and the project's
low-rank DG-GN comparison. Existing AdamW and Muon runs remain comparison
baselines. The earlier layer-wise Kronecker experiment is retained only as an
ablation; it is not the reference Gauss–Newton solver.

| Phase | Deliverable | Gate |
|---|---|---|
| Integration | Cached NLP full-GGN operator, conjugate-gradient solve, and resumable runner | exact \(Gv=J^T H_{\ell}Jv\) is exercised over every trainable parameter |
| Qualification | One-step GPU memory/time and curvature-scale measurements | no OOM; nonzero curvature or a documented damping-only start |
| Runs | GN and DG-GN metric records, checkpoints, and result JSON | completed or safely checkpointed at the four-hour boundary |
| Reporting | Metric-step and metric-time PNGs plus a `records` report | all four optimizer traces are identified and reproducible |

The first implementation test will assert that a causal language-model batch
produces factors for supported linear layers and an exact residual operator
only when the low-rank correction requests it.  The production runs use only
data already under `.cache`.

## Gauss--Newton tuning basis

The reference baseline follows generalized Gauss--Newton/Hessian-free
optimization rather than K-FAC:

- Schraudolph (2002) supplies the positive-semidefinite generalized
  Gauss--Newton construction used for classification losses.
- Martens (2010) supplies the large-model optimization procedure: damp the
  curvature system, solve a local quadratic model, compare predicted against
  actual reduction, and adapt damping with a Levenberg--Marquardt rule.
- The reference Hessian-free implementation additionally warm-starts
  conjugate gradient, retains candidate CG iterates for backtracking, and uses
  an Armijo line search.

The first sweep fixed the effective batch at eight sequences and selected
micro-batch four with accumulation two. The full-GGN memory probe uses a
single 128-token curvature sequence and only 1.86 GB peak allocated memory.
It also revealed that the current default decoder initialization produces
logit standard deviation 22.8 and an effectively zero initial GGN product
because softmax is saturated. The next gate is therefore a damped CG
qualification: establish a finite, descending first update and use the
actual-versus-predicted reduction ratio to adapt damping. K-FAC-specific
momentum and factorized-damping settings are out of scope.

Sources: [Schraudolph 2002](https://nic.schraudolph.org/pubs/Schraudolph02.pdf),
[Martens 2010](https://www.cs.utoronto.ca/~jmartens/docs/Deep_HessianFree.pdf),
and the [PyTorch Hessian-free reference implementation](https://github.com/ltatzel/PyTorchHessianFree).
