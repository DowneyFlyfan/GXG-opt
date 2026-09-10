# Multi-Step Spectral Geometry: local idea versus fetched remote implementation

## Remote revision and integration state

`git fetch origin --prune` retrieved `origin/main` at `4c5eb1a` (`feat: add
GPT-2 spectral geometry comparison`).  A fast-forward pull is impossible because
the local SRIP result commit and the remote implementation commit share
`f0103a8` as a parent.  The worktree also has a user-modified `README.md`, so a
merge must not be forced while that change is unresolved.  This comparison is
therefore based on read-only `git show origin/main:<path>` inspection.

## Updating formula

The fetched optimizer implements the same method described in
`records/ideas/Multi-Step-Spectral-Geometry.md`:

1. It forms per-matrix exponential-moving-average momentum
   \(M_t=\beta M_{t-1}+(1-\beta)G_t\).
2. It chooses among Schatten geometries \(p\in\{2,4,8,\infty\}\), using the
   energy-matched direction \(\sqrt{r}\,\Phi_p(M_t)/\|\Phi_p(M_t)\|_F\).
3. It fits a low-rank projected secant curvature model from parameter
   displacements and gradient changes, symmetrizes it, and projects its
   eigenvalues into a positive-semidefinite interval.
4. It scores every candidate through a four-step local-loss rollout plus a
   geometry-switch penalty, retaining the previous geometry unless the proposed
   score wins by the configured margin.
5. It applies \(W_{t+1}=W_t-\eta\,s(W_t)T_{p_t}(M_t)\), while vector and
   excluded parameters use AdamW fallback.

Thus there is no conceptual updating-formula mismatch.  The remote
implementation realizes the local document's `Hraw -> symmetric PSD
eigenvalue-clipped -> perpendicular-curvature` construction and its horizon
policy.  Its `reduced` transform replaces an exact singular value decomposition
with an eight-step Newton--Schulz / degree-12 polynomial approximation; that is
an implementation-speed choice, not a different optimizer objective.

## Hyperparameter and protocol differences

| Item | Fetched remote comparison | Required local matched experiment |
| --- | --- | --- |
| Model | pretrained `openai-community/gpt2`, about 124M parameters | scratch GPT2-12x512, 54,682,624 parameters |
| Dataset | Wikitext-103 raw, sequence length 512 | existing cached NLP dataset used by AdamW and Muon |
| Training duration | 2,000 steps | 5 epochs = 10,170 optimizer updates |
| Seeds | 0, 1, 2 | matched single-run protocol unless the baseline protocol is expanded consistently |
| Micro-batch / accumulation | 16 / 2 (effective 32) | 12 / 4 (effective 48) |
| Optimizer learning rate | `5e-5` | must be tuned for scratch GPT2-12x512, not copied |
| Weight decay | 0.01 | follow the project big-model rule and matched baseline protocol |
| Momentum | 0.95 | candidate starting value 0.95; verify through tuning |
| Candidate p | 2, 4, 8, infinity | same candidate set |
| Policy | warm-up 500, interval 100, horizon 4; weights 1, .8, .6, .4 | same starting policy, scheduled against 10,170 updates |
| Curvature | history/basis rank 8, fit interval 50, ridge `1e-4`, interval [0,100] | same starting configuration, audited in the local experiment record |

## Decision for the later run

Reuse the fetched formula and candidate-policy defaults, but adapt the runner
to the local scratch model, cached dataset, effective batch 48, and exact
five-epoch boundary.  Do not present the remote pretrained 2,000-step experiment
as a baseline-comparable result.  The Multi-Step run is scheduled after the
active five-epoch strict-feasible SRIP run and the Stiefel-feature method.
