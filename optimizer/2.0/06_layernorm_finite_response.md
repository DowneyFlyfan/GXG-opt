# Idea 6 — Finite-response LayerNorm–matrix coupling

**Algorithm ID:** `ln_response_v1`\
**Status:** Exploratory hypothesis; the core solve is equivalent to a particular block-damped GGN model.\
**Target:** A small current-step correction coupling one GPT-2 MLP block with its preceding LayerNorm affine parameters.\
**Core requirement:** Preserve the finite-response formula, joint PSD curvature, and equivalence test. Do not market an ordinary damped solve as a new theorem.

## 1. Scope and important mathematical limitation

The motivating question is whether matrix updates should anticipate a finite amount of normalization-parameter response. Eliminating variables in a local dynamical model motivates such a response. It does not automatically create a superior optimizer.

This v1 implements a memoryless finite-horizon approximation. It does NOT implement the full Mori–Zwanzig memory convolution. Its finite-response solve is algebraically equivalent to anisotropically damping the normalization block of a GGN quadratic. The research claim, if any, must concern a useful response-based choice of that damping and its efficient Transformer approximation.

There is no Stiefel constraint, spectral-band preservation, learned Schatten policy, or curvature-teacher distillation. The correction is applied directly at an event and does not train a persistent inverse-action memory.

## 2. Work around a complete baseline proposal

Obtain the entire baseline increment d0 without committing. Define the provisional model

$$ \phi=\theta+d_0. \tag{1} $$

Use a functional model or reversible isolated snapshot for probes at phi. Do not mutate the live training model or baseline optimizer state during trial evaluation. Every trial shares all baseline updates, including decay; only the selected local correction varies.

Initially select one GPT-2 block with `ln_2` preceding its MLP. Target the MLP input/output projection weights for matrix coordinates x, and that LayerNorm's gain gamma and bias beta for coordinates y. Other weights, biases, and normalization parameters stay at phi during candidate evaluation.

## 3. Fixed small correction basis

Build orthonormal parameter-space columns U_x and U_y, with disjoint supports, before fitting the correction model:

- U_x: one unit-Frobenius direction from the baseline learning increment of each of the two MLP weight matrices. Drop a zero direction. These columns have disjoint supports, so at most two remain.
- U_y: an all-ones direction and the pre-update training-gradient direction in gamma, and the same pair in beta. Orthonormalize within each parameter vector, with deterministic reorthogonalization; drop dependent/zero columns. At most four remain.

Use the mathematical weight orientation defined in the shared contract. Flattening is only a view. All basis columns are detached and frozen throughout the event. Do not choose them using checking data. Baseline bias/LayerNorm increments are not replaced: x and y are additional corrections around phi.

$$ \theta_{trial}=\phi+U_xx+U_yy. \tag{2} $$

If either basis is empty, skip the event rather than silently changing the target mechanism. Scaling x and y corresponds to actual Frobenius parameter displacement because columns are orthonormal.

## 4. Reduced gradient and PSD curvature

On a fitting calibration batch from training data, compute at phi

$$ g_x=U_x^\top\nabla L_{fit}(\phi),\qquad
g_y=U_y^\top\nabla L_{fit}(\phi). \tag{3} $$

Let J_z be the full-model logit Jacobian at phi and let C_z be the output cross-entropy Hessian with the exact valid-token averaging convention. Form reduced columns T_x=J_z U_x and T_y=J_z U_y using directional autodiff. Then

$$ F_r=
\begin{bmatrix}T_x^\top C_zT_x&T_x^\top C_zT_y\\
T_y^\top C_zT_x&T_y^\top C_zT_y\end{bmatrix}\succeq0. \tag{4} $$

For a token probability p, the action needed here is `C_z v = p*(v-dot(p,v))/N`, applied blockwise over valid token positions. Do not build a vocabulary-square matrix in production. Compute every cross block from the same columns and probabilities. A shared PSD projection/sketch of the output space is a future labeled approximation; independent noisy estimates of A/B/C that break joint PSD are not v1.

Add positive damping to obtain

$$ \mathcal H=F_r+\lambda I=
\begin{bmatrix}A&B\\B^\top&C\end{bmatrix}\succ0. \tag{5} $$

Start with `lambda = max(0.1*trace(F_r)/dim(F_r), 1e-8)` in the reference units. The numerical floor is an explicit hyperparameter, not a unit-independent theorem. The shared trust-budget procedure below may increase lambda. Use a symmetric eigensolver/Cholesky in FP64 for the small matrices. Symmetrization is allowed for roundoff; a materially indefinite F_r indicates a bug or an unapproved approximation.

This is GGN at the provisional full model, not the exact weight Hessian and not curvature evaluated at the original theta. The extra fit forward/backward/JVPs are part of the cost.

## 5. Derive the finite response

For a fixed candidate matrix displacement x, consider the local normalization response initialized at y(0)=0:

$$ {dy(s)\over ds}=-\big(g_y+B^\top x+Cy(s)\big). \tag{6} $$

This is gradient flow of the **regularized local quadratic**, not a proven model of Adam's or the real network's normalization dynamics. With C positive definite,

$$ R_\tau=C^{-1}(I-e^{-\tau C}),\qquad
y(\tau)=-R_\tau(g_y+B^\top x). \tag{7} $$

Insert this response into the matrix-gradient stationarity equation `g_x+A x+B y=0`:

$$ S_\tau=A-BR_\tau B^\top,\qquad
S_\tau x_\tau=-\big(g_x-BR_\tau g_y\big),\quad
y_\tau=-R_\tau(g_y+B^\top x_\tau). \tag{8} $$

Do not omit the `B R_tau g_y` term. Changing its sign is a common serious error. Do not replace R_tau by `tau*I` outside the explicitly tested small-tau limit.

## 6. Stable computation, limits, and equivalent damping

For `C=V diag(c_j) V^T`, compute

$$ R_\tau=V\operatorname{diag}(r_j)V^\top,\qquad
r_j={-\operatorname{expm1}(-\tau c_j)\over c_j}. \tag{9} $$

The `expm1` form avoids cancellation at small tau. Handle tau=0 explicitly: R=0, x=-A^(-1)g_x, y=0. For the infinite-response comparison, R=C^(-1). All eigenvalues c_j are positive under (5).

Because `0 <= R_tau <= C^(-1)`, S_tau is positive definite when the joint H is positive definite. This justifies a small Cholesky solve, not global nonlinear descent.

For finite tau>0, the SAME pair (x_tau,y_tau) minimizes

$$ q_\tau(x,y)=g_x^\top x+g_y^\top y
+\tfrac12x^\top Ax+x^\top By+\tfrac12y^\top R_\tau^{-1}y. \tag{10} $$

In particular,

$$ R_\tau^{-1}-C\succeq0. \tag{11} $$

Thus the method is exactly a GGN quadratic with extra normalization-block damping. Implement this direct block solve as an independent oracle. It MUST agree with (8), not become a competitor the proposed formula supposedly outperforms. An asserted gain over the mathematically identical solve is evidence of a bug or mismatched settings.

As tau tends to infinity, the pair tends to the ordinary coupled damped-GGN/Schur solution. As tau tends to zero, the normalization correction vanishes. These limits do not imply that either is best for training.

## 7. Concrete v1 horizon and trust-budget selection

The earlier concept did not uniquely specify how to choose tau. Use this explicit, small candidate-selection rule for v1. It is a practical hypothesis, not a derived optimal response time.

Let `c_max=lambda_max(C)` and dimensionless candidate horizons be `a in {0,0.25,1,4}`. Set `tau=a/c_max` for each finite candidate. Infinite tau is a required ablation, not part of the default search. If comparing it within a search, count the extra evaluation and label the configuration.

Define trust budgets from the selected parameters' baseline learning increments:

$$ b_x=0.25\|d^{learn,0}_{MLP\ weights}\|_2,\qquad
b_y=0.25\|d^{learn,0}_{LN\ gamma,beta}\|_2. \tag{12} $$

Skip the event if either budget is zero. For a given shared lambda, solve all candidate horizons. Require `||x||<=b_x` and `||y||<=b_y` for every candidate. If any violates its budget, multiply lambda by 10 and recompute A/B/C, c_max, and all candidate pairs, reusing F_r and g. Allow at most six increases; otherwise skip the event. Use the same final lambda for every horizon in that event.

Do NOT radially clip the solved pair and still claim it satisfies (7)/(8): joint clipping changes the term involving g_y. Increasing shared damping and resolving preserves the specified equations.

On separate checking sequences from training data, evaluate actual full-model cross-entropy of phi and of each trial (2), with identical diagnostic dropout conditions. Choose the best trial, breaking ties toward smaller a, but accept only if its loss is below the baseline-provisional loss by at least `1e-7 + 1e-6*abs(L_check(phi))`. Otherwise commit only d0.

This chooses among four response-based damping levels using training-only checking data. It is not evidence that a physical LayerNorm response time has been identified. Validation of that interpretation is a separate diagnostic below. Adaptive horizon selection beyond this finite list is not authorized as an undocumented change.

## 8. Execution schedule

```text
compute ordinary training gradients and baseline proposal d0 once
if event not due: commit baseline
select one block using a fixed round-robin schedule
construct detached U_x/U_y from pre-update information
form provisional functional model phi = theta + d0
compute reduced fit gradient and joint PSD GGN at phi
increase shared damping until all finite-horizon proposals fit budgets
evaluate baseline and proposals on checking sequences
commit theta + d0 + the selected correction, or baseline if rejected
commit baseline state once; discard current-event curvature/bases
```

Start with one block every 32 committed steps, one short fitting sequence and one checking sequence. No multi-event inverse-action cache, no teacher training, and no simultaneous correction of every block in v1. A full-model fit gradient plus up to six directional logit derivatives and several checking forwards can be expensive even for a tiny reduced solve. Measure it.

To use a longer sequence, retain complete causal context for chosen prediction positions. Do not treat isolated tokens as equivalent probes. Flash-attention/autodiff support depends on the installed backend; the coding agent must use a verified reference attention path for mathematical tests and document any different production derivative path.

## 9. Implementation and required tests

Suggested files: `src/optimizers/ln_response.py`, `src/probes/reduced_output_ggn.py`, `src/linalg/finite_response.py`, `tests/optimizers/test_ln_response.py`, and a YAML configuration.

Pure functions should include `make_local_basis`, `reduced_ggn`, `response_operator`, `solve_response_pair`, `solve_equivalent_damped_block`, `choose_shared_damping`, and `select_trial`.

- Compare reduced derivatives and (4) against explicit full Jacobians on a tiny model. Check that all derivatives are evaluated at phi, not theta.
- Verify F_r is PSD and includes B. A test with nonzero cross coupling must fail if B is dropped.
- Compare (7) against accurately integrated constant-forcing linear dynamics for fixed x.
- Compare (8) against the direct minimizer of (10), including random nonzero g_y. Verify the sign and RHS cross term.
- Check (9) across tau near zero and very large tau. Check positivity and both limiting cases.
- Verify S_tau is SPD for random joint SPD H, and demonstrate that arbitrary independently assembled blocks need not satisfy this property.
- Verify damping retries use one common lambda across candidates and that final pairs satisfy both (8) and (12).
- Check actual checking-loss selection, no-update fallback, parameter/state restoration after rejected trials, and unchanged baseline momentum.
- Check disabled equivalence, checkpointed block schedule, skipped steps, and candidate RNG consistency.

## 10. Falsification and closest alternatives

First test whether the local model predicts actual normalization response: at a saved snapshot, hold a small matrix displacement fixed and take small, explicitly counted normalization-only gradient steps on the fitting batch. Compare resulting normalization displacement and matrix-gradient change with (7). This diagnostic is not the production optimizer, and its gradient-flow step sizes/damping must match the modeled quadratic as closely as possible. Large disagreement weakens the physical response interpretation.

Then compare baseline Muon; finite-horizon grid; tau=0; static infinite-response Schur correction; B=0; an equally sized scalar/block-damping grid; and extra normalization-only optimization with the same total compute. The direct equivalent block solve (10) is a correctness oracle, not a distinct baseline for performance claims.

Log reduced spectra, cross-block norm, lambda retries, dimensionless/physical horizon, pair residuals, predicted quadratic changes, checking and true validation loss, acceptance, extra differentiation/forward counts, and wall time. If ordinary damping search performs equally well, report that the response parameterization did not establish an independent optimizer contribution.

## 11. Sources and interpretation

- [Zwanzig, Memory Effects in Irreversible Thermodynamics (1961)](https://doi.org/10.1103/PhysRev.124.983): elimination and memory inspiration.
- [The Mori-Zwanzig formulation of deep learning (2022)](https://arxiv.org/abs/2209.05544): the framework already has deep-learning applications; it is not a blank novelty area.
- [Exact Gauss-Newton Optimization for Training Deep Neural Networks](https://arxiv.org/abs/2405.14402): GGN and small numerical solves are established ingredients.

The finite-response identity alone is not a new optimizer family. Preserve its damping equivalence in both code and reporting. The practical hypothesis is whether this particular coupling and horizon parameterization help GPT training under an honest compute budget.

---

## Shared implementation and evaluation contract

This file is self-contained. Implement this idea alone before combining it with any of the other five. The canonical equations below are normative. Pseudocode explains sequencing; it does not override an equation. Configuration values are starting points for experiments, not established optimal settings.

### Baseline adapter and parameter convention

- Inspect the repository first. Preserve unrelated code and reuse its data loader, loss normalization, model definition, and checkpoint format. Proposed paths below may be adapted to repository structure.
- Pin a working Muon implementation and its source revision. Record its momentum/Nesterov equations, shape scaling, Newton–Schulz polynomial and iteration count, precision, epsilon handling, parameter groups, and decay convention. Compare with both that strong baseline and a matched ablation of this idea. Do not quietly substitute a simplified optimizer and call it the original Muon.
- The baseline adapter computes proposals without mutating parameters: `(learning_delta, decay_delta, next_state) = propose(theta, gradients, state)`. Advance baseline state exactly once per committed training iteration. Candidate replay must never advance it again.
- Write `d0 = learning_delta + decay_delta` for the complete baseline increment. All increments are signed: the update is `theta_new = theta + d0 + correction`, unless this file explicitly replaces a learning increment. A descent gradient direction therefore has a negative increment.
- Use mathematical dense-layer orientation `Y = X W + b`, with `W` of shape `[input_width, output_width]`. Handle framework transposes and fused QKV slices through tested adapters. Preserve the pinned baseline's fused-versus-separate polar convention; views used for diagnostics must not accidentally change its grouping.
- Tied embedding parameters must have one physical optimizer state and one committed update. Non-targeted parameters use the same baseline groups, including AdamW where the pinned recipe uses it. Preserve weight decay, bias handling, LayerNorm epsilon, GELU variant, causal mask, and dropout.
- No method here constrains weights to Stiefel manifolds, imposes a spectral band or effective-rank floor, learns Muon partitions, selects Schatten powers, or distills a periodic curvature teacher into a learned correction memory.

### Numerical execution rules

1. Complete gradient accumulation and distributed synchronization before optimizer logic. Unscale mixed-precision gradients before interpreting them; apply the baseline's clipping at the same point in every run. Log that convention.
2. On a skipped/nonfinite training step, skip all optimizer-state, probe-history, and refresh-counter changes. A failed auxiliary solve on an otherwise valid step falls back to that step's baseline proposal and commits baseline state once.
3. Detach fitting statistics, probes, sample choices, solver coefficients, and optimizer state. Do not differentiate through optimizer selection or accidentally add an auxiliary loss to training.
4. Use FP64 for tiny mathematical reference tests and small ill-conditioned solve diagnostics. Use FP32 optimizer state and statistics initially; mixed-precision acceleration is a later equivalence-checked variant.
5. Use an isolated RNG stream for probes. Candidate comparisons use identical dropout randomness within their comparison set. A deterministic diagnostic mode may disable dropout consistently on all auxiliary evaluations; record that it is a surrogate for stochastic training.
6. Training-only calibration data may influence updates. Validation/test data may not. Separate fitting and checking examples where specified. Never call a check minibatch the final validation set.
7. Checkpoint all new state, RNG seeds/counters, coefficient schedules, sample identities, parameter layouts, and algorithm version. Resume must reproduce the uninterrupted next update within the stated numeric tolerance.
8. A zero-strength/disabled configuration must use a direct baseline bypass and reproduce its parameter and state trajectory. Do not rely only on an algebraically equivalent sequence of floating-point operations.
9. Implement an FP64 reference path before a scalable approximation. State every approximation in the run metadata. Do not silently replace inverses with diagonals, change probe distributions, omit cross terms, or renormalize/re-polarize a final direction.

### Common experiment protocol

Use the repository's GPT-2-style causal language model and next-token cross-entropy. Keep architecture, tokenization, sequence length, effective token batch, training stream, initialization, precision, and evaluation cadence fixed within comparisons. Start with a small model for correctness, then the intended GPT-2 configuration. Keep tied embeddings enabled for idea 3.

Primary outcomes: validation cross-entropy at fixed training-token budgets and fixed committed-update counts when token batch is identical. Also report total model-token processing, including reused calibration tokens, all forward/backward/JVP/VJP calls, wall-clock time to fixed loss targets, peak memory, throughput, and optimizer overhead. Reused tokens do not become free compute. Distinguish training loss, calibration loss, and final validation loss.

Tune the baseline and candidate with comparable declared search budgets. Use at least three matched seeds for a promising configuration; report variability and failed runs. Do not select solely on one-step calibration loss. No claim of faster training follows from fewer optimizer steps when each step performs more model evaluations.

Mandatory controls: pinned Muon recipe, AdamW, the candidate disabled, its defining mechanism ablated, and the closest practical alternative named below. Include a tuned learning-rate control whenever update magnitudes change. Stop scaling a design if its diagnostic fails or its gains disappear in the defining ablation.

### Required handoff from the coding agent

Deliver implementation, configurations, mathematical unit tests, an equation-to-code table, and a short result report. The table must map every numbered equation in this file to its implementation function and an independent check. Mark hypotheses, identities, approximations, and measured results separately. Report inability to execute GPU experiments honestly; do not invent speedups or imply these plans establish publication novelty.

References describe ingredients and nearby work. They are not evidence that the proposed combined algorithm beats Muon. The specification date is 2026-09-10; this is a design document, not a new exhaustive literature search.
