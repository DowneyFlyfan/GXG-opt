# Idea 1 — Query–key finite-interaction correction

**Algorithm ID:** `qk_defect_v1`\
**Status:** Research hypothesis; an executable reference design is specified here.\
**Target:** Correct selected attention-head weight increments after the pinned Muon proposal.\
**Core requirement:** Explicitly compute the finite product interaction and repair it in weight space; do not implement only a query/key learning-rate multiplier.

## 1. What must stay consistent with the idea

The gradient already contains the first-order influence of queries and keys. This proposal addresses the additional interaction caused by changing both during a finite step. The identity below is not new. The candidate contribution is a cheap, useful correction after a matrix optimizer proposal.

Do not cancel the interaction unconditionally: it can lower loss. Construct a defect-repair candidate, then check whether its actual attention output is preferable to the original proposal under an explicitly defined local surrogate. Include ordinary scale tuning as a control.

This design imposes no constraints on the singular values or rank of the weights. It does not create learned Muon blocks. It does not claim a better first-order solution than exact polar steepest descent under the same operator-norm ball.

## 2. Exact finite-step identity

For one head, let `X: [T,d_in]`, `W_Q,W_K: [d_in,d_h]`, and frozen projected activations `Q = X W_Q + b_Q`, `K = X W_K + b_K`. Process different sequences separately. The attention logits on valid causal entries are

$$ S=QK^\top/\sqrt{d_h}. \tag{1} $$

The baseline adapter supplies complete weight/bias increments, including their decay contribution where present. Define

$$ \Delta Q=X\Delta W_Q+\Delta b_Q,\qquad
\Delta K=X\Delta W_K+\Delta b_K. \tag{2} $$

Then, with the block input held fixed,

$$ S_b-S=L+C_d,\quad
L=(\Delta QK^\top+Q\Delta K^\top)/\sqrt{d_h},\quad
C_d=\Delta Q\Delta K^\top/\sqrt{d_h}. \tag{3} $$

`S_b` is the exact logits after the baseline Q/K increment at this fixed input. Never subtract masked `-inf` logits. Compute finite logits, select valid entries, and apply the causal mask only at softmax. `C_d` is not the full loss Hessian or the full model's nonlinear remainder.

## 3. Build a realizable correction space

Corrections target weights only in v1; baseline bias increments remain as proposed. Let `R=(R_Q,R_K)` be a pair of weight corrections. Around `Q_b=Q+DeltaQ`, `K_b=K+DeltaK`, their first-order effect is

$$ \mathcal A(R)=\big[(XR_Q)K_b^\top+Q_b(XR_K)^\top\big]/\sqrt{d_h}. \tag{4} $$

The exact extra logit change also contains `(X R_Q)(X R_K)^T/sqrt(d_h)`. Do not omit it when replaying a candidate.

On fitting rows, define the PSD seminorm

$$ \|Z\|_{\Omega}^2={1\over N_q}\sum_i
Z_i^\top[\operatorname{diag}(p_{b,i})-p_{b,i}p_{b,i}^\top]Z_i,
\quad p_{b,i}=\operatorname{softmax}(S_{b,i}). \tag{5} $$

Probabilities and weights are frozen and detached. The seminorm ignores constant row shifts, which do not change softmax. Apply each row's actual causal support. Skip rows with only one valid key because their routing metric is zero.

Create at most four candidate basis pairs in concatenated weight space:

1. `(DeltaW_Q, 0)` and `(0, DeltaW_K)`.
2. The Q-only and K-only components of `-A_star(Omega C_d)`, using the adjoint of (4).

Here `Omega` includes the average over selected rows. Compute this adjoint analytically or by autodiff of `0.5*||A(R)+C_d||_Omega^2` at `R=0`, differentiating only with respect to correction variables. Do not differentiate through `Q_b`, `K_b`, `p_b`, or the baseline proposal.

Orthogonalize the nonzero pairs in the ordinary concatenated Frobenius inner product, using deterministic modified Gram–Schmidt with reorthogonalization in the reference implementation. Drop a vector when its residual norm is at most `1e-10` times its original norm in FP64. The resulting pairs `B_j` are orthonormal. Record the actual rank `r <= 4`.

The defect-gradient vectors are essential: without them the entire method could reduce to two scale multipliers. Basis construction must use fitting examples only.

## 4. Reduced defect solve and bounded correction

Set `T_j=A(B_j)` and solve

$$ a_* = \arg\min_a {1\over2}\left\|C_d+\sum_{j=1}^r a_jT_j\right\|_\Omega^2
+{\lambda\over2}\|a\|_2^2. \tag{6} $$

Equivalently,

$$ (H+\lambda I)a_*=-b,\qquad
H_{ij}=\langle T_i,T_j\rangle_\Omega,\quad
b_i=\langle T_i,C_d\rangle_\Omega. \tag{7} $$

Use Cholesky in FP64 for this tiny system. Canonical scale rule: `lambda = lambda_rel * trace(H)/r`, starting with `lambda_rel=0.01`. If the trace or basis rank is zero, return no correction; do not divide by a zero scale. Treat materially negative eigenvalues as an implementation error. Numerical failure causes a logged baseline fallback.

Let `R_* = sum_j a_j B_j`. Bound its Frobenius size relative to the baseline Q/K weight increments:

$$ R=\alpha R_*,\quad \alpha=\min\left(1,
{\rho\sqrt{\|\Delta W_Q\|_F^2+\|\Delta W_K\|_F^2}
\over \|R_*\|_F}\right),\qquad \rho=0.25\text{ initially}. \tag{8} $$

Use the explicit zero-norm branch. This radial cap is a specified safeguard, not the exact solution of a trust-region version of (6). Do not describe it as that solution. The ridge optimum and the capped proposal are different objects.

## 5. Local loss surrogate and acceptance

Use separate fitting and checking sequences from training data, with fixed inputs captured at the current model. For a chosen head, define its contribution to the attention sublayer output, including its slice of the output projection:

$$ Y=\operatorname{softmax}(S)V W_{O,h}. \tag{9} $$

Freeze the other heads' contributions. For candidate evaluation, include the baseline updates to this head's V and output-projection parameters identically in every candidate; the only differences are Q/K corrections. Capture `E_Y = partial L / partial Y` at the original current model. It must use the same token-averaged calibration loss normalization as all surrogate terms.

For each split define

$$ q(Y')=\langle E_Y,Y'-Y\rangle+{\nu\over2}\|Y'-Y\|_F^2. \tag{10} $$

`nu` is a nonnegative output-space damping coefficient. Canonical heuristic: `nu = nu_rel * ||E_Y||_F / max(||Y_b-Y||_F, eps_out)`, with `nu_rel=1` initially and `eps_out = 1e-8 * max(||Y||_F,1)` in the reference units. Estimate this scalar on the fitting split and reuse it on the checking split. Explain that this is a surrogate-damping heuristic, not known downstream curvature.

Evaluate `gamma in {0, 0.25, 0.5, 1}` with actual softmax outputs from weights `W+d0+gamma*R`; include the correction–correction product exactly. Compare every candidate against `gamma=0`. Choose the largest decrease in checking surrogate, with ties resolved toward smaller `gamma`. Accept only when the improvement exceeds `atol + rtol*abs(q(Y_b))`, initially `atol=1e-10`, `rtol=1e-4` in FP64 diagnostics. Otherwise use zero correction. Lock this rule before benchmark comparisons.

This acceptance is a statement about (10) on checking data, not a proof of full-model or population-loss descent. The unchanged-X approximation excludes incoming changes from earlier blocks. Record this limitation in results.

## 6. Exact update order

```text
compute accumulated training gradients at theta
obtain baseline d0 and next_state once, without committing
if not a correction event: commit baseline and finish
collect fit/check activations and output adjoints at theta
for the selected head only:
    form DeltaQ, DeltaK, C_d from the complete baseline proposal
    construct basis and solve the reduced ridge problem on fit data
    cap correction norm
    choose gamma with exact local attention replay on check data
commit theta_new = theta + d0 + selected Q/K weight correction
commit baseline next_state exactly once
```

Do not feed the correction into the baseline momentum in v1. Updating that momentum with a synthetic gradient would define another algorithm.

## 7. Implementation stages and interfaces

Suggested files: `src/optimizers/qk_defect.py`, `src/optimizers/proposal_adapter.py`, `src/probes/attention_replay.py`, `tests/optimizers/test_qk_defect_math.py`, and `configs/optimizers/qk_defect.yaml`.

Implement `collect_head_probe`, `finite_qk_defect`, `apply_correction_jacobian`, `build_basis`, `solve_reduced_defect`, and `select_local_candidate` as pure functions where possible. A fused QKV tensor must be updated through disjoint tested slices; never apply a correction twice through an alias.

Stage A: tiny FP64 attention head and dense reference Jacobian. Stage B: one selected head in a small GPT model. Stage C: a fixed round-robin head schedule, initially one head every eight optimizer steps. Fitting and checking probes can begin with one short sequence each and 8 selected query rows per head, retaining all valid keys for those rows. Query-row selection does not authorize truncating their causal prefixes. Avoid materializing attention matrices for every layer.

Sampling and refresh periods are explicitly approximate production choices. Save the chosen schedule and selected heads. Extrapolating cached attention statistics to later steps is not allowed in v1: recompute them at each correction event.

## 8. Mathematical tests and rejection experiments

- Check (3) against direct matrix multiplication to relative error below `1e-10` in well-scaled FP64 examples, including nonzero biases.
- Check (4) and its adjoint by finite differences and the inner-product identity. Include rectangular matrices, masks, and head slices.
- Verify constant logit row shifts have zero seminorm and leave softmax unchanged.
- Compare (7) with a direct least-squares construction. Check stationarity before the cap; check (8) afterward. Verify basis rotation does not change the unconstrained ridge result.
- Check that exact replay includes the quadratic correction product. A deliberately large correction should expose an implementation that evaluates only a linearized logit change.
- Construct synthetic cases where the baseline product term helps loss and ensure the acceptance mechanism can reject cancellation.
- Force solve failure and a no-improvement check; both must leave the baseline trajectory intact except diagnostic counters.
- Resume, disabled-equivalence, fused-layout, and mixed-precision checks follow the common contract.

Required ablations: base Muon; Q/K scale tuning with the same local replay budget; scale-only basis; full basis with `C_d=0` (which must yield zero defect correction); capped versus uncapped diagnostic proposals; exact versus linearized candidate replay; and extra training using the same compute budget.

Log `||C_d||_Omega`, `||L||_Omega`, defect-gradient alignment, basis rank, solve condition, uncapped/capped norms, selected gamma, fit/check surrogate changes, true full-model loss checks at a sparse diagnostic cadence, and all extra compute. Reject the mechanism if its benefits are explained by simple Q/K scale tuning or do not transfer beyond calibration examples.

## 9. Sources and novelty boundary

- [Dutt, Greengard, and Rokhlin, Spectral Deferred Correction Methods (2000)](https://doi.org/10.1023/A:1022338906936): inspiration for explicit defect repair; not this optimizer.
- [A Differential Game Theoretic Neural Optimizer for Training Residual Networks (2020)](https://arxiv.org/abs/2007.08880): architecture-aware optimization already exists.
- [Faster Query-Key Learning Sharpens Attention (2026)](https://arxiv.org/abs/2608.06776): relative circuit learning speeds are prior work; simple learning-rate differentiation is not the claimed novelty.
- [Muon is Scalable for LLM Training (2025)](https://arxiv.org/abs/2502.16982): baseline context.

The low-dimensional basis, ridge rule, norm cap, and surrogate above resolve previously open implementation choices. Treat them as the v1 hypothesis, not as empirically established optimal choices.

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
