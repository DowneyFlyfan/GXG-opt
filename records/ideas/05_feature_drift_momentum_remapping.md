# Idea 5 — Momentum remapping through measured feature drift

**Algorithm ID:** `feature_remap_v1` (every-step reference); `feature_remap_cohort_v1` (periodic approximation).\
**Status:** Research hypothesis with a prediction-first experimental gate.\
**Target:** Change the historical momentum estimate using measured evolution of dense-layer gradient factors.\
**Core requirement:** Fit maps on matched examples and transform only the age cohort to which a measured map applies.

## 1. Distinction from previous designs

This is a prediction of a changed gradient in fixed parameter coordinates. The actual model weights are not reparameterized or whitened. It is not a symmetry/gauge transformation and does not preserve the represented momentum by merely changing its storage coordinates.

Your earlier optimizer-state design allocated compressed momentum ranks by prediction utility. Here, momentum has a fixed storage policy; a learned map deliberately changes the full represented matrix. Fixed channel blocks below restrict the map fit only. They do not partition the Muon polar transform or create learned optimizer groups.

Do not implement this optimizer before checking whether the maps predict gradient evolution better than leaving the old gradient unchanged.

## 2. Gradient factorization and its limits

For a dense layer `Y=XW+b`, use `X: [N,d_in]` and `E_Y=partial L/partial Y: [N,d_out]`. Then

$$ G_W=X^\top E_Y. \tag{1} $$

If the loss already averages valid tokens, E_Y already contains that normalization. Do not divide (1) by N again. When probing a subset of sequences, calculate its own consistently normalized calibration loss and capture both factors from that same pass.

For identical anchor tokens at two parameter snapshots s and t, fit

$$ X_t\approx X_s A_{s\to t},\qquad
E_t\approx E_s B_{s\to t}. \tag{2} $$

When both relationships are exact,

$$ G_t=A_{s\to t}^\top G_s B_{s\to t}. \tag{3} $$

Equation (3) is an exact algebraic identity under (2). Applying it to momentum containing gradients from other batches or older snapshots is an approximation. Do not call it an exact covariant transformation of the optimizer.

## 3. Anchor data and map fit

Use a small fixed bank of training sequences, with separate fit and check sequences. Reuse the same token IDs, positions, causal context, padding, loss positions, and auxiliary dropout setting between snapshots. Maps fitted between unrelated minibatches are invalid. Keep anchors fixed for the initial experiment. If rotating anchors later, reset the comparison snapshot and perform no remap across that change.

Capture current X and E through an auxiliary forward/backward pass. Retain gradients for layer outputs while preventing accumulation into main training `.grad` buffers. Do not detach the output from the downstream loss. Normalization and dropout conventions must match between old and current probes.

The dense reference solves ridge problems close to identity:

$$ A_{raw}=\arg\min_A\|X_sA-X_t\|_F^2+\lambda_X\|A-I\|_F^2
=(X_s^\top X_s+\lambda_X I)^{-1}(X_s^\top X_t+\lambda_X I), \tag{4} $$

$$ B_{raw}=(E_s^\top E_s+\lambda_E I)^{-1}(E_s^\top E_t+\lambda_E I). \tag{5} $$

Use linear solves, not explicit matrix inversion. A zero-energy old factor produces the identity map and a low-confidence diagnostic. Starting ridge rules: `lambda_X=0.1*trace(X_s^T X_s)/d_in`, likewise for E. If a trace is zero, branch to identity; do not fabricate information with an arbitrary enormous inverse.

Initial scalable approximation: fixed contiguous channel blocks of size 32, with a smaller final block if necessary. Fit (4)–(5) independently within each block, making A and B block diagonal. Use the same blocks throughout a run and checkpoint their index order. This omits cross-block representation changes and is not invariant to arbitrary channel rotations; state that limitation.

Bound each block's departure from identity by computing an SVD of `A_raw-I` and clipping its singular values to `delta_map=0.1`; then add I back. Do the same for B. Denote the resulting maps A and B. They are bounded near-identity maps, not necessarily orthogonal matrices. Do not polar-project them. The clipped maps no longer solve the unconstrained ridge equations; tests must distinguish raw-fit stationarity from post-clipping bounds.

Use all fitting rows to estimate maps; do not center factors silently. Adding a fitted intercept or transporting dense-layer bias momentum is outside v1.

## 4. Independent predictive acceptance

On checking sequences, calculate `G_s^check = (X_s^check)^T E_s^check` and its current counterpart. Define

$$ e_{raw}=\|G_t^{check}-G_s^{check}\|_F^2,\qquad
e_{map}=\|G_t^{check}-A^\top G_s^{check}B\|_F^2. \tag{6} $$

Accept the map only if all entries and solves are finite, map bounds hold, and

$$ e_{map}\le 0.95\,e_{raw},\qquad
e_{raw}>10^{-12}\max(\|G_t^{check}\|_F^2,10^{-12}). \tag{7} $$

Otherwise replace both maps by identity for the update. Fit/check factor residuals and relative errors must be logged even when the map is rejected. These are starting confidence thresholds; sensitivity studies must declare changes. The check bank is training-only calibration data and can itself be overfit over long runs, so periodically audit on a third fresh diagnostic batch that does not control maps.

Do not accept A and B independently based only on their separate factor errors. The defining predictor is their combined gradient action (6). Activation-only A with B=I is an important named ablation.

## 5. Every-step reference momentum

Write baseline EMA momentum as `M_t = beta*M_(t-1)+(1-beta)*G_t`. For this convention, replace it by

$$ M_t=\beta A_t^\top M_{t-1}B_t+(1-\beta)G_t, \tag{8} $$

where A_t and B_t are fitted from snapshot t-1 to t. Then pass M_t and G_t through the pinned baseline's remaining direction transformation, including its declared Nesterov expression if applicable, polar approximation, and shape scaling. Do not average (8) a second time inside the baseline adapter.

If the repository uses unnormalized momentum `M=beta*M+G`, either implement the identical remapping with that gradient coefficient or convert both candidate and matched baseline explicitly. Do not switch conventions in only one run. Nesterov on/off is held fixed within a comparison; remapping changes only the historical first-moment input.

The reference samples anchor features every committed optimizer step, so it can be expensive. It is the oracle for the periodic approximation. Checkpoints contain the prior anchor factors and the snapshot time associated with them, not just M.

## 6. Periodic maps require age-separated momentum

**Forbidden shortcut:** Fit A from t-R to t, then apply it to the entire current M. M contains gradients computed after t-R, so this transforms recent information using a map for the wrong time interval.

Use two full momentum buffers H and F. At the most recent refresh s, H stores `M_s`, F is zero, and the saved anchor factors are from the pre-update model snapshot theta_s. H is the historical cohort; F accumulates younger gradients.

For each subsequent committed step t, first compute

$$ H^-_t=\beta H_{t-1},\qquad
F^+_t=\beta F_{t-1}+(1-\beta)G_t. \tag{9} $$

On an ordinary non-refresh step, set `M_t=H^-_t+F^+_t` and retain those two components. At a refresh, fit A and B from the saved snapshot s to the current pre-update snapshot t, apply (7), and set

$$ M_t=A_{s\to t}^\top H^-_t B_{s\to t}+F^+_t. \tag{10} $$

Then use M_t to form this step's matrix direction. On successful step commit, reset `H_t=M_t`, `F_t=0`, and save current anchor factors and time t. If a valid probe produced a rejected map, still advance the anchor snapshot and reset cohorts with the identity-mapped M_t. If the probe itself failed, use no remap and retain the old anchor/cohort boundary; retry at the next scheduled event, subject to a recorded maximum snapshot age. On exceeding that age, clear the old anchor and start a new identity refresh rather than fitting against unusably old data.

Equation (10) transforms only the portion that predates the old snapshot. Its younger cohort is deliberately left unremapped. This remains an approximation; it is not identical to applying fresh one-step maps to every historical contribution. With R=1 and valid probes, F contains only the current gradient and (10) reduces to (8).

Initial production schedule: R=8, maximum snapshot age 32, one fixed matrix family such as MLP output projections. If beta=0, the historical cohort vanishes and the method must reduce to the baseline. Do not transport Adam second moments or weight decay in v1.

## 7. Exact processing order

```text
compute accumulated training gradient at theta_t
before changing weights, collect anchors if refresh is due
fit/check maps using the saved matching snapshot
advance H and F with the baseline momentum coefficients exactly once
on refresh, remap H only, then form M
apply baseline Nesterov/polar/shape-scaling pipeline using this M
commit weights and all state once
if a valid refresh completed, reset cohorts and save current anchors
```

The coding agent must provide an adapter boundary that accepts externally supplied first momentum. A baseline optimizer whose internal `step()` always updates momentum again is not a valid adapter. Maintain single-state identity for fused tensor views and tied parameters; embeddings are not targeted in the initial experiment.

Suggested files: `src/optimizers/feature_remap.py`, `src/probes/gradient_factors.py`, `src/linalg/near_identity_maps.py`, and `tests/optimizers/test_feature_remap.py`.

## 8. State, cost, and forbidden approximations

Save H/F, accepted map statistics, fixed channel layout, saved fit/check factors, anchor token IDs, normalization, dropout mode/seed, refresh age, and failed-probe counters. Between refreshes A/B need not remain resident once their action is committed, but record diagnostics needed to reproduce selection.

Costs include an auxiliary model forward/backward at every refresh, storage for old/current factors, two momentum cohorts, small block solves, and left/right multiplication of a full momentum matrix by block-diagonal maps. Do not describe this as a few scalar updates. Compare every-step R=1 to periodic R=8 only after the exact reference passes.

Forbidden: fitting maps between different examples; transporting newly accumulated F with an old-interval map; silently replacing B with identity; changing parameter values to force representation agreement; compressed-state basis rotation labeled as prediction; or momentum smoothing applied twice.

## 9. Tests and go/no-go experiment

- Verify (1) against the layer's weight gradient, including mean-loss normalization, padding, and nonzero bias.
- For synthetic exact X_t=X_s A and E_t=E_s B, verify (3) to FP64 tolerance.
- Check ridge stationarity of raw (4)/(5), block shapes, and singular-value bounds of clipped map differences separately. Include zero/rank-deficient factor cases.
- Construct a map that fits training anchors but fails held-out gradient prediction and ensure (7) rejects it.
- Expand H and F into an explicit weighted list of past gradients on a tiny example. Verify (9)/(10), map only the correct cohort, and prove R=1 reduction numerically.
- Construct a non-identity multi-step map where applying it to the entire M produces a different wrong result; this is a required regression test.
- Verify beta=0 and disabled bypass, no double momentum update, matched Nesterov transformation, and checkpoint resume around a refresh.

Before optimizer training, run a baseline trajectory and collect matched anchor snapshots. Measure the held-out gradient-prediction error of raw previous gradients, activation-only maps, error-only maps, both maps, and simple decay/scalar rescaling. This experiment can falsify the entire idea cheaply relative to a full optimizer sweep.

If prediction improves, compare optimizer variants at matched token and compute budgets: baseline momentum, scalar momentum decay/restart, R=1 maps, periodic cohort maps, and an intentionally naive full-buffer periodic remap as a diagnostic only. Log prediction gains versus actual future loss gains, acceptance rate, map spectra, cohort norms, and extra cost. A map that predicts anchor gradients but worsens training is a negative result, not a reason to silently redefine the acceptance rule.

## 10. Sources and novelty limits

- [Natural Neural Networks / PRONG (2015)](https://arxiv.org/abs/1507.00210): representation conditioning and function-preserving reparameterizations already exist; this proposal instead predicts a changed gradient in fixed weight coordinates.
- [Arbitrary-Lagrangian-Eulerian DG schemes on moving meshes](https://arxiv.org/abs/1612.04068): numerical inspiration for carrying information when a representation evolves; it is not a neural optimizer precedent.
- [Towards understanding how momentum improves generalization](https://arxiv.org/abs/2207.05931): historical gradients can carry useful feature information, so discarding or transforming them is not automatically beneficial.

The near-identity map fit, held-out predictor gate, and cohort bookkeeping concretize the previous high-level hypothesis. They are not established claims of universal gradient transport or Muon improvement.

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
