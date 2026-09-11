# Idea 3 — Joint curvature of tied embedding paths

**Algorithm ID:** `tied_path_curvature_v1`\
**Status:** Research hypothesis; the GGN identity is established mathematics.\
**Target:** The single shared input-embedding/output-head parameter of a GPT-2-style model.\
**Core requirement:** Use paired probes on both computational paths and assemble a PSD joint metric before solving. Do not implement gradient reweighting and call it this algorithm.

## 1. Scope and claim boundary

A tied embedding matrix affects logits by changing the model input and by changing the output projection. The proposed economical sketch retains interaction between these two effects. Full GGN already includes this interaction, so the identity is not a novelty claim. The research question is whether a small path-aware sketch improves the embedding update enough to justify its cost.

Use the pinned baseline's embedding learning increment, commonly AdamW in a Muon recipe. This idea changes that increment while leaving the Transformer-block optimizer intact. Do not untie the trained model, rescale input gradients, change the token loss, or create two optimizer states for the same parameter.

## 2. Exact mathematical target

Let `E: [V,d]` be the shared embedding matrix and `z(E_in,E_out): [N,V]` the logits of a function in which the two uses can be varied independently. All other weights are fixed during a probe. Evaluate the function at `E_in=E_out=E`.

$$ J_{in}={\partial\operatorname{vec}z\over\partial\operatorname{vec}E_{in}},\quad
J_{out}={\partial\operatorname{vec}z\over\partial\operatorname{vec}E_{out}},\quad
J=J_{in}+J_{out}. \tag{1} $$

For the average next-token cross-entropy over N valid prediction positions,

$$ C={1\over N}\operatorname{blockdiag}_i
\big[\operatorname{diag}(p_i)-p_i p_i^\top\big],\quad
p_i=\operatorname{softmax}(z_i). \tag{2} $$

The output-loss generalized Gauss–Newton matrix for the tied parameter is

$$ F=(J_{in}+J_{out})^\top C(J_{in}+J_{out})
=F_{in}+F_{out}+J_{in}^\top CJ_{out}+J_{out}^\top CJ_{in}. \tag{3} $$

Do not describe F as the exact nonconvex weight Hessian. It excludes second derivatives of logits weighted by the loss residual.

## 3. Paired stochastic probes with the correct covariance

The dense FP64 reference computes C and its square root directly on a tiny vocabulary. The scalable v1 uses model-sampled categorical probes, avoiding a vocabulary-square matrix.

For probe `a=1,...,k`, independently for each valid prediction position draw `c_i^(a) ~ Categorical(p_i)` and form

$$ r_i^{(a)}={e_{c_i^{(a)}}-p_i\over\sqrt N}. \tag{4} $$

Concatenate positions. Conditioned on the current model, these probes have zero mean and covariance C. Different positions and different probes must use independent categorical draws. Ignore padded/masked loss positions and count only valid N. Detach both p and r before differentiation.

Use the SAME r for both paths:

$$ u_a=J_{in}^\top r^{(a)},\qquad
v_a=J_{out}^\top r^{(a)},\qquad
z_a={u_a+v_a\over\sqrt k},\qquad Z=[z_1,\ldots,z_k]. \tag{5} $$

Then

$$ \widehat F_{joint}=ZZ^\top\succeq0,\qquad
\mathbb E[\widehat F_{joint}\mid\theta]=F. \tag{6} $$

Never replace the sampled categorical c with the observed training label. The latter produces an empirical-gradient outer product, which is generally a different matrix. Never use independent r for the two paths: zero-mean independent probes erase the intended cross terms in expectation.

The estimated metric is unbiased conditionally; its inverse and the resulting update are not generally unbiased. Do not make the stronger claim.

## 4. Correct treatment of tied aliases in autodiff

Implement a functional probe forward with two distinct differentiable leaf inputs, `E_in_probe` and `E_out_probe`, initialized to the same detached numerical E. It must use the first only for embedding lookup and the second only for the final output projection. All other trainable weights are numerical constants for this probe, but operations through the hidden network remain differentiable with respect to its embedded inputs.

For `psi = sum_i dot(r_i.detach(), z_i)`, a single reverse traversal can return gradients with respect to both leaves. These are u and v. Freezing other parameter gradients must not detach hidden activations or suppress the input path. Verify any framework functional-call alias/tie handling in the installed version; do not assume two dictionary keys create independent leaves.

This temporary split is a differentiation device only. The training model has one shared E, one state, and one commit. Ensure probes never overwrite the main training gradient or accumulate an extra gradient into it. Loss-gradient clipping does not apply to these curvature probes; their covariance definition is (4).

## 5. Canonical embedding update

Let `e0` be the vectorized baseline embedding **learning increment**, before decay. Define

$$ e_* = \arg\min_e {1\over2}\|e-e_0\|_2^2
+{\kappa\over2}e^\top\widehat F_{joint}e
=(I+\kappa ZZ^\top)^{-1}e_0. \tag{7} $$

Compute it with a k-by-k solve:

$$ e_*=e_0-\kappa Z(I+\kappa Z^\top Z)^{-1}Z^\top e_0. \tag{8} $$

Use Cholesky for the small system. Start with `rho=1` and the detached scale `kappa = rho/lambda_max(Z^T Z)` when that eigenvalue is positive. A zero metric or zero strength is the baseline bypass. Absolute kappa must be held equal in paired mechanism ablations; recomputing it separately for each metric changes two mechanisms at once.

Apply `E_new = E + reshape(e_*) + embedding_decay_delta` once. Keep the baseline first/second-moment update unchanged. Do not append the cross terms to an unrelated matrix and assume PSD. Do not use `(I + kappa*cross_only)^(-1)` in v1: the cross-only matrix can be indefinite.

Equation (7) filters the baseline increment using output-loss geometry. It is not an exact Newton update, and it need not be a descent direction for the full objective when the baseline uses momentum. It changes step magnitude as well as direction. Learning-rate and norm-matching controls are required, but do not put norm matching into the reference algorithm.

## 6. The defining cross-term ablation

Save u and v long enough to construct the comparison metric

$$ \widehat F_{separate}={1\over k}\sum_a(u_a u_a^\top+v_a v_a^\top). \tag{9} $$

Use the same current model, positions, random samples, VJP computations, baseline proposal, and absolute kappa in the paired comparison. Equation (9) has at most 2k columns instead of k, so report its small-solve cost; do not pretend computational shapes are identical. Probe-generation cost is shared and comparable.

An independent-path-probe variant is another ablation, not an implementation shortcut. The contribution is supported only if retaining paired interactions adds benefit beyond same-cost self-path curvature.

## 7. Refresh schedule and memory accounting

Reference mode rebuilds the metric every step using a tiny calibration batch from training data. Initial production starting point: `k=2` probes, refresh period `R=16`, and one short training sequence per refresh. Retain Z between refreshes in the original embedding coordinates; use the current baseline proposal each step. This is a **stale metric approximation**, not distillation or a learned teacher memory. Log metric age. A fresh-only event variant applies filtering only on refresh steps and is a required staleness control.

At each refresh, compute p and all paired probes at the same pre-update parameter snapshot. Replacing Z is atomic: if any probe fails, keep the previous metric only if its age is below a configured `max_age=R`; otherwise use baseline until the next successful refresh. Log this fallback explicitly. A skipped training step does not age the cache.

Store only Z for ordinary candidate training; retain U,V temporarily for ablation diagnostics. Dense storage is `k*V*d` FP32 numbers, plus transient probe activations and split leaves. For `V=50,257`, `d=768`, `k=2`, Z alone is about 309 MB in decimal units. Keeping both U and V doubles that factor storage. Include transient copies and optimizer state in peak-memory measurement; do not claim the method is cheap solely because the solve is k-by-k.

Do not introduce embedding-row sparsification in v1. The output path is dense over vocabulary, and retaining only input-token rows discards part of the intended interaction. Quantization or row restriction needs its own error study and algorithm ID.

## 8. Coding sequence

Suggested files: `src/probes/tied_embedding_paths.py`, `src/optimizers/tied_path_curvature.py`, `src/linalg/low_rank_prox.py`, `tests/optimizers/test_tied_path_curvature.py`, and a YAML configuration.

```text
compute the ordinary accumulated training gradient
obtain baseline proposals and next_state without committing
if refresh is due:
    run the split-leaf probe forward at current theta
    sample detached model-label probes
    compute paired u,v; build Z and its Gram
    atomically replace the cache if all results are valid
filter the current embedding learning increment using a valid cache
commit the shared embedding once and all other baseline increments
commit baseline state, cache age, probe RNG, and counters
```

Expose an explicit `split_path_forward` oracle, `sample_output_covariance_probe`, `paired_vjp`, `build_joint_sketch`, `build_separate_sketch`, and `proximal_embedding_increment`.

## 9. Mathematical tests and experimental gates

- Verify the split forward at equal leaves exactly matches the tied forward with identical dropout conditions.
- Verify the actual tied gradient equals `g_in+g_out` numerically; checking only matching logits is insufficient.
- Compute tiny J_in and J_out explicitly and verify (1)–(3), including a nonzero mixed term.
- Enumerate categorical outcomes for tiny vocabularies to verify (4)'s mean, within-token covariance, and vanishing cross-token covariance. Check average-loss normalization `1/N` versus `1/sqrt(N)` carefully.
- Check paired VJPs against explicit Jacobian products. Include a test that detaches the hidden network and must fail the input-path check.
- Verify the paired and separate metrics differ by the two cross terms for an explicit set of probes. Verify both complete metrics are PSD; show the cross-only matrix can be indefinite.
- Compare (8) with a dense solve and check stationarity and non-amplification of norm for fixed Z/kappa.
- Verify one physical update/state for tied aliases, disabled equivalence, cache refresh timing, skipped steps, and checkpoint resume.

Ablations: paired versus separate curvature; input-only and output-only curvature; independent probes; true-label empirical-Fisher probes as a named alternative; fresh-only versus cached metrics; k sweep; tuned embedding LR; and a fixed-compute baseline.

Log path-gradient norms, `||U^T V||_F`, the actual action of the mixed metric on the proposal, paired/separate update differences, Gram spectrum, cache age, probe variance across seeds, memory, and full loss curves. `U^T V` alone is not a norm of the full mixed matrix and must not be labeled as such. Reject the path-interaction hypothesis if separate self-curvature performs equally well under matched probes.

## 10. Sources and novelty limits

- [Weight Tying Biases Token Embeddings Towards the Output Space (2026)](https://arxiv.org/abs/2603.26663): gradient imbalance and input-gradient scaling already studied.
- [Exact Gauss-Newton Optimization for Training Deep Neural Networks](https://arxiv.org/abs/2405.14402): output-loss curvature and low-rank numerical solves are prior ingredients.
- [Symmetry-Compatible Principle for Optimizer Design (2026)](https://arxiv.org/abs/2605.18106): role-specific optimizer geometry is not a new claim here.

The proposed contribution must be the useful, affordable paired interaction approximation. Do not call generic sketched GGN or the expansion (3) a new discovery.

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
