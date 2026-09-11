# Idea 2 — Sparse attention-curvature preconditioning

**Algorithm ID:** `routing_resistance_v1`\
**Status:** Research hypothesis with an exact categorical-graph sampling rule.\
**Target:** Filter the baseline learning increment of a selected query/key head using a sampled routing metric.\
**Core requirement:** Preserve the probability distribution, edge weights, joint Q/K Jacobian, and PSD solve below.

## 1. Scope and distinction

Use the geometry of attention probabilities to penalize an update that changes routing strongly. This is an internal routing metric, not the full language-model Fisher or loss Hessian. Dense causal attention and the training objective stay unchanged.

The reference algorithm is an explicitly defined proximal filter of Muon's learning increment. It is not a claim that sparse Fisher estimation or graph sparsification is new. It differs from idea 1 because it uses a first-order metric at the current model, not a finite product-defect repair. It does not constrain the weights or split the Muon polar transform into newly learned blocks.

## 2. Exact softmax graph

For a valid causal attention row with `n` keys and positive probabilities `p`,

$$ L(p)=\operatorname{diag}(p)-pp^\top
=\sum_{j<k}w_{jk}b_{jk}b_{jk}^\top,
\quad w_{jk}=p_jp_k,\quad b_{jk}=e_j-e_k. \tag{1} $$

For this particular complete weighted graph, effective resistance and edge leverage are

$$ R_{jk}=b_{jk}^\top L^+b_{jk}=1/p_j+1/p_k,
\qquad w_{jk}R_{jk}=p_j+p_k. \tag{2} $$

The sum of these leverages is `n-1`. Therefore the exact resistance-based unordered-edge distribution is

$$ q_{jk}^{R}={p_j+p_k\over n-1},\qquad j<k. \tag{3} $$

An efficient sampler for (3): draw `j ~ Categorical(p)`; draw `k` uniformly from the other `n-1` keys; sort the pair. Both ordered paths contribute to the unordered probability. Do not use `p_j p_k` as this sampling probability.

Canonical robustness mixture:

$$ q_{jk}=(1-\epsilon_q)q_{jk}^{R}
+\epsilon_q{2\over n(n-1)},\qquad \epsilon_q=0.05. \tag{4} $$

Sample from the corresponding mixture, with replacement, and retain duplicates with their own weights. The uniform branch draws an unordered pair uniformly. Each row's distribution is separate. Skip `n=1` rows. In numerics, compute softmax in at least FP32, use the actually computed normalized probabilities, and do not evaluate `1/p` to sample. Equation (2) assumes positive probabilities; (4) and the estimator below still handle represented zero-weight edges without division by zero in `p`.

## 3. Pull each edge back to weight space

For query row `i`, `q_i = x_i W_Q + b_Q`, keys `k_j = x_j W_K + b_K`, and logit `s_ij=q_i dot k_j/sqrt(d_h)`, an edge logit difference has gradients

$$ a_{ijk,Q}={x_i^\top(k_j-k_k)\over\sqrt{d_h}},\qquad
a_{ijk,K}={(x_j-x_k)^\top q_i\over\sqrt{d_h}}. \tag{5} $$

In (5), `x`/`q`/`k` are row vectors. Both weight gradients have shape `[d_in,d_h]`. Concatenate their vectorizations into `a_e`. Biases are held fixed for this metric in v1. Including existing Q/K biases in q and k is mandatory, but preconditioning bias increments is a separate extension.

The exact selected-row routing metric is

$$ F_{route}={1\over N_q}\sum_i\sum_{j<k}p_{ij}p_{ik}
a_{ijk}a_{ijk}^\top. \tag{6} $$

With `m` independent edges from each selected row, construct columns

$$ z_{ie}=\sqrt{{p_{ij}p_{ik}\over N_q m q_{jk}}}\,a_{ijk},
\qquad Z=[z_1,\ldots,z_M],\quad \widehat F=ZZ^\top. \tag{7} $$

Conditional on the selected rows and current model, `E[F_hat] = F_route`. If query rows themselves are sampled uniformly from all valid rows, their mean is also an unbiased estimator of the corresponding row-averaged full metric. Nonuniform query sampling requires a further specified importance weight; do not silently add it.

The inverse-filtered update is nonlinear in `F_hat`, so it is NOT an unbiased estimator of the exact inverse-filtered update. Distinguish those claims in the report.

## 4. Canonical update: a PSD proximal filter

Let `v0` concatenate the baseline **learning increments** for the selected Q/K weights. Do not include weight decay in `v0`. Solve

$$ v_* = \arg\min_v {1\over2}\|v-v_0\|_2^2
+{\kappa\over2}v^\top\widehat Fv
=(I+\kappa ZZ^\top)^{-1}v_0. \tag{8} $$

Use the small Gram matrix `K=Z^T Z`:

$$ v_*=v_0-\kappa Z(I+\kappa K)^{-1}Z^\top v_0. \tag{9} $$

The canonical strength convention is `kappa = rho/lambda_max(K)` for a nonzero metric, starting with `rho=1`. If the metric is zero or `rho=0`, use the baseline bypass. The normalization is detached. A fixed absolute kappa is an explicitly named ablation.

This choice bounds the linear filter's eigenvalues between `1/(1+rho)` and `1`. It cannot amplify Euclidean update norm for a fixed metric. It does not guarantee a better loss decrease. In comparisons that isolate paired versus separate Q/K terms or sampling rules, reuse the same absolute kappa for a paired instance; independently normalizing each ablation can confound the mechanism.

Reshape `v_*` back to Q/K weight increments and apply their original decay increments separately. Other parameters and baseline optimizer-state updates remain unchanged. Do not re-polarize or renormalize `v_*` in v1. Frobenius-norm matching may be tested only as a labeled variant, alongside learning-rate tuning.

## 5. Factored implementation without a parameter-size matrix

Each `a_e` consists of two rank-one matrices. Store its input/key/query factors and scalar importance weight. Compute `Z^T v0`, weighted sums `Z c`, and the Gram matrix through these factors. For two edges `e,f`,

$$ \langle a_e,a_f\rangle=
{\langle x_{i_e},x_{i_f}\rangle
\langle k_{j_e}-k_{k_e},k_{j_f}-k_{k_f}\rangle
+\langle x_{j_e}-x_{k_e},x_{j_f}-x_{k_f}\rangle
\langle q_{i_e},q_{i_f}\rangle\over d_h}. \tag{10} $$

Include the column square-root weights from (7) when forming K. Cross-sequence edge pairs may contribute to the parameter-space Gram; token indices must never cross sequences when constructing an individual attention edge.

Solve `I+kappa K` by Cholesky, initially in FP64. Do not materialize a `[2*d_in*d_h]^2` matrix in production. The dense reference must materialize it for tiny shapes to verify the factored path.

Sampling M edges does not imply a full spectral approximation guarantee for arbitrary small M. The classical graph result is not automatically a guarantee for this fixed-budget, changing neural-network metric. Report approximation errors empirically and restrict claims accordingly.

## 6. Execution schedule and data access

```text
compute synchronized training gradients
obtain baseline learning/decay proposals and next_state
for a selected head:
    collect current X, Q, K and selected valid attention probability rows
    sample weighted unordered edges using (4)
    construct implicit Z and exact small Gram K
    compute kappa; solve (9)
    replace that head's Q/K learning increments
commit all increments plus the baseline decay increments
commit baseline state once
```

Reference mode uses every valid row of a tiny head and enumerates all edges. Initial scalable mode: `N_q=4` selected query rows, `m=4` edges per row, one head per event, rotating deterministically. Compute the selected rows from retained q/k or a small auxiliary replay; do not force every layer to expose its full attention matrix. Count any replay cost.

Recompute sampled factors on every event; no persistent curvature cache in v1. An event period greater than one means other steps use the ordinary baseline, not stale inverses. Record which heads and steps were modified. Extend coverage only after overhead and mechanism tests pass.

## 7. Coding tasks

Suggested files: `src/optimizers/routing_resistance.py`, `src/probes/attention_rows.py`, `src/linalg/rank_one_pair_gram.py`, `tests/optimizers/test_routing_resistance.py`, and one YAML configuration.

Required functions: `sample_unordered_edges`, `edge_sampling_probability`, `edge_weight_grad_factors`, `weighted_factor_gram`, `apply_route_metric`, `filter_proposal`, and a tiny explicit-Jacobian oracle. Store per-head RNG counters and the head schedule in checkpoints. Curvature construction must not populate model `.grad` buffers.

## 8. Tests and decisive ablations

- Verify (1) by explicit edge enumeration, including highly unequal p.
- Verify (2) against a dense pseudoinverse for strictly positive p. This catches the common missing `(n-1)` normalization in (3).
- Enumerate all unordered pair probabilities and confirm their sum is one. Verify both sampler branches and duplicates. Use deterministic exact probability checks plus one statistically meaningful sampling check.
- Compare (5) against autodiff of a scalar logit difference, including nonzero Q/K biases.
- Verify the expectation of (7) by exact summation over the sampling distribution on tiny inputs. Do not rely solely on one Monte Carlo realization.
- Compare factored Gram (10) and Woodbury result (9) with dense matrices. Verify PSD, stationarity of (8), zero-metric behavior, and non-amplification of proposal norm.
- Verify independent constant shifts of valid row logits do not change routing geometry.
- A test that deliberately drops the joint Q/K cross block must produce a different metric in a coupled example.
- Check disabled equivalence, checkpoint resume, causal masking, and fused tensor adapters.

Required ablations: uniform edges with identical importance correction and edge count; exact metric on tiny models; separate Q-only/K-only metrics; activation-covariance preconditioning; unfiltered Muon with tuned learning rate; norm-matched filtering; and increasing M at fixed total experiment compute.

Log sampling probabilities/weight extrema, effective edge count, Gram spectrum, metric trace, kappa, solve residual, routing quadratic before/after, update norm, alignment with the training gradient, and actual loss/compute outcomes. If resistance sampling adds no benefit over uniform sampling, remove the resistance-specific claim. If gains are explained only by a smaller effective step, remove the geometry claim.

## 9. Sources and interpretation

- [Spielman and Srivastava, Graph Sparsification by Effective Resistances](https://arxiv.org/abs/0803.0929): graph-sampling ingredient.
- [Exact Gauss-Newton Optimization for Training Deep Neural Networks](https://arxiv.org/abs/2405.14402): related curvature and low-rank-solve techniques; the routing metric here is different from full output-loss GGN.
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982): baseline context.

The exact categorical resistance formula makes the initial sampler executable without a large inverse. The proposed research contribution still requires evidence that this internal metric and its sparse approximation improve GPT training.

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
