# Idea 4 — Resonance-selective filtering of Muon updates

**Algorithm ID:** `proposal_notch_v1`\
**Status:** Research hypothesis using a classical digital notch filter.\
**Target:** The temporal sequence of actual matrix-update directions after the baseline polar transformation.\
**Core requirement:** Preserve unit steady-input gain, estimate resonance from unfiltered proposals, and use the precise state/reset rules below.

## 1. What this algorithm is and is not

The candidate detects persistent oscillation in matrix proposals and suppresses its narrow temporal frequency band. Ordinary momentum, frequency-based momentum, and complex-valued momentum are prior work. The possible contribution is measured resonance selection in the matrix-proposal sequence with a controlled steady-input gain.

Do not filter raw gradients and then re-polarize in the reference version. A nonlinear polar transform can undo the intended amplitude suppression. Do not use a Hessian–preconditioner commutator: that was a different, already excluded design. The original Muon momentum update remains unchanged.

## 2. Baseline signal and signed update

Have the baseline adapter expose the matrix direction `D_t` after its momentum, polar approximation, and shape scaling, but before the scalar learning rate and decoupled weight decay. In this file,

$$ d^{learn,0}_t=-\eta_t D_t. \tag{1} $$

For parameter-group learning rates, use the appropriate scalar eta. When eta is zero, obtain D directly rather than dividing an all-zero increment by eta. Parameters without a supported matrix-proposal signal use their baseline update.

The candidate direction Y_t will produce

$$ W_{t+1}=W_t-\eta_tY_t+d^{decay}_t. \tag{2} $$

Filtering the already LR-scaled increment instead of D would entangle resonance estimation with warmup and schedule changes. That is not v1.

## 3. Frozen-coefficient transfer function

Let `z^(-1)` denote one-step delay. For a selected frequency `omega` in `[pi/4, pi]` and fixed `0<r<1`,

$$ H(z)=c\,{1-2\cos(\omega)z^{-1}+z^{-2}
\over1-2r\cos(\omega)z^{-1}+r^2z^{-2}},\qquad
c={1-2r\cos(\omega)+r^2\over2-2\cos(\omega)}. \tag{3} $$

For frozen coefficients, `H(1)=1`, and `H(exp(i*omega))=0`. The poles have modulus r. The denominator in c explains why frequencies near zero are excluded. Start with `r=0.8`; tune r only in a declared search budget.

Define

$$ b_0=c,\ b_1=-2c\cos\omega,\ b_2=c,\qquad
a_1=-2r\cos\omega,\ a_2=r^2. \tag{4} $$

Use a transposed direct-form-II recurrence with two matrix-sized state buffers:

$$ Y_t=b_0D_t+s_{1,t-1}, \tag{5} $$
$$ s_{1,t}=b_1D_t-a_1Y_t+s_{2,t-1},\qquad
s_{2,t}=b_2D_t-a_2Y_t. \tag{6} $$

Compute both new states from old states and the same Y before writing either buffer. Store them in FP32 initially. The filter is elementwise in time with common coefficients per target matrix, so there is no parameter-space matrix inverse.

The frozen filter is stable as a standalone linear system. This is NOT a proof of closed-loop training stability, nor of stability under arbitrary coefficient switching. Y can have greater norm than D away from the notch. The safeguards below are part of the specified algorithm.

## 4. Initial conditions, switching, and identity behavior

On activation or any accepted coefficient change, prime the incoming states using the current input D:

$$ s_1=(1-b_0)D,\qquad s_2=(b_2-a_2)D. \tag{7} $$

Then process the current sample with (5)–(6). This makes its output Y=D and initializes the constant-input steady state. It deliberately discards earlier filter transients; record coefficient changes in logs. Zero-initializing active states would produce a different startup algorithm.

When disabled, bypass the filter exactly, set Y=D, and clear its active flag/state. Reenable only through (7). The disabled algorithm must not allocate or update large filter buffers unless needed for an enabled diagnostic mode.

Changing coefficients in-place while retaining arbitrary old states is forbidden in v1. Do not silently introduce interpolation or a state-transport formula.

## 5. Reference resonance detector

The following is a concrete v1 heuristic, not an optimal detector theorem.

For each monitored matrix, generate `K_s=8` fixed Rademacher sketch tensors R_k, scaled by `1/sqrt(number_of_elements)`. Reuse their seed and index mapping forever for that parameter. Compute

$$ h_{t,k}={\langle R_k,D_t\rangle_F\over\max(\|D_t\|_F,\epsilon_D)}. \tag{8} $$

Use `epsilon_D=1e-12` in the FP64 oracle and an appropriately logged FP32 floor. If D is exactly zero, record zero sketches. The detector reads **unfiltered D**, not Y and not the parameter trajectory. Generate sketches without retaining eight extra matrix tensors if memory matters; deterministic chunked generation is allowed.

Maintain the last `L=64` sketch samples. Every 32 committed steps, subtract each sketch's window mean, apply a Hann window `w_n=0.5*(1-cos(2*pi*n/(L-1)))`, and compute the one-sided DFT. Define

$$ P_j=\sum_{k=1}^{K_s}\left|\sum_{n=0}^{L-1}w_n(h_{n,k}-\bar h_k)
e^{-2\pi ijn/L}\right|^2. \tag{9} $$

Search bins `j=8,...,32`, corresponding to omega in `[pi/4,pi]`. Find the largest P_j, resolving ties toward smaller j. Let `P_band` be the sum over its valid neighboring bins `j-1,j,j+1`, and `P_total=sum_{j=1}^{32} P_j`.

A window is eligible only if `P_total > 1e-16`, `P_band/P_total >= 0.45`, and the peak is at least 8 times the median P over search bins (with `1e-16` added only to the denominator). The absolute floor is a starting detector threshold in normalized sketch units; log and sensitivity-test it.

Require two successive eligible windows with peak bins differing by at most one. Choose the newer peak. Do not activate before step 128. Hold active coefficients for at least 64 steps. After that dwell time, use a newly persistent eligible frequency or deactivate after two successive ineligible windows. A new frequency differing by at most one bin does not trigger a change during the dwell time. Define all counters against committed optimizer steps.

This detector tests persistence, not whether oscillation is harmful. The harm hypothesis must be evaluated through the loss experiments; do not claim the threshold proves it.

## 6. Output guard and cooldown

Before committing Y, require finite values and

$$ \|Y\|_F\le 2\|D\|_F+\epsilon_{guard},\quad
\epsilon_{guard}=10^{-12}\max(1,\|W\|_F)\text{ in reference units}. \tag{10} $$

If the guard fails, output the baseline D for this step, deactivate and clear both buffers, and impose a 128-step activation cooldown. Keep observing the unfiltered proposals during cooldown. The guard is a heuristic safety cap; it is not an exact projection and must not rescale Y silently.

A zero input can produce a filter tail. Equation (10) may reject that tail; this deliberately makes the guarded production algorithm nonlinear. Frequency-response tests must first test the unguarded frozen filter, then test the guard as a separate wrapper.

Do not re-polarize, normalize Y to D's Frobenius norm, clip individual entries, or enforce a new gradient-alignment gate in v1. Each would change the transfer behavior. Named ablations may add them.

## 7. Execution order and interfaces

```text
obtain baseline D_t, learning rates, decay increments, and next_state
update detector sketches from unfiltered D_t
at a detector event, decide activation/deactivation/coefficient change
if activating or changing coefficients, prime from current D_t
if active, compute Y and tentative new filter states
apply output guard; otherwise use baseline D and reset/cool down
commit W - eta_t*Y + decay and baseline next_state
commit detector/filter state and counters atomically
```

Suggested files: `src/optimizers/proposal_notch.py`, `src/optimizers/proposal_signals.py`, `src/probes/temporal_spectrum.py`, and `tests/optimizers/test_proposal_notch.py`.

Separate `notch_coefficients`, `prime_steady_state`, `df2t_step`, `spectral_detector`, and `guarded_filter_step`. Checkpoint s1/s2, coefficients, active flag, cooldown, window ring buffer, sample position, sketch seed, peak-history, and dwell counters. The implementation needs approximately two extra FP32 matrix buffers per active matrix; report this overhead separately from small sketches.

Begin with one matrix family, such as MLP output projections, while all other groups use baseline. Do not select a family from final validation results and then omit the selection cost. Later coverage changes are declared configurations.

## 8. Mathematical and behavioral tests

- Check H(1)=1 and H(exp(i*omega))=0, including omega=pi, for several r values.
- On a constant matrix input, priming via (7) must output that same matrix immediately and indefinitely within tolerance.
- Compare (5)–(6) against an independent direct difference-equation oracle, not another call to the same helper.
- Drive the frozen unguarded filter with sinusoids: measured steady-state amplitude/phase must match (3). Include an off-notch frequency and a slow drift, not only the zero-response target.
- Test a synthetic stable-frequency-plus-noise signal and pure white noise. The latter must not be interpreted as evidence that every high-frequency component is harmful.
- Verify coefficient changes re-prime as specified, minimum dwell/cooldown are respected, and filtered Y never enters the detector.
- Force a guard violation and confirm the exact baseline fallback and state reset.
- Verify changing eta alone does not alter D-based detector inputs when baseline D is held fixed.
- Check disabled equivalence and exact resume of active/inactive/just-switched states.

Required ablations: tuned Muon momentum; fixed notch; adaptive notch; a random-frequency notch with identical activation schedule; FSGDM or a faithful comparable frequency-momentum baseline; complex momentum where feasible; matched extra-state momentum; and norm-matched filtering as a named alternative. Keep schedules and tuning budgets visible.

Log selected frequency, band fraction, median ratio, activation/guard rates, dwell lengths, `||Y||/||D||`, `<G,Y>` and `<G,D>`, loss oscillation diagnostics, total memory, and validation-loss curves. If a fixed notch or ordinary momentum performs equally well, the online detector has not established a contribution.

## 9. Sources and novelty boundary

- [On the Performance Analysis of Momentum Method: A Frequency Domain Perspective / FSGDM](https://arxiv.org/abs/2411.19671): frequency-shaped momentum is prior work.
- [Complex Momentum for Optimization in Games](https://arxiv.org/abs/2102.08431): complex poles and oscillatory optimization are prior work.
- [The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137): multiple temporal momentum scales are prior work.

The proposed addition is specifically resonance identification and narrow suppression after the matrix proposal transform. A notch filter by itself is classical signal processing. No convergence or advantage over Muon is established by its transfer function alone.

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
