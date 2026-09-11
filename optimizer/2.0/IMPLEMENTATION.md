# Optimizer 2.0 implementation and local experiment

The six design documents in this directory define independent research
hypotheses. Their identities and reduced mathematical models are implemented;
bounded implementation checks do not establish training superiority. The default
experiment is eight methods, one seed, five complete WikiText-103 epochs from
random GPT-2 initialization. No full hyperparameter search or proposal-specific
research ablation campaign is included in that eight-method screen.

## Baseline and execution boundary

`src/optimizer_v2/adapter.py` pins the installed PyTorch 2.11 implementation by version,
Git revision and source-file SHA-256 in the comparison manifest. The executable
source, rather than an alternative momentum convention in prose documentation,
defines the recipe:

- FP32 EMA: `M.lerp_(G, 1-beta)`, with `beta=0.95`.
- Nesterov signal: `G.lerp(M, beta)`.
- Five BF16 quintic Newton-Schulz steps with coefficients
  `(3.4445, -4.775, 2.0315)` and norm floor `1e-7`.
- Direction scaling `0.2 * sqrt(max(rows, columns))`; decay uses the original,
  unadjusted learning rate.
- Full fused GPT-2 hidden matrices use Muon. Embeddings and vectors use AdamW;
  tied word embeddings have one physical state. Bias/normalization decay is zero.
- AdamW uses `(0.9, 0.95)`, epsilon `1e-8`, and the stock single-tensor path.

`ProposalAdapter.propose()` returns learning/decay increments, the direction
before scalar LR, the complete proposed parameter value and next baseline state.
It does not mutate live parameters, gradients or state. `commit()` applies each
physical parameter once. The adapter accepts an already-updated first momentum
for feature remapping; Nesterov and orthogonalization do not update momentum
again. Disabled/zero-strength methods call the same stock baseline directly.
Reference parity is checked on CPU and CUDA.

Auxiliary probes run in FP32 with dropout disabled and isolated RNG; training
keeps dropout and BF16 autocast. Tiny reference tests use FP64. Eager attention
supports the required directional derivatives. Fused QKV is never repartitioned
for the baseline polar transform. Q/K diagnostics and corrections use tested,
disjoint column slices in GPT-2's `[input, output]` weight convention.

## Production schedules and approximations

- Q/K repair: one rotating head every eight committed updates, eight uniformly
  selected valid query rows per fit/check sequence. All causal keys for each row
  remain present. The basis includes both defect-gradient directions. The
  correction cap and checking-surrogate selection are those of the design.
  Full-model checking losses are additionally logged every 512 updates, without
  changing the local acceptance rule.
- Routing metric: one rotating head every eight updates; four rows, four sampled
  edges per row, mixture `0.05`, `rho=1`. Duplicate samples retain their weights.
  Factors are freshly sampled at each event. No sparse training attention or
  persistent routing inverse is introduced.
- Tied embedding metric: two paired model-categorical probes at updates
  `1, 17, 33, ...`, maximum age 16. The stored FP32 joint columns include the dense
  output path and are not row-sparsified. Failed refreshes use the previous cache
  only while its age is strictly below the limit.
- Notch: all MLP output projections; eight fixed Rademacher sketches per matrix,
  64-sample window, detector every 32 updates, no activation before 128, radius
  `0.8`, 64-update dwell and 128-update guard cooldown. The detector reads the
  unfiltered, unscaled direction. Coefficient changes re-prime the filter.
  Deterministic int8 sketch signs are cached separately and included in memory
  accounting; they are regenerated from saved seeds on resume.
- Feature remapping: MLP output projections; fixed contiguous 32-channel map
  blocks; refresh at `1, 9, 17, ...`; maximum snapshot age 32. Only the historical
  cohort is remapped. Fit/check anchors remain fixed. A separate rotating training
  sequence is captured at refreshes and audited every 128 updates against its
  matching prior snapshot, without controlling map acceptance. Setting interval
  1 gives the every-step reference; the class also exposes `feature_remap_v1`.
- LayerNorm response: one rotating `ln_2`/MLP block every 32 updates; up to two
  matrix and four normalization directions. The joint GGN is evaluated at the
  complete provisional model. All cross blocks share the same logit derivatives
  and softmax metric. Horizons `(0, 0.25, 1, 4)` use common damping, with at most six
  damping increases. Solved corrections are not radially clipped.

The Muon baseline collects the feature prediction diagnostic during its first
128 updates. The user selected a diagnostic-only campaign gate: all six methods
remain in the comparison even when prediction does not improve. Per-event
rejection/fallback rules remain active. Calibration checking is not validation.

## Equation-to-code and independent checks

Paths below are relative to `src/optimizer_v2/`. Every numbered equation in the six
documents is covered; grouped ranges share the named implementation/check.

| Design | Equations | Implementation | Independent check |
|---|---|---|---|
| Q/K | 1-3 | `attention.attention_rows`, `finite_qk_defect`; complete adapter increments | Direct biased Q/K multiplication; finite identity in FP64 |
| Q/K | 4 | `correction_jacobian`, `correction_adjoint` | Central differences and adjoint inner-product identity |
| Q/K | 5 | `omega_action` | Causal mask support and constant-shift nullspace |
| Q/K | 6-7 | `qk_correction`, `linalg.solve_spd` | Ridge stationarity; orthonormal-basis check |
| Q/K | 8 | `qk_correction` | Bound checked after solving, including zero defect |
| Q/K | 9-10 | `qk_correction` local replay/selection | Quadratic correction term and rejection of helpful-interaction cancellation |
| Routing | 1-4 | `edge_probability`, `sample_unordered_edges` | Exact edge enumeration, pseudoinverse resistance, normalization and sampling frequency |
| Routing | 5 | `edge_factors` | Autodiff of biased edge-logit differences |
| Routing | 6-7 | `edge_factors`, `factor_projection` | Exact expectation over edges and explicit weighted columns |
| Routing | 8-9 | `filter_routing_increment`, `proximal_coefficients` | Dense inverse, stationarity and non-amplification |
| Routing | 10 | `weighted_factor_gram`, `factor_combination` | Explicit parameter-space Gram and coupled Q/K cross block |
| Tied paths | 1-3 | `probes.split_path_forward`, `paired_embedding_sketch` | Equal-leaf forward, summed tied gradient and explicit two-path Jacobians/GGN |
| Tied paths | 4 | `sample_output_covariance_probe` | Enumerated joint categorical outcomes, mean/covariance and independent positions |
| Tied paths | 5-6 | `paired_embedding_sketch` | Paired path VJPs; explicit joint PSD metric |
| Tied paths | 7-8 | `linalg.low_rank_prox` | Dense solve, stationarity and norm bound |
| Tied paths | 9 | Explicit separate-path metric in mathematical tests | Joint-minus-separate equals both mixed terms; mixed-only metric may be indefinite |
| Notch | 1-2 | `adapter.Proposal.direction`, `optimizer.OptimizerV2.step` | Stock baseline parity and disabled trajectory |
| Notch | 3-4 | `temporal.notch_coefficients` | Unit DC gain, notch zero and off-notch frequency response |
| Notch | 5-6 | `df2t_step` | Independent direct difference equation |
| Notch | 7 | `prime_steady_state` | Constant input, activation and coefficient-switch restart |
| Notch | 8-9 | `fixed_sketch`, `spectral_detector` | Persistent sinusoid, zero signal and delayed activation |
| Notch | 10 | `guarded_filter_step` | Forced guard failure, exact bypass, reset and cooldown |
| Feature | 1-3 | `probes.collect_factors`, `temporal.remap_gradient` | Weight-gradient factorization with averaged loss; exact synthetic maps |
| Feature | 4-5 | `near_identity_map`, `fit_block_maps` | Raw ridge stationarity; clipped SVD bounds; zero energy and partial block |
| Feature | 6-7 | `predictive_maps` | Held-out rejection despite fitting improvement |
| Feature | 8 | `cohort_step`, every-step variant | Explicit one-step mapped-momentum equation |
| Feature | 9-10 | `cohort_step`, `OptimizerV2._feature` | Expanded gradient history, younger-cohort exclusion, R=1 and beta=0 |
| LayerNorm | 1-2 | `layernorm.make_local_basis`, `layernorm_correction` | Functional provisional model; disjoint basis supports |
| LayerNorm | 3-5 | `reduced_ggn` | Independent dense coordinate Jacobian and loss gradient at the provisional model |
| LayerNorm | 6-9 | `response_operator`, `solve_response_pair` | Small/large/zero horizon limits and joint-SPD Schur system |
| LayerNorm | 10-11 | `solve_equivalent_damped_block` | Independent block minimizer and positive extra damping |
| LayerNorm | 12 | `choose_shared_damping` | Shared damping retries, budget bounds and unchanged response equations |

Tests live in `tests/test_optimizer_v2_math.py` and
`tests/test_optimizer_v2_integration.py`. Existing spectral/A100 and legacy tests remain in
place. The mathematical functions retain FP64 inputs; the production scalar
systems are promoted to FP64 while full-model statistics/state remain FP32.

## Epochs, accounting and resume

The local profile dispatches through `scripts/run_gpt2_v2_comparison.py` into
`src/gpt2_v2_experiment.py`. The existing spectral/A100 runner and legacy
`src/optimizers.py` retain their prior implementations.
Epoch permutations use an independent seed/epoch generator, include the last
partial batch, and never accumulate across epoch boundaries. Since all packed
rows have the same length, weighting microbatch loss by its fraction of rows is
exactly weighting by its fraction of valid next-token targets.

The default dataset has 232,000 training and 486 validation blocks. One epoch is
7,250 updates. Five epochs process 593,920,000 input tokens and 592,760,000
next-token targets per method. Training/validation block hashes, initialization
hashes, calibration indices and epoch-order hashes are recorded. Data packing
retains the existing convention of dropping incomplete blocks within packing
chunks; an epoch means a complete pass over the resulting frozen packed split.

Checkpoints retain parameters, baseline state, all probe/filter/cohort history,
epoch cursor, Python/NumPy/Torch/CUDA RNG, counters, timing and prior peak memory.
Signals finish the active update, checkpoint, mark the job paused and stop the
comparison. Resume trims uncheckpointed/partial log records and verifies the
resolved configuration, implementation, model config, dataset SHA-256 inventory
and completed-checkpoint hashes. No completed final checkpoint is deleted.

Nonfinite training loss/gradients stop before an update and preserve the last
committed state, including the RNG and learning rates before the failed attempt.
A failed final validation is retried on resume before completion is reported.
Auxiliary numerical failures fall back to the baseline and are
logged; materially indefinite supposedly-PSD metrics raise a correctness error.
Programming errors are not swallowed. Failed checkpoints are not called completed
runs. `--steps` is an explicitly recorded smoke cap, not an epoch substitute.

Metrics separate training tokens, validation processing and auxiliary processing.
`auxiliary_input_tokens` counts tokens once per forward, backward, JVP or VJP
pass; `total_model_token_passes` adds training forward/backward and validation
passes. These are work counters, not FLOP-equivalence estimates. Local Q/K
attention replays are counted separately. Optimizer timing includes baseline
proposal/commit; auxiliary timing excludes those baseline operations. Peak
allocated/reserved CUDA memory includes transient live tensors. Persistent byte
counts deduplicate shared storage and separate baseline state, method state and
sketch caches. Comparing step counts alone does not establish a compute advantage.

## Verification commands

```bash
PYTHONPATH=src:scripts .venv-gpt2-v2/bin/python -m pytest -q tests/test_optimizer_v2_math.py tests/test_optimizer_v2_integration.py tests/test_gpt2_spectral_comparison.py tests/test_multi_step_spectral_geometry.py
bash scripts/run_gpt2_v2_5090.sh --dry-run
bash scripts/run_gpt2_v2_5090.sh --output results/gpt2_v2_5090_smoke --steps 33
```

The smoke must use its own directory. The migration and bounded validation record
is `records/2026-09-10-optimizer-v2-migration.md`. The original design sources are
preserved beside this document; the implementation has no dependency on the
source workspace. Dataset assets, environments and training outputs are ignored.

The v2 environment is declared separately in `requirements-gpt2-v2.txt` because
this experiment pins PyTorch 2.11.0+cu130 and Transformers 5.7.0. The launcher
inherits those installed packages and installs only missing dataset/plot
requirements. It does not install the repository-wide requirements or upgrade
PyTorch/Transformers. Set `GPT2_PYTHON` to select a compatible base interpreter.

This profile preserves the user-approved fixed settings and five complete epochs;
they retain the original 124M GPT-2 model and fixed settings for this profile.
No time cutoff may silently mark a partial five-epoch run complete. Signals pause
the comparison after the current update; reuse the output directory to resume.
Metric-versus-step and metric-versus-time PNGs include all requested methods once
a paired seed is complete. Bounded smokes are labelled in the plots.

The full five-epoch experiment is started only by the launch command in the
repository README.
