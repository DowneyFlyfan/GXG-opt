# Qwen3-0.6B active-proposal porting record

## Scope

The active idea set is limited to the four documents directly under `records/ideas`: `routing_resistance_v1`, `tied_path_curvature_v1`, `proposal_notch_v1`, and `feature_remap_cohort_v1`.  Files under `records/ideas/Done` and `records/ideas/Failed` are excluded.

## Baseline boundary

`src/qwen3_proposals.py` introduces `QwenMuonProposalAdapter`.  It computes the repository custom Muon matrix increment without mutating a parameter or its momentum buffer, separates learning and decoupled-decay increments, and commits an optional correction plus the next momentum state exactly once.  The Qwen test compares one adapter proposal/commit against `optimizers.Muon.step()` with a numerical tolerance of `2e-5`; this tolerance is required because separately compiled bfloat16 Newton--Schulz evaluations are not bitwise identical.

## Qwen-specific facts verified on ABA

- Qwen3-0.6B has 28 layers, hidden width 1,024, 16 query heads, 8 key/value heads, vocabulary 151,936, and tied input/output embeddings.
- Attention uses separate `q_proj`, `k_proj`, `v_proj`, and `o_proj` modules.  The existing GPT-2 proposal implementation assumes fused `c_attn` tensors, so its head-probe adapter cannot be reused unchanged.
- The active formal AdamW and Muon baselines use the independent Qwen runner and are untouched by this proposal-porting work.

## Verification

- Local: `tests/test_qwen3_proposals.py`, `tests/test_qwen3_model.py`, and `tests/test_qwen3_ppl_experiment.py`: 9 passed.
- ABA: `tests/test_qwen3_proposals.py`: 2 passed while both formal A100 jobs remained active.
- Existing mathematical/integration coverage for the four source mechanisms: `tests/test_optimizer_v2_math.py` and `tests/test_optimizer_v2_integration.py`: 55 passed, 1 skipped.

## Proposal-notch matrix wrapper

`QwenProposalNotchOptimizer` now owns the selected Muon matrix route for
`proposal_notch_v1`.  On each update it obtains exactly one non-mutating
baseline proposal, applies `qwen_notch_corrections` to its post-polar
direction, and commits the resulting parameter and momentum state exactly
once.  It deliberately does not own auxiliary parameters: the eventual Qwen
candidate will retain their independently tuned AdamW route.

The first optimizer-step test compares this wrapper with direct custom Muon
while the notch detector is still ineligible.  The parameters agree to the
same `2e-5` compiled Newton--Schulz tolerance used for the proposal adapter;
the diagnostic is explicitly inactive.  Local verification after adding the
wrapper: `tests/test_qwen3_proposals.py`: 4 passed.

The Qwen optimizer factory now exposes the wrapper as `proposal_notch_v1` and
keeps all non-matrix parameters in the selected baseline's independently
tuned AdamW auxiliary group.  The command-line trial interface accepts that
candidate, but `render_qwen_comparison` remains explicitly baseline-only
(`AdamW`, `Muon`, and `Muown`).  This prevents a candidate trace from being
silently treated as one of the required matched baselines.  The local factory,
trial-path, command-line, and primitive suite passed: 13 tests.

For `routing_resistance_v1`, `QwenAttentionReplayCapture` registers a
temporary Qwen attention forward-pre-hook and stores only one selected causal
sequence's detached hidden states and rotary-embedding pair.  Its replay
therefore uses the ordinary training forward rather than retaining activations
for all 26 selected layers or adding a full-model forward.  It is strictly a
data-access adapter: no parameter or optimizer state is changed.  Local
attention tests, including a three-sequence capture that confirms the stored
batch dimension is one, passed: 4 tests.  ABA CPU-only verification against
the cached real Qwen3-0.6B attention layer also passed for a three-token,
two-sequence input: the selected replay shapes were `X=(3,1024)`,
`Q=(3,128)`, and `K=(3,128)`.

## Routing-resistance controller

`QwenRoutingResistanceOptimizer` implements `routing_resistance_v1` as a
non-mutating Muon-proposal filter.  On an event it installs the one-sequence
capture before the ordinary forward, replays a rotating interior Q/K head,
selects causal rows from `1..T-1`, and passes only the Q/K learning increments
to `qwen_route_head_corrections`.  That helper supplies the resistance-mixture
edge factors and the joint Q/K Woodbury proximal correction; decay remains in
the unmodified baseline proposal.  The adapter then commits all matrix
parameters and their Muon momentum buffers exactly once.

The initial scalable schedule is one head every 8 committed updates, four
sampled query rows and four sampled edges per row, with `rho=1` and mixture
0.05.  These are exposed through the Qwen trial command as declared tuning
arguments rather than hidden constants.  `rho=0` is a direct no-hook Muon
bypass.  The active-path test covers one captured four-token sequence and
verifies 12 sampled edges (three causal rows times four), selected-layer
diagnostics, and hook removal; the disabled-path test numerically matches
direct Muon.  The local Qwen-specific suite passed: 20 tests.  No GPU screen
has been launched; the formal baseline gate remains in force.

## Tied-path curvature controller

`QwenTiedAdamWProposalAdapter` mirrors the tuned auxiliary AdamW route
(`betas=(0.9, 0.95)`) without mutating the physical embedding or its first and
second moments until commit.  Its two-step test agrees with direct PyTorch
AdamW.  `QwenTiedPathOptimizer` owns that one embedding state, refreshes the
paired split-leaf sketch on a saved training sequence, applies
`qwen_tied_proximal_correction` only to the AdamW learning increment, and then
commits decay and state once.  The Qwen factory keeps interior matrices under
Muon and removes the physical embedding from the remaining AdamW auxiliary
group, preventing tied-alias double updates.

The initial candidate settings are the design-document values: `rho=1`, two
paired categorical probes, refresh interval 16, and maximum cache age 16.
All are explicit trial arguments.  Zero strength directly bypasses both batch
retention and the split-leaf probe.  For Qwen's vocabulary 151,936 and width
1,024, two retained FP32 sketch columns occupy 1,244,659,712 bytes
(approximately 1.16 GiB), excluding the two temporary split leaves, their
gradients, and probe activations.  Capacity must therefore be checked on an
idle A100 before launch; an equal effective batch with gradient accumulation
is permitted if the probe peak exceeds the standard microbatch's headroom.
Local verification including runner handoff passed: 29 tests.  No tied-path
GPU screen has started before the matched-baseline gate.

## Feature-remap preflight

`qwen_feature_prediction_diagnostics` now evaluates the defining held-out
criterion before a Qwen momentum-remap controller can be enabled.  It accepts
only matched dictionaries of fit/check dense factors and calls the existing
block-ridge, near-identity `predictive_maps` implementation.  The output is
diagnostics only: it cannot change model parameters, momentum, or optimizer
state.  Its FP64 synthetic exact-drift test verifies that the accepted map has
lower held-out gradient error than the unremapped historical gradient.  This
is the required prediction-first gate; an actual Qwen anchor trajectory will
be collected after the formal matched baselines are complete.

`qwen_feature_drift_preflight` is the snapshot-level runner for that gate.  It
takes before/after Qwen snapshots, fixed fit/check training anchors, and named
dense modules; it forces both models to the same evaluation mode, captures all
four factor sets, and returns only the held-out diagnostics.  Its test changes
a tiny dense layer between snapshots and verifies that neither model receives
a parameter gradient.  This is the exact operation to run on the initial and
formal-Muon final snapshots before feature-remap training is authorized.

## Candidate comparison artifacts

`render_qwen_candidate_comparison` and the `render-candidate` command render
one proposal trace together with the fixed AdamW, Muon, and Muown formal
traces.  Each call writes a PNG for completed optimizer steps and a PNG for
elapsed wall-clock time, both with validation perplexity as the vertical
metric.  This is deliberately separate from baseline-only rendering so a
candidate is never mislabeled as a baseline.  Unit coverage writes synthetic
traces for all four curves and verifies both candidate output files.

## Long-run checkpoint durability

The Qwen runner now writes an atomic checkpoint after each periodic validation
record and otherwise writes the final checkpoint.  Each checkpoint contains
the model, all optimizer states, completed-update count, completed-epoch count,
the active epoch, peak allocated memory, and the token-cache manifest digest.
The write first targets a sibling `.partial` file and then atomically replaces
the cached checkpoint, so an interruption cannot leave a half-written final
checkpoint.  Both the final and partial files remain exclusively under
`.cache/qwen3_0p6b/checkpoints`; metrics and compact result evidence remain in
their normal `metrics/nlp` and `results/nlp` locations.  The existing formal
AdamW and Muon processes predate this source change and are intentionally not
restarted.

The regression test runs a two-update CPU trial with validation every update
and confirms the checkpoint writer is invoked exactly at updates 1 and 2.
The complete focused Qwen suite after this change passed: 36 tests.

The runner now supports `--resume` for checkpoints created by this version.
Each periodic checkpoint includes the model and optimizer states, completed
updates/epochs/batches, active-epoch sampler-generator state, accumulated
elapsed time, last validation perplexity, CPU/CUDA random-number-generator
states, and manifest-bound resolved configuration.  Resumption rejects a
different configuration or cache digest, reconstructs the saved shuffled epoch,
skips only already committed batches, and appends no duplicate metric point.
The interruption regression deliberately raises after checkpointing step 1,
then resumes to step 3 with exactly `[1, 2, 3]` in the metric trace.  The two
currently active formal processes began before this format existed and are not
restarted; future Muown and proposal trials receive this recovery capability.
The same test runs an uninterrupted control with identical seed and data, then
compares every final model-state tensor and final perplexity to the resumed
trial; both are equal under the CPU test contract.

## Formal-baseline admission gate

The Qwen runner now enforces the baseline boundary rather than relying only on
the external monitor.  Any non-baseline proposal (`proposal_notch_v1`,
`routing_resistance_v1`, or `tied_path_curvature_v1`) is rejected before model
loading unless all three results named by its run label exist and each records
its own optimizer name, that label, a positive update count, finite final
perplexity, exactly the requested epoch count, and the candidate cache's
manifest digest.  This prevents a partial, differently-tokenized, or
incomplete baseline from authorizing a proposal screen.  The unit test first
observed the missing gate, then creates three matching completed synthetic
results and verifies admission.

Candidate tuning uses its own run label while passing
`--baseline-run-label formal_3epoch_b8_v64_i1000`.  The runner applies the
completion/manifest gate to that formal label, then writes the candidate trace
under its separate screen or final label.  Likewise, `render-candidate` accepts
the candidate run label together with the baseline label, so a 50-update
tuning trace cannot overwrite or masquerade as the final candidate comparison.
Tests cover both the command-line label split and rendering a separately
labelled proposal alongside three formally labelled baseline traces.

## Remaining before a Qwen proposal screen

1. Let the active formal AdamW and Muon runs complete, then launch the exact
   matched Muown run with its independently tuned direction, gain, and
   auxiliary learning rates.
2. Render the three formal baseline perplexity-versus-step and
   perplexity-versus-time PNG artifacts.
3. Screen routing resistance, tied-path curvature, and proposal notch one at a
   time against those three artifacts; run the feature-remap preflight before
   authorizing its controller.
