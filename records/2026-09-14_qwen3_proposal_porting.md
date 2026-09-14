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

## Remaining before a Qwen proposal screen

1. Add a Qwen attention replay/probe for the routing-resistance factors using separate Q/K projections and grouped-query heads.
2. Add a temporary split-leaf tied-embedding forward for the paired-path sketch.
3. Route the notch and feature-remap controllers through the proposal adapter, with their required Qwen activation/gradient probes.
4. Screen each candidate only after the matched AdamW, Muon, and Muown baseline artifacts are complete.
