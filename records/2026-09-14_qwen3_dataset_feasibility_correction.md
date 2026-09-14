# Qwen3-0.6B FineWeb-Edu dataset feasibility correction

## Decision

The initial packed FineWeb-Edu cache with 2,000,000,000 training tokens and
100,000,000 validation tokens is retained only as an infeasible-run artifact.
It must not be used for a formal optimizer result.  The matched study instead
uses a deterministic 20,000,000-token training subset and a disjoint
5,000,000-token validation subset, both packed at sequence length 2,048 with
selection seed 1337.

## Evidence for rejecting the initial configuration

On ABA, formal AdamW and Muon runs began at 2026-09-14 08:20:13 UTC with
micro-batch size 8, gradient accumulation 1, and an evaluation/checkpoint
interval of 1,000 updates.  After 3 hours 47 minutes, neither had reached the
first interval: no metric trace, checkpoint, or result existed.  Both A100
80GB devices were fully occupied (about 72--74GiB allocated), so this was
active computation rather than a queue or allocation failure.

At 16,384 tokens per optimizer update, the 2B-token cache contains 122,070
updates per epoch and 366,210 updates for the required three epochs.  The
observed pre-first-evaluation pace makes that protocol months per optimizer,
before proposal tests.  It therefore fails the goal's required training-time
criterion.  The two runs were terminated deliberately before producing any
scientific result; their logs remain under `.cache/qwen3_0p6b/logs/` and the
old cache was renamed to
`.cache/qwen3_0p6b_fineweb_edu_2b_infeasible_20260914/`.

## Replacement protocol

The 20M/5M cache provides 9,765 packed training blocks and 2,441 held-out
blocks.  With batch size 8 it gives 1,221 updates per epoch and 3,663 matched
updates in three epochs.  This retains a multi-thousand-step perplexity curve
and a multi-million-token final validation estimate, while making AdamW,
Muon, Muown, and the proposal screens finish on an experimental timescale.

The cache identity is recorded by its generated manifest and SHA-256 digests;
every baseline and proposal must use the same manifest.  The formal label is
`formal_3epoch_b8_v64_i1000_20m` so it cannot be confused with the abandoned
2B-token attempt.
