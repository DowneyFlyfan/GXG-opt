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

## Verified replacement launch

The replacement manifest has SHA-256
`97a4ad1da6bbc128a2161a276c2d093821783133e50b27587fa20c9dd45f672c` and records
19,267 training documents and 4,722 validation documents.  The packed files
are exactly 80,000,000 and 20,000,000 bytes respectively (uint32 token IDs).

At 2026-09-14 12:13 UTC, the formal AdamW run (PID 3853368, A100 0) and formal
Muon run (PID 3853369, A100 1) were launched with the tuned learning rates
from the baseline protocol: AdamW `3e-5`; Muon matrix `5e-5` and auxiliary
AdamW `3e-5`.  Both were confirmed live, each at 100% GPU utilization with
about 74.4GiB allocated.  Muown will use its separately tuned direction
`5e-5`, gain `3e-6`, and auxiliary `3e-5` configuration when either device is
released.

## Measured lower bound after launch

At 10 hours 03 minutes of uninterrupted execution, both formal processes were
still live at full A100 utilization and had not reached their first 1,000-step
evaluation/checkpoint boundary.  This proves a lower bound of 36 hours for a
3,663-update, three-epoch run at the present implementation's throughput;
actual duration also includes validation and checkpoint time.  The study is
therefore an A100 multi-day benchmark, not a short screen.  The active runs
remain valid and are intentionally continuing, but later result reporting must
use this observed lower bound rather than the earlier optimistic estimate.
