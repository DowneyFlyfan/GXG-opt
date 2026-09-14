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

## Measured first interval after launch

The elapsed display from `ps` was initially misread: for an elapsed duration
below one hour it uses `MM:SS`, not `HH:MM`.  The durable AdamW record at step
1,000 establishes the correct measurement: 836.0116 seconds (13.93 minutes)
for 16,384,000 exposed training tokens, validation perplexity
`16.5583577720`, and peak PyTorch allocation 68,225.61MiB.  Its atomic
checkpoint was written under `.cache/qwen3_0p6b/checkpoints/` and is
3,576,770,295 bytes.

Thus the 3,663-update, three-epoch run is expected to take on the order of 51
minutes before evaluation/checkpoint overhead, rather than the previously
misreported multi-day lower bound.  The corrected 20M/5M protocol is feasible
for the three baselines and sequential proposal study on ABA A100 capacity.

## Matched first-interval evidence

Both formal runs reached step 1,000 on the same manifest and token exposure
(16,384,000 tokens) during epoch 1.  AdamW recorded validation perplexity
16.5583577720 in 836.0116 seconds; Muon recorded 16.5312007434 in 945.4358
seconds.  Muon is therefore 109.42 seconds slower (13.1%) at this first
interval while its perplexity is 0.0271570 lower.  This is an early matched
observation, not a final comparison.

The AdamW and Muon atomic checkpoints are respectively 3,576,770,295 bytes
and 2,758,759,049 bytes, both below `.cache/qwen3_0p6b/checkpoints/` and bound
to manifest `97a4ad1da6bbc128a2161a276c2d093821783133e50b27587fa20c9dd45f672c`.

## AdamW second-interval evidence

AdamW reached step 2,000 during epoch 2 on the same manifest and with
32,768,000 exposed training tokens.  Its 64-batch validation perplexity was
`16.4296490720` at 1,692.9553 seconds, with the same 68,225.61MiB peak PyTorch
allocation.  This is a 0.1287087000 perplexity reduction from its matched
step-1,000 point.  Muon was still training when this evidence was captured, so
this is a within-AdamW trajectory observation, not a cross-optimizer result.

## Matched second-interval evidence

Muon reached step 2,000 during epoch 2 on the same manifest and 32,768,000
exposed training tokens.  Its 64-batch validation perplexity was
`16.6182285301` at 1,908.0183 seconds with a 67,445.61MiB peak allocation.
That is 0.0870277867 above its own step-1,000 value and 0.1885794582 above
AdamW at the matched step, while taking 215.06 seconds longer.  These are
intermediate matched measurements only; final full-validation evidence is
required before selecting a baseline or judging either optimizer.

## AdamW third-interval evidence

AdamW reached step 3,000 during epoch 3 on the same manifest with 49,152,000
exposed training tokens.  Its 64-batch validation perplexity was
`16.5418971957` at 2,552.1347 seconds, again with a 68,225.61MiB peak
allocation.  This is 0.1122481237 above the AdamW step-2,000 minimum.  The
remaining 663 updates and final result are required to interpret this
non-monotone interval rather than treating it as a completed comparison.

## Muon third-interval evidence

Muon reached step 3,000 during epoch 3 on the same manifest and 49,152,000
exposed tokens.  Its 64-batch validation perplexity was `17.2108652486` at
2,890.5055 seconds with a 67,445.61MiB peak allocation.  This is
0.5926367185 above its Muon step-2,000 value and 0.6689680530 above AdamW at
the matched step.  The run remains live; these are intermediate measurements,
not a final baseline judgment.

## Completed AdamW baseline and Muown handoff

AdamW completed all three epochs and 3,663 updates on the formal manifest.
Its final 64-batch validation perplexity is `16.4880045840` after 3,169.7594
seconds, with 60,014,592 exposed tokens and a 68,225.61MiB peak allocation.
The final JSON confirms the manifest digest, completed epoch/update counts, and
checkpoint-backed result under the formal label.

After confirming AdamW had exited and GPU 0 held only 4MiB, one fresh formal
Muown trainer was launched on that device as PID 3881012.  It uses the matched
three-epoch, batch-8, 64-batch-validation protocol and its separately tuned
direction/gain/auxiliary rates `5e-5` / `3e-6` / `3e-5`.  It reached 100% GPU
utilization during model loading/training setup.  No prior Muown metric,
checkpoint, result, or trainer existed, so this did not duplicate a run.

## Completed Muon baseline

Muon completed the same three epochs and 3,663 updates on the identical
manifest.  Its final 64-batch validation perplexity is `17.1557168418` after
3,537.4299 seconds, with 60,014,592 exposed tokens and a 67,445.61MiB peak
allocation.  Relative to the completed AdamW run, this final intermediate
validation value is 0.6677122577 higher and its elapsed time is 367.67 seconds
longer.  Full held-out validation for all three formal baselines remains
pending until Muown completes; these values do not yet select a winner.

## Muown first-interval evidence

Muown reached step 1,000 in epoch 1 on the formal manifest, exposing
16,384,000 tokens.  Its 64-batch validation perplexity was `16.5301853239` at
955.7704 seconds, with 68,228.04MiB peak allocation.  This is 0.0010154195
lower than Muon at the same step but takes 10.33 seconds longer; it is still
an early matched point rather than a conclusion about the final baseline.

## Muown second-interval evidence

Muown reached step 2,000 during epoch 2 on the formal manifest with
32,768,000 exposed tokens.  Its 64-batch validation perplexity was
`16.6168451977` at 1,926.2696 seconds with 68,228.04MiB peak allocation.  It
is 0.0013833325 lower than Muon at the matched step but 18.25 seconds slower.
This remains intermediate matched evidence; full held-out validation after all
three final checkpoints is still required.

## Muown third-interval evidence

Muown reached step 3,000 during epoch 3 with 49,152,000 exposed tokens.  Its
64-batch validation perplexity was `17.2021598009` at 2,898.9145 seconds and
its peak allocation remained 68,228.04MiB.  This is 0.0087054477 lower than
Muon at the matched step, while taking 8.41 seconds longer.  The Muown process
was still finalizing its periodic atomic checkpoint when captured, so this is
not its final result.
