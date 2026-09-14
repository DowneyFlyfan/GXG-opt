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

## Completed Muown baseline

Muown completed the matched three-epoch protocol without resuming or
duplicating a prior trainer.  Its final 64-batch validation perplexity is
`17.1508224156` after 3,663 updates and 3,567.5726 seconds, with the same
`60,014,592` exposed tokens, manifest digest
`97a4ad1da6bbc128a2161a276c2d093821783133e50b27587fa20c9dd45f672c`, and
68,228.04MiB peak allocation.  It is 0.0048944262 lower than Muon on this
intermediate validation estimate but 30.14 seconds slower.  The three
training runs are now complete; the next required evidence is the separate
full-held-out evaluation of every saved checkpoint before selecting a
baseline or screening a proposed optimizer.

## Full-held-out baseline validation and incumbent

Each completed checkpoint was then evaluated once over all 2,441 packed
held-out blocks (`4,999,168` tokens, delivered as 306 batch-8 validation
passes).  The results share the formal cache manifest and are: AdamW
`16.0881141034`, Muown `16.7316303852`, and Muon `16.7366716663` perplexity.
Thus AdamW is the formal incumbent for this Qwen3-0.6B/FineWeb-Edu protocol;
Muown is 0.0050412811 lower than Muon but both are materially above AdamW.
These checkpoint evaluations, rather than the 64-batch curve points, are the
baseline selection evidence for proposal trials.

## Proposal-notch stability screen

The first independent proposal screen used `proposal_notch_v1` at learning rate
`1e-4`, auxiliary rate `3e-5`, batch 8, and the same formal cache/seed.  It
completed 50 updates (819,200 exposed tokens) without a nonfinite update or
out-of-memory failure.  Its 64-batch validation perplexity was
`17.9260624942` after 69.8084 seconds; peak allocated memory was
70,566.03MiB.  This is a stability and resource result only: the formal
baselines have no 50-update validation point, so a ranking against their
1,000-update values would be an invalid comparison.  The next notch run must
reach at least step 128 to exercise the specified resonance detector, and
step 1,000 to obtain a directly matched curve point.

## Routing-resistance stability screen

The first routing launch failed before committing an update because a CPU
sampler received a CUDA generator.  A targeted CUDA/CPU generator regression
test exposed the same mismatch; sampling is now performed on the probability
tensor's device.  The clean retry completed 50 updates on the same formal
cache and seed: 64-batch validation perplexity `17.9234457190` at 819,200
exposed tokens, 72.4473 seconds, and 67,445.61MiB peak allocation.  As with
the notch 50-update screen, this establishes execution feasibility but is not
yet a valid performance comparison with the step-1,000 baseline curve.

## Active-mechanism memory correction and tied-path screen

The initial batch-8 active-notch and tied-path attempts exhausted the
79.25GiB A100 when an active mechanism required an additional 9.27GiB.  They
produced no checkpoint, metric, or result artifact.  The retry preserves the
required effective batch of 8 by using micro-batch 4 with accumulation 2 and
enables expandable allocator segments.  The tied-path 50-update screen then
completed at `18.3678238040` perplexity, 819,200 exposed tokens, 77.9904
seconds, and 37,370.67MiB peak allocation.  It is resource-valid but early
screening evidence only; the separate active-notch step-1,000 job remains the
first directly comparable candidate curve.

## Proposal-notch matched point

With the memory-safe micro-batch-4/accumulation-2 configuration, the
proposal-notch run reached the directly comparable 1,000-update point:
`17.8235363734` perplexity at 16,384,000 exposed tokens in 1,080.9697
seconds, with 39,304.10MiB peak allocation.  This is finite and resource-valid
but 1.2923356300 higher than the matched Muon value (`16.5312007434`), so the
tested notch configuration is not competitive.  Its standalone 50-update
screen is retained as stability evidence; no advantage is inferred from it.

## Routing-resistance matched point

The memory-safe routing-resistance run also reached step 1,000 with the same
micro-batch-4/accumulation-2 effective batch, cache manifest, seed, and
64-batch validation convention.  Its perplexity was `17.8226207706` after
`1,040.3120` seconds, using `36,183.65MiB` peak allocated memory.  This is
finite and resource-valid, but `1.2914200272` higher than Muon's matched
step-1,000 value (`16.5312007434`), so this tested routing configuration is
not competitive.  Its full held-out checkpoint evaluation is
`16.9405418546` perplexity over all `4,999,168` validation tokens (611
micro-batch-4 passes), `0.8524277512` above the formal AdamW incumbent
(`16.0881141034`).  It therefore supplies a valid negative screening result,
not evidence of a general failure of routing-aware curvature or of every
hyperparameter setting.

## Tied-path curvature matched point

The tied-embedding curvature candidate reached the same 1,000-update point
with micro-batch 4 and accumulation 2: `17.8228654358` perplexity after
`1,151.8512` seconds and `37,370.67MiB` peak allocated memory.  It is finite
and resource-valid but `1.2916646924` above the matched Muon value
(`16.5312007434`), so this tested tied-path configuration is not competitive.
Its full held-out checkpoint evaluation was `16.9398539996` perplexity over
all `4,999,168` validation tokens (611 micro-batch-4 passes),
`0.8517398961` above formal AdamW.  This is a valid negative screening result
for the tested configuration, not a conclusion about every tied-path design.

## Feature-remapping prediction gate

Before implementing or training the feature-remapping optimizer, a separate
eight-update AdamW trajectory was produced with the formal AdamW learning rate
`3e-5`, effective batch 8 (micro-batch 4, accumulation 2), seed 1337, and the
same manifest.  The pretrained model and the post-step-8 checkpoint were
probed using one fixed training block for fitting and a distinct fixed training
block for checking.  All 26 interior MLP down-projection maps passed the
defined held-out `e_map <= 0.95 e_raw` gate.  The median map/raw error ratio
was `0.8018571734` (best `0.7513718373`, worst `0.9015710494`).

This does **not** yet establish that map remapping is the useful intervention:
the required scalar-rescaling control achieved a much lower median error ratio
of `0.2129675001` on the same checks.  The measured map therefore has a real
prediction signal but is presently dominated by a simpler control.  The next
eligible feature experiment must compare the cohort map against that scalar
control, retain distinct fit/check anchors, and add the specified fresh
training-only diagnostic anchor before any performance claim.

## Feature-cohort anchor screens

The feature-cohort implementation was exercised first with the candidate
matrix learning rate `1e-4`, auxiliary AdamW learning rate `3e-5`, and the
same effective batch eight (micro-batch four, accumulation two).  At nine
updates, where the first history remapping has actually occurred, the
feature-map arm recorded `14.5722232974` 64-batch perplexity in `25.9574`
seconds; the scalar control recorded `14.5604467453` in `26.2933` seconds.
Both used about `40,188MiB` peak allocated memory.  The scalar control is
therefore lower by `0.0117775520` at this deliberately tiny diagnostic
screen.  It is not a formal comparison and does not establish an advantage
over any baseline.

The implementation maps only the decayed historical cohort contribution and
then adds the fresh gradient cohort unchanged, consistent with Muon's
unnormalized momentum convention `m <- beta*m + g`.  It also converts the
learned floating-point map to the momentum tensor's dtype before applying it;
this fixes the active BF16 mixed-dtype path.  The current anchor pair is fixed
and distinct for fit versus check, but the fresh training-only diagnostic
anchor required by the design has not yet been persisted.  Consequently the
following run is evidence for the scalar-control implementation and not a
completed validation of the map mechanism.

## Feature-scalar cohort matched 1,000-update result

The scalar-control cohort run completed exactly 1,000 updates on the formal
cache manifest, again with micro-batch four and accumulation two.  Its
64-batch curve metric was `17.8198637558` after `1,649.2295` seconds with
`40,187.65MiB` peak allocated memory.  This is materially slower than the
ordinary Muon-family baseline because it executes fixed fit/check anchor
forwards for all 26 target MLP down-projections every eight updates.

Its saved checkpoint was evaluated over all `4,999,168` held-out tokens
(`611` micro-batch-four validation passes), producing perplexity
`16.9340264165`.  That is `0.8459123131` above the formal AdamW incumbent
(`16.0881141034`), so the tested scalar-control configuration is a valid
negative result.  The metric, full-validation JSON, and matched step/time
plots are retained locally before releasing its 6.96GB remote checkpoint for
the map-control comparison; no checkpoint is needed to reproduce this
reported evaluation artifact.

## Feature-map cohort matched 1,000-update result

The accepted-map cohort arm then completed the same matched 1,000-update
configuration. Its 64-batch curve value was `17.8215783421` after
`1,625.2345` seconds with `40,187.65MiB` peak allocated memory. This is
`0.0017145863` higher perplexity than the scalar control's corresponding
point, although it was `23.9950` seconds faster in this one run.

The complete held-out evaluation over `4,999,168` validation tokens (`611`
micro-batch-four passes) yielded `16.9387846280` perplexity. It is
`0.0047582115` higher than the scalar control and `0.8506705246` higher than
the AdamW incumbent. Thus the fitted feature-map intervention is not
competitive in this matched screen, despite passing the earlier two-anchor
prediction gate. Together, the scalar and map controls are valid negative
screens for this feature-cohort proposal at the tested rates and schedule;
they do not establish a general failure of gradient-feature prediction or
settle the unimplemented fresh-diagnostic-anchor audit. Both metric/result
JSON files and the metric-steps/time plots are retained locally. The remote
map checkpoint can be released after this evidence is committed, because it
is only needed for the completed full-validation pass already recorded here.

## Deferred high-rate local screen

After identifying that the candidate runs were screens rather than a proper
learning-rate search, a `5e-3` proposal-notch stability screen was requested.
Both ABA A100s were actively occupied by unrelated CryoET processes, so the
identical packed model and cache were copied to the idle local RTX 5070 Ti.
The proposed local configuration retained effective batch eight with
micro-batch one and accumulation eight.  Without activation checkpointing it
failed on the first backward call after allocating `13.98GiB` of the 15.47GiB
device, needing a further `1.16GiB` allocation.

An explicit opt-in activation-checkpointing runner mode was then added and
tested; it leaves the normal matched protocol untouched.  The same local
screen still left only about 15MiB free and retried an allocator request during
the first backward pass without committing an update.  It was interrupted
after repeated allocation failures.  No metric, checkpoint, or result JSON
was produced, so this is a local-hardware feasibility failure, not evidence
about the `5e-3` learning rate.  A valid high-rate screen requires an A100
with sufficient free memory; it must retain the declared rate, cache manifest,
and effective batch when that capacity becomes available.

## AdamW `5e-3` local high-rate screen

The baseline-only retry is feasible with the same local RTX 5070 Ti, the
opt-in activation checkpointing mode, micro-batch one, and accumulation eight.
It completed 50 updates and exposed 819,200 training tokens from the formal
cache manifest.  The 64-batch validation perplexity was `4263.8680965176`
after `150.5729` seconds, with `8,899.10MiB` peak allocated memory.  The
metric/result pair is retained as
`qwen3_0p6b__adamw_lr5e3_s50_local5070ti_ckpt_mb1a8_20m__adamw`.

This is a valid numerical-stability screen, unlike the earlier candidate
memory failure: it completed and produced a finite metric.  It also decisively
rejects AdamW learning rate `5e-3` for this protocol: the value is orders of
magnitude worse than the formal AdamW held-out perplexity `16.0881141034`.
It must not be extended to 1,000 updates or promoted to a formal baseline.
The next rate is selected from a lower but still aggressive bracket, while
Muon and Muown retain separate matrix/direction, gain, and auxiliary rates.

## Muon `5e-3` / `3e-4` local high-rate screen

Muon was then screened locally with a matrix learning rate `5e-3` and a
separate auxiliary AdamW learning rate `3e-4`, still using micro-batch one,
accumulation eight, activation checkpointing, seed 1337, and the formal cache
manifest.  It completed all 50 updates, exposed 819,200 tokens, and used
`8,119.10MiB` peak allocated memory.  Its 64-batch validation perplexity was
`440.5633492583` after `156.3349` seconds.

This is finite and locally feasible but rejects this aggressive Muon setting:
the 50-update perplexity is catastrophically high, so it will not be extended
or compared to the 1,000-update formal curves.  Importantly, this result does
not support using a shared rate across Muown's subsystems; Muown needs an
independent direction/gain/auxiliary screen.

## Muown independent-rate `5e-3` local high-rate screen

Muown was screened with deliberately distinct rates: matrix direction `5e-3`,
gain `3e-5`, and auxiliary AdamW `3e-4`.  The other settings were unchanged:
local RTX 5070 Ti, activation checkpointing, micro-batch one, accumulation
eight, seed 1337, the formal cache manifest, and 50 updates.  It completed
with `443.4509109184` 64-batch validation perplexity after `157.6255`
seconds, 819,200 exposed tokens, and `8,901.53MiB` peak allocation.

The distinct gain rate does not rescue this extreme direction rate: this
configuration is rejected and will not be extended.  The result is useful for
bracketing because it demonstrates a completed, reproducible high-rate
failure while preserving the required separate Muown subsystems.  The next
bracket remains aggressive but is reduced to `5e-4` rather than pretending
that a 50-update exploding run is an optimizer comparison.

## AdamW `5e-4` local bracket screen

The first reduced-rate AdamW bracket used `5e-4`, still 16.7 times the formal
AdamW rate, with the exact same local activation-checkpointed effective-batch
eight protocol.  It completed 50 updates at `92.5351713742` 64-batch
validation perplexity in `150.3659` seconds, exposing 819,200 tokens and
using `8,899.10MiB` peak allocation.  It is substantially less unstable than
`5e-3`, but still far outside a useful range and is rejected before any long
run.  The next AdamW bracket is `1e-4` (3.33 times the formal rate).

## AdamW `1e-4` local bracket screen

At `1e-4` (3.33 times the formal AdamW rate), the 50-update screen completed
with finite `19.9008978517` 64-batch validation perplexity in `150.1721`
seconds, `819,200` exposed tokens, and `8,899.10MiB` peak allocated memory.
This is neither a formal comparison nor a selected learning rate: the formal
baselines do not have a matched 50-update metric.  Unlike the two higher
rates, however, it is not an immediate exploding trajectory.  The exact saved
checkpoint will therefore be resumed to 200 updates under the same label and
all other controls before deciding whether a 1,000-update comparison is
justified.

## AdamW `1e-4` 200-update local continuation

The exact 50-update checkpoint was extended without reinitializing the model
or replaying metric steps.  The matched 64-batch perplexities at updates 100,
150, and 200 were `19.6552786200`, `19.7794295693`, and `19.6312532669`.
The final point took `594.8787` seconds and used `10,036.15MiB` peak
allocation.  This flat, high trajectory is not suitable for a 1,000-update
comparison and rejects AdamW `1e-4`; it is higher-rate evidence, not a change
to the formal `3e-5` incumbent.

## ABA availability race and local Muon `1e-4` screen

Both ABA A100s were observed at 4MiB used and an AdamW `5e-5` screen was
started immediately.  During model loading, unrelated CryoET trainers began
using about 19.7GiB on each GPU; AdamW then could not allocate its 9.27GiB
backward workspace.  It produced no metric, checkpoint, or result JSON, so it
is not learning-rate evidence and was not retried against those active jobs.

The matched fallback screen for Muon uses matrix learning rate `1e-4` and its
separate auxiliary AdamW rate `5e-5`, with local micro-batch one and
accumulation eight.  It completed 50 updates with `18.8077592748`
64-batch perplexity after `158.0378` seconds, 819,200 exposed tokens, and
`8,119.10MiB` peak allocation.  It is finite enough to extend before a formal
selection, but cannot be compared to the formal curves until a matched longer
point is available.

## Muon `1e-4` 200-update local continuation

Muon was resumed from its saved 50-update checkpoint without duplicate metric
steps.  Its 64-batch perplexity sequence at updates 50/100/150/200 is
`18.8077592748`, `18.6448649635`, `18.6871418920`, and `18.6490005416`.
The best value occurs at step 100 and the curve is effectively flat through
step 200; peak allocation was `9,256.06MiB` and total elapsed time was
`615.7707` seconds.  This is a completed high-rate bracket, not yet a formal
winner: it must be weighed against Muown at the same interval and then, if
selected, against the formal step-1,000 baseline evidence.

## Muown independent-rate `1e-4` local screen

The corresponding Muown screen used direction `1e-4`, gain `6e-6`, and
auxiliary AdamW `5e-5`; it did not reuse Muon's rate for the gain update.  It
completed the same 50 updates with `18.8076339301` 64-batch perplexity in
`159.5453` seconds, 819,200 exposed tokens, and `8,901.53MiB` peak allocation.
It is finite and numerically almost identical to the Muon high-rate screen
(`0.0001253447` lower), but this short local screen alone cannot select either
variant.  Muon is extended first to the next shared checkpoint interval;
Muown's checkpoint remains intact for the same continuation.

## Muown independent-rate `1e-4` 200-update local continuation

Muown was resumed from the preserved 50-update checkpoint under the same
effective-batch-eight protocol.  Its step 50/100/150/200 perplexities are
`18.8076339301`, `18.6541403022`, `18.6993103125`, and `18.6429582433`.
Its best 200-update value is only `0.0060422984` below Muon's corresponding
`18.6490005416`, which is too small for a one-seed selection claim.  It took
`619.6277` seconds with `10,038.49MiB` peak allocation.  The next test moves
the selected Muown high-rate configuration to ABA with micro-batch four and
accumulation two, preserving effective batch eight while fitting beside the
unrelated 20GiB-per-GPU workload.

## ABA micro-batch availability check

The ABA Muown micro-batch-four/accumulation-two attempt was launched only
after confirming the nominal memory headroom, but the CryoET workload holds
GPU 1 in an exclusive/busy state.  CUDA rejected model placement before the
first training forward, metric, checkpoint, or result JSON.  This is neither
a memory measurement nor learning-rate evidence, and the active user-owned
processes were not interrupted.  The selected high-rate Muown configuration
therefore proceeds locally to its directly comparable step-1,000 point.

## ABA AdamW `1e-4` matched point

The direct ABA AdamW screen used micro-batch eight and no accumulation, so its
effective batch, manifest, seed, and 64-batch validation convention exactly
match the formal AdamW curve.  At update 1,000 it produced perplexity
`18.3283349999` after `858.4137` seconds, with `68,225.61MiB` peak allocation.
The formal AdamW result at the same step is `16.5583577720`; therefore
`1e-4` is rejected for AdamW, despite being finite and resource-valid.  The
next narrower but still higher-rate bracket is `5e-5` on the released A100.

## ABA AdamW `5e-5` matched point

The narrowed AdamW screen uses the same batch-eight ABA protocol as formal
AdamW.  It reached update 1,000 with `16.6631990511` perplexity after
`860.1101` seconds and `68,225.61MiB` peak allocation.  This is only
`0.1048412791` above the formal `3e-5` value `16.5583577720`, but it is still
worse; `5e-5` is therefore not selected.  The remaining intermediate high
rate `4e-5` is tested before concluding which AdamW curve is best.

## Local Muown `1e-4` / `6e-6` / `5e-5` matched point

The selected independent-rate Muown screen completed the directly comparable
1,000-update point with `18.3614770010` perplexity after `3,000.5728`
seconds and `8,901.53MiB` peak allocation.  It shares the formal manifest,
seed, effective batch eight, and 64-batch validation convention.  The formal
Muown step-1,000 value is `16.5301853239`; this high-rate configuration is
therefore rejected.  The next independent Muown bracket reduces direction to
`7.5e-5`, gain to `4.5e-6`, and auxiliary rate to `4e-5`.

## ABA Muon `1e-4` / `5e-5` matched point

Muon's ABA screen used micro-batch four and accumulation two, retaining the
formal effective batch of eight while sharing GPU 0 safely.  At update 1,000
it recorded `17.9021746667` perplexity after `1,634.6057` seconds with
`36,183.65MiB` peak allocation.  This is `1.3709739233` above the matched
formal Muon value `16.5312007434`, so matrix `1e-4` with auxiliary `5e-5` is
rejected.  The next Muon bracket is matrix `7.5e-5` and auxiliary `4e-5`,
using batch eight on the released A100 GPU.

## ABA Muon `7.5e-5` / `4e-5` matched point

The batch-eight ABA bracket completed update 1,000 at perplexity
`16.8568076835`, after `936.3392` training seconds and with
`67,445.61MiB` peak allocation.  It shares the fixed token manifest, seed,
effective batch, and 64-batch validation protocol with the formal curve.
It is `0.3256069401` worse than formal Muon at matrix `5e-5` / auxiliary
`3e-5` (`16.5312007434`), so this higher-rate point is rejected.  GPU 0 is
immediately reused for the remaining intermediate `6e-5` / `3.5e-5` bracket;
only that point can determine whether the formal Muon rate remains the best
tested curve.

## ABA AdamW `4e-5` matched point

The last initial AdamW bracket completed with perplexity `16.5539210167` at
step 1,000, after `839.4452` training seconds (`861.4821` total elapsed) and
with `68,225.61MiB` peak allocation.  Its deterministic protocol is identical
to the formal `3e-5` and rejected `5e-5` points.  It improves on formal
`3e-5` (`16.5583577720`) by `0.0044367553`, while `5e-5` was materially worse
at `16.6631990511`.  Thus `4e-5` is the best AdamW point tested so far, but
the three-point local quadratic bracket has its minimum near `3.54e-5`; the
next one-seed refinement is `3.5e-5`, not a premature final-selection claim.

## ABA Muon `6e-5` / `3.5e-5` bracket: metric persisted, checkpoint blocked

The step-1,000 validation completed at perplexity `16.6142332383` after
`934.9520` seconds and `67,445.61MiB` peak allocation.  This is
`0.0830324949` worse than formal `5e-5` / `3e-5` Muon (`16.5312007434`) and
also better than the rejected `7.5e-5` point, confirming the adverse trend
above `5e-5`.  The metric JSON is preserved.  Its subsequent checkpoint write
failed because the ABA filesystem reached zero available bytes; there is no
final-result JSON or resumable checkpoint for this bracket.  The metric is
valid evaluation evidence but not a durable model checkpoint.  After verifying
that results were retained both remotely and locally, four completed rejected
checkpoints and the failed partial file were removed from remote `.cache`,
recovering 13GiB for live trials.

## ABA AdamW `3.5e-5` matched refinement

The refinement completed cleanly at step 1,000 with perplexity
`16.5310959902`, `838.6483` training seconds (`860.8464` total elapsed), and
`68,225.61MiB` peak allocation.  It improves on formal `3e-5` by
`0.0272617818` and on `4e-5` by `0.0228250266`; `5e-5` was already rejected
at `16.6631990511`.  Thus `3.5e-5` is the selected best tested AdamW rate for
this matched one-seed protocol.  A still-finer quadratic vertex is not claimed
from one-seed differences smaller than the demonstrated `3.5e-5` advantage.

## ABA Muown independent-rate `6e-5` / `3.6e-6` / `3.5e-5` matched point

With distinct direction, gain, and auxiliary rates, this ABA batch-eight
bracket reached step 1,000 at perplexity `16.6121615664`, after `965.3384`
training seconds (`987.0773` total elapsed), using `68,228.04MiB` peak
allocation.  It is `0.0819762425` worse than the formal independent-rate
Muown setting (direction `5e-5`, gain `3e-6`, auxiliary `3e-5`) at
`16.5301853239`.  Together with the already-pending local `7.5e-5` branch,
this rules out increasing Muown's direction rate above the formal setting;
the formal rate remains the best tested Muown configuration unless that live
confirmation supplies contrary evidence.
