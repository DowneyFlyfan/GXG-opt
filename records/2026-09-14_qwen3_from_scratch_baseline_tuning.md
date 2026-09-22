# Qwen3-0.6B from-scratch baseline tuning

## Objective and acceptance

Pretrain the Qwen3-0.6B architecture from random weights and tune AdamW
(Adam with decoupled weight decay), Muon, and Muown. Preserve all tuning
traces; select by held-out perplexity under matched data, initialization,
token exposure, effective batch, and schedule. Confirm selected settings for
five epochs on `/home/yufan/New_Optimizer/.cache/Fineweb_Edu_2B`, then produce
perplexity-versus-optimizer-step and perplexity-versus-time PNG figures.
“Best” means best verified among the documented candidates, not a global optimum.

## Diagnosis verified on 2026-09-14

The original `load_qwen3_model` calls `AutoModelForCausalLM.from_pretrained`.
Thus the referenced tied-path plot and its historical baseline curves are
continued pretraining, contrary to the requested from-scratch experiment.
The local historical cache contains 20M training and 5M validation tokens.
The required ABA cache contains 2B training and 100M validation tokens.
The historical evaluation uses shifted next-token labels correctly, restores
training mode, and sums loss over tokens. Worsening validation perplexity
alone does not establish that pretrained initialization is its sole cause.
The old loop also uses a constant learning rate, no clipping, and bfloat16
optimizer parameters/states. These settings are not reused as tuned scratch
hyperparameters.

## Implementation

- Explicit `--initialization scratch` (command-line default) builds the model
  from cached architecture configuration only, never pretrained weights.
- Float32 master parameters and optimizer state with bfloat16 autocast.
- Random seed 1337; initial complete parameter SHA-256 fingerprint and
  step-zero held-out perplexity are persisted.
- Configurable linear warmup, cosine decay, and global gradient clipping.
  Muown direction and gain rates are scaled independently with their ratio retained.
- An explicit data-directory argument supports the required 2B cache without
  replacing the existing historical subset. Resume binds the full configuration
  and data-manifest digest; old pretrained checkpoints cannot resume a scratch trial.
- Historical traces remain historical; they cannot establish scratch performance.

## Resource evidence and first experiment

ABA GPUs 0 and 1 were occupied by live five-epoch pretrained AdamW and Muon
jobs (PIDs 36304 and 36305), approximately 76GB each at full utilization.
They were not interrupted. Local RTX 5070 Ti was free (16GB). Local disk had
318GB free; ABA had 52GB. A two-update memory probe uses micro-batch 1,
accumulation 8, sequence length 2048, activation checkpointing, AdamW peak
rate 0.0003, weight decay 0.1, warmup 100, cosine horizon 2000, minimum ratio
0.1, clipping norm 1. Validation uses 8 micro-batches for the probe only.

The random-initialization regression failed on the old loader and passes on
the new loader. The targeted model/training/data suite passes 32 tests.
Training success and final curve quality remain unverified until actual
runtime metrics are available. Final runs must use matched validation token
counts even when physical micro-batches differ across devices.

## Memory recovery and first screening grid

The dense-logit float32-master probe ran out of device memory in backward.
The full vocabulary projection is now computed in 128-token chunks with
checkpointed recomputation. A tiny real Qwen regression verifies both loss
and every parameter gradient against the dense calculation (33 targeted tests
pass). This changes peak memory and recomputation cost, not the objective.
The retry's step-zero perplexity is 182,914.6258, consistent with random
initialization. Its full parameter digest is
`23fdeb32ff39547b21b0330aed45e4deeeb7673925d6af2f08babaecdc89f1d6`.

The first grid uses the full 2B cache, 300-update screens, batch 1 accumulated
8 times, 64 fixed validation blocks (131,072 target tokens), validation every
100 updates, warmup 100, cosine horizon 2,000, minimum rate ratio 0.1,
clipping norm 1, weight decay 0.1, seed 1337. AdamW rates: 0.0003, 0.001,
0.003. Muon matrix rates: 0.003, 0.01, 0.03, each with auxiliary rate 0.0003.
Muown direction rates: 0.003, 0.01, 0.03; gain and auxiliary rates 0.0003.
This is an initial bracket, not completed tuning. Subsequent work must refine
auxiliary/gain rates, extend promising candidates, and confirm five-epoch
winners. Screening horizons do not count as completed five-epoch pretraining.

## Resource and protocol update (supersedes initial local grid batch)

The local AdamW chunked probe completed at batch 1 / accumulation 8:
2 updates, 14.23 seconds including the periodic checkpoint, 11,441.27 MiB
peak allocation, validation perplexity 182,914.63 -> 162,949.66. Batch 4 /
accumulation 2 also passed (11,526.39 MiB). A local Muown batch-32 memory
probe failed; its partial metrics are not a successful learning curve.

Both ABA devices subsequently became free without intervention from this
session. Checked the process table and empty compute-process inventory before
launching anything. Synchronized the four changed source files from local
checkout to `/home/yufan/New_Optimizer/src/`. ABA has the full requested data
cache already. The A100 batch-16 uncheckpointed-activation probes failed near
79GB. Batch-12 probes are the next measured capacity check.

The remote screen protocol uses micro-batch 12, accumulation 4 (effective
batch 48), 96 fixed validation blocks (196,608 tokens), 300 updates, warmup
100 and cosine horizon 2,000. The previously listed learning-rate grid is
unchanged. All candidates share these settings and the same 2B-token manifest.
Only full effective batches are committed at epoch ends. Screens are run
sequentially per device; AdamW has its own queue, Muon then Muown share the
other queue. Successful screen checkpoints are removed only after result and
complete metric trace persistence; failed logs/checkpoints are retained for
diagnosis. This is needed because ABA has only 53GB free. Final winning model
checkpoints must be retained, and promising screens will be rerun from scratch
at the longer horizon.

## Verified remote launch

Both A100 batch-12 probes completed two committed updates. AdamW peak
allocation was 65,173.69 MiB and Muown 65,173.85 MiB. Their initial parameter
fingerprints match each other and the local probe exactly. Both bind to full
2B-cache manifest `ced8d55d8e0849b22e5c6678194d0ca31fe42998b0760fea8c6d99ab00090649`.
Their metric and result files have been synchronized locally. Local successful
memory-probe checkpoints were removed after preserving these results; they
are not final trained models.

Launched with `nohup` on ABA:
- GPU 0: `scripts/run_qwen3_scratch_screen.py --data-directory /home/yufan/New_Optimizer/.cache/Fineweb_Edu_2B --optimizers adamw --micro-batch-size 12 --gradient-accumulation 4`
- GPU 1: the same command with `--optimizers muon muown`.

Queue logs are `.cache/qwen3_0p6b/logs/scratch_screen_adamw.log` and
`scratch_screen_muon_muown.log`. Per-trial logs include the full run label.
Queues persist summary JSON under `records/2026-09-14_qwen3_scratch_screen_*.json`.
Next: monitor initial 100-update points, handle any instability, retain/sync all
screen results, refine promising parameter brackets, rerun selected candidates
at longer horizons, and then complete matched five-epoch pretraining and plots.
No full scratch curve or best baseline is claimed yet.

## First scratch AdamW 100-update evaluation

AdamW peak learning rate 0.0003 reached update 100 after 725.6142
seconds (12.09 minutes before this evaluation/checkpoint). Validation
perplexity fell from 183,646.8644 at random initialization to 1,076.5817.
Token exposure is 9,830,400; validation covers the same fixed 196,608
target tokens as initialization. The run has finished warmup and continues
toward its 300-update screening horizon. This proves early scratch learning,
not completion of parameter tuning or five-epoch pretraining. Muon had no
matched post-training validation point at this observation.

## Matched scratch AdamW and Muon at 100 updates

Muon matrix rate 0.003 with auxiliary rate 0.0003 reached perplexity
703.4606 after 743.9960 seconds. AdamW rate 0.0003 reached 1,076.5817
after 725.6142 seconds. Both used 9,830,400 training tokens, identical
initial weights, and the same 196,608 validation tokens. Muon is 34.66%
lower in perplexity at this early point. Both have finished warmup and
continue to their common 300-update horizon. This comparison ranks only
these two current configurations at step 100; it does not select the best
optimizer or eliminate untested rates. Muown screening remains queued.

## Runtime projection from measured scratch training

The full 2B-token cache has 976,562 training blocks. Effective batch 48
commits 20,345 updates per epoch and 101,725 updates across five epochs,
exposing 9,999,974,400 training tokens per optimizer. Extrapolating the
first 100-update timings yields approximately 8.54 days for AdamW and
8.76 days for Muon per five-epoch run on one A100. These are early
throughput projections, not promised completion times; sustained speed,
checkpoint/evaluation overhead and GPU availability may change them.
The full objective remains five epochs, with short screens used only to
choose rates. No reduced-data or shortened run counts as final completion.

## Matched scratch AdamW and Muon at 200 updates

Both first-rate candidates improved after warmup. AdamW perplexity went
from 1,076.5817 at step 100 to 482.6559 at step 200; Muon went from
703.4606 to 272.9985. Both exposed 19,660,800 training tokens and used
the unchanged held-out validation blocks. Elapsed times before the current
evaluation/checkpoint were 1,459.8877 and 1,490.5954 seconds respectively.
These are improving scratch curves at the current rates, but the 300-update
screens, alternative rates, Muown comparison and five-epoch confirmations
are still pending. Both live processes were observed writing checkpoints
after their second evaluations; no restart or configuration change was made.

## Step-200 checkpoint I/O delay and recovery

After step 200, Muon PID 99067 temporarily waited in `rq_qos_wait`
while saving its checkpoint. Server I/O pressure reached approximately
40% over the 10-second window; even a filesystem listing was delayed.
The process remained live. A follow-up inspection confirmed atomic
checkpoint completion (5.2 GiB, no partial file), declining I/O pressure,
and both GPUs back at 100% utilization. AdamW checkpoint is 6.7 GiB.
Free disk space remained 41 GiB. This was transient checkpoint I/O, not a
training failure; no process was restarted or interrupted.

## First two completed 300-update screens

Both initial-rate screens completed successfully and their result files and
full traces were synchronized locally. Common training exposure: 29,491,200
tokens. Initialization, data manifest, batch, validation, warmup, decay,
weight decay and clipping configurations match.

| Optimizer | Rate | Step 100 perplexity | Step 200 | Step 300 | Elapsed seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| AdamW | 0.0003 | 1076.5817 | 482.6559 | 299.9560 | 2312.2343 |
| Muon | matrix 0.003, auxiliary 0.0003 | 703.4606 | 272.9985 | 148.0988 | 2355.5711 |

This demonstrates improving scratch curves at both initial settings. It does
not establish final optimizer winners. The queues automatically launched
AdamW 0.001 (PID 117664) and Muon matrix 0.01 / auxiliary 0.0003
(PID 118643), both verified live. Seven initial screens remain incomplete,
including all Muown screens; longer refinements and five-epoch confirmation
remain required. Completed screening checkpoints were removed by the queue
only after persisting their metrics and results, as documented above.

## Higher-rate candidates at 100 updates

At equal exposure of 9,830,400 tokens and the unchanged validation set,
AdamW 0.001 achieved perplexity 831.1094 versus 1,076.5817 at 0.0003.
Muon matrix rate 0.01 with auxiliary 0.0003 achieved 713.2468 versus
703.4606 at matrix 0.003. Thus higher AdamW is better at this early point,
whereas higher Muon is slightly worse. Both higher-rate processes are still
live and continue to 300 updates. Neither early comparison selects a final
rate or justifies interrupting a matched screening run.

## Higher-rate candidates at 200 updates

AdamW 0.001 reached perplexity 394.1015 at step 200, improving on
482.6559 from rate 0.0003. Muon matrix 0.01 / auxiliary 0.0003 reached
290.7169, worse than 272.9985 from matrix rate 0.003. The step-100
rate ordering therefore persists at step 200 for each optimizer. Both
comparisons use identical initial weights, the same data-manifest digest,
19,660,800 exposed training tokens and the fixed validation set. Both
higher-rate trials remain live and continue toward 300 updates. The
interrupted local monitor did not terminate or restart either worker.

## Four completed 300-update screens

The second-rate AdamW and Muon screens both completed successfully.
All four completed candidates share initialization, data fingerprint,
effective batch, validation, warmup, decay, clipping and weight decay.
Each exposed 29,491,200 training tokens.

| Optimizer | Rate | Final screening perplexity |
| --- | --- | ---: |
| AdamW | 0.0003 | 299.9560 |
| AdamW | 0.001 | 238.0775 |
| Muon | matrix 0.003, auxiliary 0.0003 | 148.0988 |
| Muon | matrix 0.01, auxiliary 0.0003 | 170.2140 |

At this screening horizon, AdamW 0.001 improves on 0.0003; Muon 0.003
remains better than 0.01. These are provisional within-optimizer choices,
not final five-epoch selections. Both queues advanced successfully: AdamW
0.003 is running as PID 134396; Muon matrix 0.03 / auxiliary 0.0003
is running as PID 135671. Five initial screens remain, including all
three Muown configurations. No requested final pretraining is complete.

## All three AdamW and Muon rates at 100 updates

The third-rate candidates reached their first matched evaluation. All six
trials have identical initial weights, data fingerprint, step-zero perplexity
and 9,830,400 exposed training tokens at this comparison.

| Optimizer | Rate | Step-100 perplexity |
| --- | --- | ---: |
| AdamW | 0.0003 | 1076.5817 |
| AdamW | 0.001 | 831.1094 |
| AdamW | 0.003 | 859.5329 |
| Muon | matrix 0.003, auxiliary 0.0003 | 703.4606 |
| Muon | matrix 0.01, auxiliary 0.0003 | 713.2468 |
| Muon | matrix 0.03, auxiliary 0.0003 | 787.8572 |

AdamW 0.001 and Muon matrix 0.003 lead their respective early brackets.
The largest-rate trials remain live and continue to 300 updates; early
rankings do not replace full-screen or five-epoch confirmation.

## All three AdamW and Muon rates at 200 updates

At equal exposure of 19,660,800 training tokens, AdamW rate 0.003
reached perplexity 393.8945, essentially tied with 394.1015 at 0.001
and better than 482.6559 at 0.0003. The 0.2070 perplexity difference
between the two higher rates is too small for a meaningful final-rate claim.
Muon matrix 0.03 / auxiliary 0.0003 reached 344.8645, worse than
290.7169 at matrix 0.01 and 272.9985 at 0.003. All six initialization
and data fingerprints match. The largest-rate runs remain live and
continue to the full 300-update screening endpoint; Muown remains queued.

## Prepared Muon lower-rate and auxiliary-rate refinement

Muon matrix 0.003 leads the completed 0.003/0.01 screening bracket;
0.03 is also worse at both measured partial points. Since 0.003 is the
lower tested boundary, it cannot yet be called a bracketed optimum. The
next prepared candidates are matrix/auxiliary rates (0.001, 0.0003),
(0.003, 0.001), and (0.003, 0.0001). Each changes one rate relative to
the existing 0.003/0.0003 reference. Data, seed, effective batch 48,
validation, warmup, cosine horizon, clipping and weight decay stay matched.
The existing queue runner now accepts a candidate JSON and a distinct
campaign name; its default nine-trial behavior is unchanged. The original
dispatch script is preserved as `.cache/qwen3_0p6b/screen_dispatch_v1.py`.
The candidate file is `records/2026-09-14_qwen3_muon_refinement_candidates.json`.
These three refinements are prepared, not launched; use the A100 released
by the AdamW queue after verifying completion and GPU ownership. The other
queue must continue into the three original Muown screens.

## Six completed coarse screens and verified refinement/Muown launch

All coarse AdamW and Muon screens completed at 300 updates.

| Optimizer | Rate | Final screening perplexity |
| --- | --- | ---: |
| AdamW | 0.0003 | 299.9560 |
| AdamW | 0.001 | 238.0775 |
| AdamW | 0.003 | 234.6931 |
| Muon | matrix 0.003, auxiliary 0.0003 | 148.0988 |
| Muon | matrix 0.01, auxiliary 0.0003 | 170.2140 |
| Muon | matrix 0.03, auxiliary 0.0003 | 208.9775 |

AdamW 0.003 leads this short screen only narrowly over 0.001; both need
longer comparison before final selection. Muon 0.003 is the best coarse
matrix rate, and lower-matrix/auxiliary refinement is needed to resolve
its boundary optimum. No five-epoch baseline is complete.

After confirming AdamW worker and parent exit and an empty GPU-0 compute
process inventory, launched `--campaign muon_refine --optimizers muon
--candidates records/2026-09-14_qwen3_muon_refinement_candidates.json`
with the unchanged matched protocol. Dispatcher PID 152903, first worker
152904 (matrix 0.001 / auxiliary 0.0003), log
`.cache/qwen3_0p6b/logs/scratch_muon_refine.log`.

The original GPU-1 queue advanced to Muown direction 0.003 / gain 0.0003 /
auxiliary 0.0003 as PID 154470. Both new workers were confirmed in the GPU
process inventory, and their persisted initialization/data fingerprints
match all six completed screens. Three original Muown screens and three
Muon refinements are now active or queued, followed by longer confirmations.

## First Muown and lower-matrix Muon evaluations

At 100 updates (9,830,400 exposed training tokens), Muon matrix 0.001 /
auxiliary 0.0003 reached perplexity 786.1812, worse than 703.4606 at
matrix 0.003. Together with the higher-rate screens, this provides an
early lower-side bracket around the provisional 0.003 matrix rate.
Muown direction 0.003 / gain 0.0003 / auxiliary 0.0003 reached 721.1427,
close to but worse than matched Muon 0.003 at this point. All initial
parameter and data fingerprints match. Both live trials continue to
300 updates; no final rate or optimizer selection follows from step 100.

## First Muown and lower-matrix Muon at 200 updates

At identical exposure of 19,660,800 training tokens, Muown direction 0.003 /
gain 0.0003 / auxiliary 0.0003 reached perplexity 261.2686, compared with
272.9985 for Muon matrix 0.003 / auxiliary 0.0003. Lowering the Muon
matrix rate to 0.001 gave 317.0351. Initial parameter and dataset hashes
match across all three traces. This reverses the small Muown/Muon ordering
at step 100, reinforcing the need for longer comparison. Both current
workers were live at this check and continue to 300 updates; final five-epoch
pretraining and optimizer selection remain pending.

## Prepared longer AdamW confirmation bracket

Prepared candidate rates 0.001, 0.003 and 0.006 in
`records/2026-09-14_qwen3_adamw_confirmation_candidates.json`. The first
two were almost tied at 300 updates; 0.006 checks the unexplored upper
side of the current boundary winner. Run each from the common scratch
seed for 1,000 updates (98,304,000 tokens), using the existing fixed
100-update warmup and 2,000-update cosine horizon, batch 48, and fixed
validation subset. This is a longer tuning screen, not five-epoch training.
The final training schedule must instead span the full 101,725 updates.
Candidates are prepared only; no additional worker was launched while
both devices remain occupied by the existing queues.

## Completed lower-matrix Muon refinement

Muon matrix 0.001 / auxiliary 0.0003 completed all 300 screening updates
with perplexity 174.2374, worse than 148.0988 at matrix 0.003. The
initialization and data fingerprints match. Among tested matrix rates
0.001, 0.003, 0.01 and 0.03, 0.003 is now an interior short-screen
winner with the auxiliary rate held at 0.0003. This is not proof of a
long-run optimum. Dispatcher 152903 advanced normally to worker 170477,
Muon matrix 0.003 / auxiliary 0.001. The first Muown trial remains active.

## First completed Muown screen and eight-trial audit

Muown direction 0.003 / gain 0.0003 / auxiliary 0.0003 completed 300
updates with perplexity 131.7712 and elapsed time 2330.5638 seconds.
Its final result persisted and dispatcher 99066 advanced to worker 172098,
direction 0.01 with gain/auxiliary rates unchanged. The current Muon
auxiliary-0.001 worker is 170477. Eight completed screens (three AdamW,
four Muon, one Muown) have matching initial parameter/data hashes,
effective batch 48, validation 96 blocks, and finite endpoint perplexity.
The best observed short endpoints are AdamW 234.6931, Muon 148.0988,
and Muown 131.7712. These do not establish final five-epoch rankings.
Full trace/result artifacts and the eight-screen audit are retained locally.

## Higher Muon auxiliary and Muown direction rates at 100 updates

At equal exposure of 9,830,400 tokens, Muon matrix 0.003 / auxiliary
0.001 reached perplexity 559.2787, improving on 703.4606 at auxiliary
0.0003. Muown direction 0.01 / gain 0.0003 / auxiliary 0.0003 reached
657.8540, improving on 721.1427 at direction 0.003. All four traces
have identical initialization/data fingerprints. The two new workers
170477 and 172098 remain live and continue to 300 updates. Early
improvements do not yet determine longer-run parameter selection.

## Higher Muon auxiliary and Muown direction rates at 200 updates

At equal exposure of 19,660,800 training tokens, Muon matrix 0.003 /
auxiliary 0.001 reached perplexity 204.1519, improving on 272.9985
at auxiliary 0.0003. Muown direction 0.01 / gain 0.0003 / auxiliary
0.0003 reached 251.6191, slightly better than 261.2686 at direction
0.003. All four traces share initialization/data fingerprints. Workers
170477 and 172098 remain live and continue to the 300-update endpoint.
The larger Muon auxiliary rate is promising but remains a short-screen
observation; final rates and five-epoch outcomes are not established.

## Ten completed screens: larger auxiliary rate improves Muon

Muon matrix 0.003 / auxiliary 0.001 completed 300 updates at perplexity
104.7498, improving substantially over 148.0988 at auxiliary 0.0003.
The queue advanced to worker 188425, matrix 0.003 / auxiliary 0.0001.
Muown direction 0.01 / gain 0.0003 / auxiliary 0.0003 completed at
131.0751 versus 131.7712 at direction 0.003, a small difference that
does not settle long-run rate choice. Its queue advanced to worker
190238, direction 0.03 with other rates unchanged. Ten complete
short screens now have matching initialization/data hashes, batch 48,
validation 96 blocks, and finite endpoints; their results and full traces
are synchronized locally. Best observed endpoints are AdamW 234.6931,
Muon 104.7498 and Muown 131.0751. These rankings are provisional and
all five-epoch formal runs remain pending.

## Prepared upper auxiliary-rate bracket for Muon

The completed auxiliary-0.001 Muon screen is substantially better than
0.0003, but is still at the upper tested boundary. Prepared two further
300-update candidates at matrix rate 0.003 and auxiliary rates 0.003
and 0.01, preserving all other screen settings. Candidate file:
`records/2026-09-14_qwen3_muon_auxiliary_upper_candidates.json`; use
campaign `muon_aux_upper`. These are prepared only, for available
capacity after current screens finish. The longer AdamW candidate
file remains ready for the confirmation stage. Current workers 188425
and 190238 were verified in the accelerator process inventory with the
same scratch initialization/data hashes as previous trials.

## Lower Muon auxiliary and highest coarse Muown rate at 100 updates

At 100 matched updates, Muon matrix 0.003 / auxiliary 0.0001 reached
perplexity 1113.9546, worse than 703.4606 at auxiliary 0.0003 and
559.2787 at auxiliary 0.001. Muown direction 0.03 / gain 0.0003 /
auxiliary 0.0003 reached 663.0015, close to but worse than 657.8540
at direction 0.01 and better than 721.1427 at direction 0.003.
All six traces have matching initialization/data fingerprints and exposure
9,830,400 tokens. Workers 188425 and 190238 remain live and continue
to 300 updates. These are intermediate observations, not final choices.

## Lower Muon auxiliary and highest coarse Muown rate at 200 updates

At 200 matched updates, Muon matrix 0.003 / auxiliary 0.0001 reached
perplexity 404.1594, worse than 272.9985 at auxiliary 0.0003 and
204.1519 at auxiliary 0.001. Muown direction 0.03 / gain 0.0003 /
auxiliary 0.0003 reached 255.5897, slightly worse than 251.6191
at direction 0.01 and better than 261.2686 at direction 0.003.
All six traces share initialization/data hashes and exposure 19,660,800
tokens. Both workers remain live and continue to 300 updates. The
Muown direction-rate differences remain small; selection needs the
completed screens and longer confirmations.

## Twelve initial screens complete; next refinement queues launched

All nine coarse screens and three Muon refinements completed successfully.
Muon auxiliary 0.0001 finished at 236.1651, worse than 148.0988 at
0.0003 and 104.7498 at 0.001, with matrix rate 0.003 held fixed.
Muown direction 0.03 finished at 132.6203; 0.01 remains narrowly best
at 131.0751 versus 131.7712 for 0.003. All twelve result/trace sets
are synchronized locally and audited for scratch initialization, identical
parameter/data hashes, 300 updates, effective batch 48, 96 validation
blocks, and finite perplexity. None is a completed five-epoch run.

After verifying old dispatcher exit and empty device process inventories,
launched two new matched 300-update campaigns. GPU 0: `muon_aux_upper`,
auxiliary 0.003 then 0.01 with matrix 0.003; dispatcher 208551 and
first worker 208552. Its scratch initialization was verified. GPU 1:
`muown_gain_aux`, direction 0.01 and (gain, auxiliary) combinations
(0.001, 0.0003), (0.0003, 0.001), (0.001, 0.001); dispatcher 211010.
This separates each rate change and then tests their combination.
Free disk was 53 GB before the second launch. The longer AdamW
confirmation candidates remain prepared for the next available capacity.

## Muon auxiliary 0.003 at 100 updates

With matrix rate fixed at 0.003, auxiliary 0.003 reached perplexity
533.2561 at 100 updates, compared with 559.2787 at auxiliary 0.001
and 703.4606 at 0.0003. Initialization/data hashes and exposure
9,830,400 tokens match. Worker 208552 remains live and continues to
300 updates. Muown gain-0.001 worker 211011 remains live but its first
evaluation is pending. This early improvement alone does not establish
the final auxiliary rate.

## Muown gain 0.001 at 100 updates

With direction 0.01 and auxiliary 0.0003 fixed, increasing gain rate
from 0.0003 to 0.001 gave perplexity 659.2085 at 100 updates, nearly
tied with but slightly worse than 657.8540 at the reference gain rate.
Initialization/data hashes and exposure 9,830,400 tokens match. Worker
211011 remains live and continues to 300 updates; worker 208552 also
remains active. The gain-only increase has no measured early benefit.

## Muon auxiliary 0.003 at 200 updates

With matrix rate fixed at 0.003, auxiliary 0.003 reached perplexity
207.6774 at 200 updates, now slightly worse than 204.1519 at auxiliary
0.001 and better than 272.9985 at 0.0003. The ordering of 0.001
and 0.003 reversed since step 100, so neither early point establishes
a final rate. Initial parameter/data hashes and exposure 19,660,800
tokens match. Worker 208552 remains live and continues to 300 updates;
Muown gain worker 211011 remains live with step 200 still pending.

## Muown gain 0.001 at 200 updates

With direction 0.01 and auxiliary 0.0003 fixed, gain 0.001 gave
perplexity 251.5819 at 200 updates versus 251.6191 at gain 0.0003.
The difference is negligible for parameter selection. Initialization/data
hashes and exposure 19,660,800 tokens match. Both workers remain
live and continue to 300 updates; auxiliary-only and combined-rate
Muown candidates remain queued.

## Completed Muon auxiliary 0.003 screen

Muon matrix 0.003 / auxiliary 0.003 completed 300 updates at perplexity
103.8163, only narrowly better than 104.7498 at auxiliary 0.001. The
ordering changed between intermediate evaluations, so both settings
merit longer comparison before selecting the five-epoch configuration.
All thirteen completed screens share initialization/data fingerprints
and have finite endpoint perplexity. Results and traces are synchronized
locally. The upper-auxiliary queue advanced to its 0.01 candidate;
Muown gain worker 211011 remains active at the latest check.

## Completed Muown gain-only refinement

Muown direction 0.01 / gain 0.001 / auxiliary 0.0003 completed
300 updates at perplexity 130.7956, versus 131.0751 with gain
0.0003. The gain-only effect is small across all three evaluations.
The result persisted and dispatcher 211010 advanced to the auxiliary-only
candidate (direction 0.01, gain 0.0003, auxiliary 0.001). Muon upper
auxiliary-0.01 worker 301657 remains active. Fourteen completed
screens are retained locally with matching initialization/data hashes
and finite endpoint metrics. Full five-epoch runs remain pending.

## Local capacity fallback and longer AdamW confirmations launched

Both ABA A100s are occupied by authorized Muon/Muown refinement queues.
The local RTX 5070 Ti was verified idle with no compute processes, so
used the project-approved local fallback to overlap longer AdamW tuning.
The first background launcher exited before creating metrics; verified
its absence and empty artifacts, then used a tracked foreground nohup
session. A full-data micro-batch 12 / accumulation 4 / activation-checkpointed
probe failed with device-memory exhaustion in backward. Memory was released.
Micro-batch 8 / accumulation 6 passed two optimizer updates: peak allocation
13,271.77 MiB, elapsed 68.90 seconds including evaluation/checkpoint work,
final perplexity 114206.76, and matching parameter/data hashes. Its successful
probe checkpoint was deleted after retaining full metrics/result; failed
probe evidence remains retained.

Launched campaign `adamw_confirm_local` at 1,000 updates per rate (0.001,
0.003, 0.006), dispatcher 3704282 / first worker 3704283, tracked session
53452. Effective batch stays 48; validation stays 96 blocks. The local
runs use activation checkpointing and micro-batch 8, unlike A100 micro-batch
12 without activation checkpointing. All three AdamW confirmation candidates
share this local protocol. Hardware/micro-batching differences must remain
explicit, and their elapsed times must not be interpreted as cross-optimizer
A100 speed comparisons. The same full 2B-token data, seed 1337, warmup 100,
cosine horizon 2000, clipping 1, weight decay 0.1, and float32 master weights
are retained. Each confirmation exposes 98,304,000 training tokens and is
still tuning, not a five-epoch result. Local free disk was 310 GB.

## Muon auxiliary 0.01 at 100 updates

Muon matrix 0.003 / auxiliary 0.01 reached perplexity 609.8365 at
100 updates, worse than 533.2561 at auxiliary 0.003 and 559.2787
at 0.001. Initialization/data fingerprints and exposure 9,830,400
tokens match. Worker 301657 remains live and continues to 300 updates.
The local AdamW confirmation and Muown auxiliary-only worker were also
verified live during this interval. Final choices remain pending.

## Muown auxiliary 0.001 at 100 updates

With direction 0.01 and gain 0.0003 fixed, auxiliary 0.001 reached
perplexity 511.1354 at 100 updates, improving on 657.8540 at auxiliary
0.0003. Initialization/data hashes and exposure 9,830,400 tokens match.
The auxiliary-only change has a much larger early effect than the gain-only
change, but worker 304061 must continue to 300 updates before endpoint
selection. Muon worker 301657 and local AdamW worker 3704283 were also
verified live in this monitoring interval.

## Prepared 1000-update Muon confirmations

Prepared matched 1,000-update A100 confirmations at matrix 0.003 and
auxiliary rates 0.001 and 0.003, whose completed 300-update endpoints
are 104.7498 and 103.8163. Use the existing screen runner with campaign
`muon_confirm`, updates 1000, micro-batch 12 / accumulation 4, and
`records/2026-09-14_qwen3_muon_confirmation_candidates.json`. The same
warmup 100 and cosine horizon 2000 keep the first 300 updates comparable
to the screens. Launch after the current auxiliary-0.01 candidate completes
and its device is verified free; if its endpoint beats both, revise the
confirmation shortlist from that new evidence first. These confirmations
are prepared, not launched. Five-epoch training still requires its full
101,725-update schedule and cannot use this screening horizon.

## Muon auxiliary 0.01 at 200 updates

Muon matrix 0.003 / auxiliary 0.01 reached perplexity 242.2285 at
200 updates, worse than 207.6774 at auxiliary 0.003 and 204.1519
at 0.001. The initial parameter/data hashes and exposure 19,660,800
tokens match. The planned 0.001/0.003 longer-confirmation shortlist
remains supported by current evidence, subject to the completed endpoint.
Worker 301657 remains live and continues to 300 updates; Muown worker
304061 remains active with its step-200 evaluation pending.

## Muown auxiliary 0.001 at 200 updates

With direction 0.01 and gain 0.0003 fixed, auxiliary 0.001 reached
perplexity 173.0112 at 200 updates, improving on 251.6191 at auxiliary
0.0003. Initial parameter/data hashes and exposure 19,660,800 tokens
match. The improvement persists at the second evaluation; worker 304061
continues to 300 updates before endpoint selection. Muon worker 301657
and local AdamW worker 3704283 were also confirmed active during this
interval. Five-epoch outcomes remain unmeasured.

## Muon short screens complete; longer confirmations launched

Muon matrix 0.003 / auxiliary 0.01 completed 300 updates at perplexity
118.4919, worse than 103.8163 at auxiliary 0.003 and 104.7498 at
0.001. All fifteen completed 300-update screens are synchronized locally,
with matching initialization/data hashes and finite endpoints. Verified
upper-auxiliary dispatcher exit, successful completion of both candidates,
and empty GPU-0 compute inventory before launching `muon_confirm`:
1,000 updates each, matrix 0.003, auxiliary 0.001 then 0.003,
micro-batch 12 / accumulation 4. Dispatcher 319935, first worker
319936, log `.cache/qwen3_0p6b/logs/scratch_muon_confirm.log`.
Free remote disk was 46 GB. This extends the tuning comparison; it
is not five-epoch pretraining. Muown auxiliary-only worker 304061 and
local AdamW confirmation worker 3704283 remain active.

## Completed Muown auxiliary-only refinement

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 completed
300 updates at perplexity 95.4584, improving on 131.0751 at auxiliary
0.0003. The improvement persists across all three saved evaluations.
The result persisted and dispatcher 211010 advanced to the combined-rate
candidate (direction 0.01, gain 0.001, auxiliary 0.001). Sixteen
completed screens are synchronized locally with matching initial weights
and dataset fingerprints and finite endpoints. Longer Muon and AdamW
comparisons remain active; these short-screen results do not establish
final five-epoch rankings.

## Prepared longer Muown auxiliary-rate bracket

The completed auxiliary-only Muown result (95.4584) is substantially
better than auxiliary 0.0003, but 0.001 remains its largest tested
auxiliary rate. Prepared 1,000-update candidates at direction 0.01, gain
0.0003, and auxiliary 0.001, 0.003, 0.01 in
`records/2026-09-14_qwen3_muown_confirmation_candidates.json`. These
combine longer confirmation with checking the upper auxiliary-rate bracket,
rather than adding another separate 300-update campaign. Use campaign
`muown_confirm`, A100 micro-batch 12 / accumulation 4, and unchanged
100-update warmup / 2000-update cosine horizon. Launch only after the
current combined gain/auxiliary screen completes and its device is free;
revise the gain choice if that completed result shows a material gain
benefit. Prepared only, not launched.

## First longer AdamW local validation

The local AdamW learning-rate-0.001 confirmation reached step 100
(9,830,400 tokens) at perplexity 851.9674. Initial weight and dataset
hashes match the A100 screen, whose corresponding perplexity was
831.1094. Hardware, micro-batching and activation checkpointing differ,
so this is a distinct confirmation cohort; compare its three rates
within that common local protocol and do not pool hardware timing.
Worker 3704283 remains live and continues to 1,000 updates. Both A100
workers were also verified live; their current-run first evaluations
are still pending.

## Longer Muon first evaluation and repeat-variation audit

The 1,000-update Muon confirmation reached step 100 at perplexity
539.4958 versus 559.2787 in the earlier 300-update screen, despite
matching initial weights/data hashes and identical initial perplexity.
Read-only audit against the original environment record verified all five
core source hashes, model config hash, and torch/transformers/numpy/
safetensors versions. Parsing the live process command showed only
run_label and maximum_updates differ from the earlier saved trial config.
The data loader uses an explicit seeded generator and optimizer parameter
ordering is sorted in the inspected source. These checks do not isolate
the cause of the approximately 3.5 percent endpoint variation, and exact
reproducibility must not be claimed. Small screening differences are
inconclusive; retain both close Muon finalists for longer comparison.
No live process was interrupted or settings changed during this audit.

## Combined Muown rates at 100 updates

Muown direction 0.01 / gain 0.001 / auxiliary 0.001 reached
perplexity 504.6505 at 100 updates, versus 511.1354 with the same
direction/auxiliary rates and gain 0.0003. Initial weights/data hashes
and exposure 9,830,400 tokens match. This small single-run gap is
not sufficient to establish a gain-rate advantage, especially given the
observed repeat variation. Worker 322457 continues to 300 updates;
longer Muon and AdamW workers remain active.

## Longer Muon confirmation at 200 updates

The Muon matrix-0.003 / auxiliary-0.001 confirmation reached perplexity
200.8774 at 200 updates, down from 539.4958 at 100. The earlier
300-update screen gave 204.1519 at the same exposure; repeat variation
remains visible, with matching initialization/data fingerprints. Worker
319936 continues toward 1,000 updates. The combined Muown worker
322457 remains live with its next evaluation pending; local AdamW
worker 3704283 was also verified live during this interval.

## Combined Muown rates at 200 updates

Muown direction 0.01 / gain 0.001 / auxiliary 0.001 reached
perplexity 174.8324 at 200 updates, versus 173.0112 with gain
0.0003. The small gain-rate ordering reversed from step 100 and
does not establish an advantage for the larger gain. Initial weights/
data hashes and exposure 19,660,800 tokens match. Worker 322457
continues to 300 updates; the planned longer auxiliary-rate bracket
with gain 0.0003 remains appropriate pending the completed endpoint.

## Longer Muon confirmation reaches original screening horizon

The Muon matrix-0.003 / auxiliary-0.001 confirmation reached step 300
at perplexity 104.5165, down from 200.8774 at step 200. The earlier
short screen ended at 104.7498 with matching initialization/data hashes
and exposure 29,491,200 tokens. Their endpoints are close despite the
larger step-100 repeat difference. Worker 319936 remains live and now
continues beyond the original screening horizon toward 1,000 updates;
this metric does not indicate completion of the longer trial. Muown
combined-rate worker 322457 remains active at its latest step-200 check.

## Seventeen short screens complete; Muown confirmations launched

The combined Muown setting (direction 0.01, gain 0.001, auxiliary
0.001) completed 300 updates at perplexity 95.3542, essentially tied
with 95.4584 at gain 0.0003. The gain ordering changed across evaluation
points, so there is no consistent material advantage for the larger gain.
Retained gain 0.0003 for the longer auxiliary-rate bracket. All seventeen
short-screen result/trace sets are synchronized locally and verified for
matching initialization/data hashes, batch 48, validation 96 blocks,
300 updates, and finite endpoints.

After verifying all three final queue entries succeeded and the GPU-1
compute inventory was empty, launched campaign `muown_confirm`: 1,000
updates each at direction 0.01 / gain 0.0003 and auxiliary 0.001,
0.003, 0.01. Dispatcher 341415, log
`.cache/qwen3_0p6b/logs/scratch_muown_confirm.log`. Free disk was 47 GB.
The old dispatcher still existed at the first inventory snapshot while
finishing its queue; no additional candidate was pending in that queue.
Longer AdamW and Muon workers remain active. All three optimizer families
are now in longer tuning; no five-epoch formal run is complete or launched.

## Local AdamW confirmation at 200 updates

The local AdamW learning-rate-0.001 run reached perplexity 400.4505
at 200 updates, improving from 851.9674 at 100. Training exposure
is 19,660,800 tokens; elapsed time is 4360.7892 seconds. Worker
3704283 continues toward 1,000 updates. Hardware-separated timing
and within-local-cohort rate comparisons remain required.

## Muon improves beyond the short-screen horizon

The longer Muon run reached perplexity 76.7972 at 400 updates
(39,321,600 training tokens), improving from 104.5165 at step 300.
This is the first saved point beyond the original screening horizon.
Worker 319936 remains live and continues to 1,000 updates; Muown
confirmation worker 341440 is also live with its first evaluation pending.
Local AdamW worker 3704283 was verified active during this interval.
The continued decrease is measured evidence, not a completed five-epoch
result or a final cross-optimizer ranking.

## Longer Muown confirmation at 100 updates

The Muown direction-0.01 / gain-0.0003 / auxiliary-0.001 confirmation
reached perplexity 498.7204 at 100 updates, compared with 511.1354
in the earlier short screen. Initial parameter/data hashes and exposure
9,830,400 tokens match. As with the Muon repeat, small early differences
do not establish a rate advantage. Worker 341440 remains live and
continues toward 1,000 updates; the other longer runs remain active.

## Longer Muon confirmation at 500 updates

The matrix-0.003 / auxiliary-0.001 Muon run reached perplexity
65.1331 at 500 updates (49,152,000 training tokens), improving from
76.7972 at step 400. Initialization and data fingerprints match the
shared scratch protocol. Worker 319936 is live and halfway through its
1,000-update confirmation; this is not a completed five-epoch result.
Muown worker 341440 is also live with its next evaluation pending.

## Longer Muown confirmation at 200 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 179.3254 at step 200, down from 498.7204 at step 100.
Training exposure is 19,660,800 tokens, with matching initialization
and data fingerprints. The earlier 300-update screen measured 173.0112
at step 200; observed repeat variation still limits small-gap claims.
Worker 341440 remains live and continues to 1,000 updates. Full
five-epoch pretraining and final baseline selection remain pending.

## Longer Muon confirmation at 600 updates

Muon matrix 0.003 / auxiliary 0.001 reached perplexity 58.2521
at 600 updates, improving from 65.1331 at step 500. Exposure is
58,982,400 training tokens; initialization and data fingerprints match.
Worker 319936 remains live and continues toward 1,000 updates.
This confirms continued improvement beyond the short-screen horizon,
while the competing rate and five-epoch formal run remain unfinished.

## Local AdamW confirmation at 300 updates

Local AdamW learning rate 0.001 reached perplexity 240.7337 at
300 updates, down from 400.4505 at step 200. Training exposure is
29,491,200 tokens, with matching initial weights and data fingerprints.
Elapsed time was 6544.3456 seconds. The earlier A100 screen ended at
238.0775; hardware, microbatch and checkpointing differ, so timing must
stay separate and rate selection uses the within-local comparison.
Worker 3704283 remains live and continues to 1,000 updates.

## Longer Muown confirmation reaches 300 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 96.3512 at 300 updates, improving from 179.3254 at
step 200. The earlier short trial ended at 95.4584; this repeat is
close at the original screening horizon. Exposure is 29,491,200 tokens
with matching initial weights and data fingerprints. Worker 341440
remains live and now continues beyond that horizon toward 1,000 updates.
Final parameter selection and five-epoch pretraining remain pending.

## Longer Muon confirmation at 700 updates

Muon matrix 0.003 / auxiliary 0.001 reached perplexity 53.1417
at step 700, improving from 58.2521 at step 600. Exposure is
68,812,800 training tokens, with matching initialization/data hashes.
Worker 319936 remains live and continues toward 1,000 updates.
The alternative auxiliary-rate confirmation is still queued, so final
rate selection and five-epoch pretraining remain pending.

## Muown improves beyond the short-screen horizon

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 72.8414 at step 400, improving from 96.3512 at step 300.
Exposure is 39,321,600 training tokens, with matching initial weights
and data fingerprints. Both A100s were verified at full utilization;
worker 341440 remains active and continues toward 1,000 updates.
This is the first saved Muown point beyond the short-screen horizon,
not a completed longer-rate comparison or five-epoch result.

## Longer Muon confirmation at 800 updates

Muon matrix 0.003 / auxiliary 0.001 reached perplexity 49.2899
at step 800, improving from 53.1417 at step 700. Exposure is
78,643,200 training tokens, with matching initial weights and data
fingerprints. Worker 319936 remains live with 200 updates left in
this confirmation. The auxiliary-0.003 trial is still queued; no
final rate selection or five-epoch completion is claimed.

## Longer Muown confirmation at 500 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 62.6343 at step 500, improving from 72.8414 at step 400.
Exposure is 49,152,000 training tokens, with matching initialization
and data fingerprints. Worker 341440 remains live and is halfway
through this 1,000-update confirmation. Auxiliary rates 0.003 and
0.01 remain queued; final selection and five-epoch runs are pending.

## Longer Muon confirmation at 900 updates

Muon matrix 0.003 / auxiliary 0.001 reached perplexity 46.1721
at step 900, improving from 49.2899 at step 800. Exposure is
88,473,600 training tokens, with matching initial weights and data
fingerprints. Worker 319936 remains active, with 100 updates left
in this trial. The competing auxiliary-0.003 trial is queued next;
full five-epoch training has not started.

## Local AdamW improves beyond the short-screen horizon

Local AdamW learning rate 0.001 reached perplexity 162.5986 at
400 updates, improving from 240.7337 at step 300. Exposure is
39,321,600 tokens; initial weights and data fingerprints match.
Elapsed time was 8727.2951 seconds. Worker 3704283 remains live
and continues toward 1,000 updates. The two higher-rate local trials
remain queued; hardware-separated timing and within-cohort rate
comparisons are still required. Five-epoch formal runs remain pending.

## Longer Muown confirmation at 600 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 55.6538 at step 600, improving from 62.6343 at step 500.
Exposure is 58,982,400 training tokens, with matching initial weights
and data fingerprints. Worker 341440 remains active and continues
toward 1,000 updates. Competing rates and full five-epoch pretraining
remain pending.

## First 1,000-update Muon confirmation completed

The matrix-0.003 / auxiliary-0.001 Muon trial completed successfully
with perplexity 43.8401 after 1,000 updates and 98,304,000 training
tokens. All eleven evaluations from step 0 through 1000 are finite,
strictly decreasing, and share the expected data and initialization
fingerprints. Saved elapsed time is 7636.5699 seconds. The dispatcher
record reports exit code zero; result and complete trace are synced
locally. Completed epochs is zero, so this is a tuning result only.

Dispatcher 319935 launched worker 380878 for matrix 0.003 / auxiliary
0.003, using the same A100 protocol and 1,000-update horizon. Initial
metrics were not yet written at the first launch check. Muown and
local AdamW remain active. Final Muon rate selection waits for this
second confirmation; full five-epoch formal runs remain pending.

Follow-up launch verification: the auxiliary-0.003 Muon trial wrote
step 0 with perplexity 183646.8644 and the exact expected initial
weight/data hashes. Worker 380878 is live.

## Longer Muown confirmation at 700 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 50.9258 at step 700, improving from 55.6538 at step 600.
Exposure is 68,812,800 tokens, with matching initial weights and data
fingerprints. Worker 341440 remains live and continues toward
1,000 updates; the two higher auxiliary-rate trials remain queued.
The final rate comparison and five-epoch runs remain pending.

## Second Muon confirmation at 100 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 500.7792
at step 100, versus 539.4958 for auxiliary 0.001 in the first
longer trial. Exposure is 9,830,400 tokens, with matching initial
weights, data and A100 protocol. This early difference does not
establish the final rate choice, given prior repeat variation and
incomplete horizon coverage. Worker 380878 remains active and
continues toward 1,000 updates.

## Longer Muown confirmation at 800 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 47.2947 at step 800, improving from 50.9258 at step 700.
Exposure is 78,643,200 tokens, with matching initial weights and data
fingerprints. Worker 341440 remains active with 200 updates left in
this trial. The auxiliary-0.003 and auxiliary-0.01 trials remain
queued; final settings and full five-epoch pretraining are pending.

## Second Muon at 200 updates and local AdamW at 500

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 214.6102
at step 200, down from 500.7792 at step 100. Auxiliary 0.001
reached 200.8774 at step 200; the rate ordering has reversed from
step 100. This supports waiting for the full longer comparison.
Worker 380878 remains live.

Local AdamW learning rate 0.001 reached perplexity 115.0086 at
step 500, improving from 162.5986 at step 400. Exposure is
49,152,000 tokens and elapsed time 10910.1838 seconds. Worker
3704283 remains live and is halfway through this trial. Both new
metrics pass initial-weight/data fingerprint and exposure checks.
Local timing remains separate from A100 timing; final parameter
selection and five-epoch formal runs remain pending.

Sync correction: downloading all scratch traces copied the older ABA
backup over the live local AdamW trace, removing its step-500 line.
The exact complete JSON line was captured in tool output before the
sync and restored verbatim in content, then verified with steps
0,100,200,300,400,500. The trainer opens the metric file for each
append, so it does not retain an unlinked file handle; the worker
remained active. Future remote-to-local metric sync MUST exclude
`*confirm_local*`; local-run metrics flow only local-to-remote.

## Longer Muown confirmation at 900 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 reached
perplexity 44.3721 at step 900, improving from 47.2947 at step 800.
Exposure is 88,473,600 tokens, with matching initial weights and data
fingerprints. Worker 341440 remains live with 100 updates left in
this confirmation; two higher auxiliary-rate trials remain queued.
The local AdamW trace is excluded from the remote download. Final
settings and five-epoch formal runs remain pending.

## Second Muon confirmation reaches the short-screen horizon

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 104.4865
at step 300, down from 214.6102 at step 200. The completed
auxiliary-0.001 confirmation had 104.5165 at step 300, effectively
a tie given the observed repeat variation. Exposure is 29,491,200
tokens and the initialization/data fingerprints match. Worker 380878
remains active and continues for 700 more updates. Final rate
selection awaits the full matched horizon; five-epoch runs are pending.

## First 1,000-update Muown confirmation completed

Muown direction 0.01 / gain 0.0003 / auxiliary 0.001 completed
1,000 updates successfully at perplexity 42.2124, with elapsed time
7904.8720 seconds and 98,304,000 training tokens. The dispatcher
reports exit code zero. All eleven evaluations are finite, strictly
decreasing, and use the expected initialization/data fingerprints.
Result and full trace are synchronized locally. Completed epochs
is zero; this is a tuning result, not five-epoch completion.

Dispatcher 341415 launched worker 480344 for auxiliary rate 0.003
with direction 0.01 and gain 0.0003 unchanged. Its saved step-zero
evaluation matches the exact initial weight/data hashes and perplexity
183646.8644. It remains live under the same A100 protocol; auxiliary
0.01 is still queued. Muon and local AdamW continue. No final rate
selection or five-epoch formal run is complete.

## Second Muon confirmation at 400 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 74.3325
at step 400, down from 104.4865 at step 300. Auxiliary 0.001
reached 76.7972 at step 400. Exposure is 39,321,600 tokens and
the initial weight/data fingerprints match. Both longer curves are
improving; the small current gap does not finalize the rate choice.
Worker 380878 remains active toward 1,000 updates. New Muown
worker 480344 was verified at full A100 utilization; local AdamW
continues. Full five-epoch runs remain pending.

## Second Muown at 100 updates and local AdamW at 600

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 465.4763 at step 100, compared with 498.7204 for
auxiliary 0.001 at the same exposure of 9,830,400 tokens. This
early comparison is not a final rate choice; worker 480344 remains
active toward 1,000 updates.

Local AdamW learning rate 0.001 reached perplexity 92.9861 at
step 600, improving from 115.0086 at step 500. Exposure is
58,982,400 tokens and elapsed time 13093.4214 seconds. The
trainer appended the new row normally after the earlier sync repair;
the trace now contains every scheduled step from 0 through 600.
Both runs pass initialization/data fingerprint and exposure checks.
Local timing remains separate from A100 timing. Five-epoch formal
runs and final rate selection remain pending.

## Second Muon confirmation at 500 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 63.7115
at step 500, improving from 74.3325 at step 400. Auxiliary 0.001
reached 65.1331 at the same step. Exposure is 49,152,000 tokens
with matching initial weights and data fingerprints. Worker 380878
remains live and is halfway through this trial. The full matched
1,000-update comparison is still pending, as are the formal five
epochs.

## Second Muown confirmation at 200 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 167.4201 at step 200, down from 465.4763 at step 100.
Auxiliary 0.001 reached 179.3254 at step 200. Exposure is
19,660,800 tokens, with matching initial weights and data fingerprints.
Worker 480344 remains live toward 1,000 updates. The current
early improvement does not complete the rate comparison; auxiliary
0.01 and all five-epoch formal runs remain pending.

## Second Muon confirmation at 600 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 56.6935
at step 600, improving from 63.7115 at step 500. Auxiliary 0.001
reached 58.2521 at the same step. Exposure is 58,982,400 tokens
and the initial-weight/data fingerprints match. Worker 380878
remains live with 400 updates left in this confirmation. Final
rate selection and all five-epoch formal runs remain pending.

## Second Muown confirmation reaches the short-screen horizon

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 90.5436 at step 300, down from 167.4201 at step 200.
Auxiliary 0.001 reached 96.3512 at step 300. Exposure is
29,491,200 tokens and the initial-weight/data fingerprints match.
Worker 480344 remains live and continues beyond the original
short-screen horizon toward 1,000 updates. The auxiliary-0.01
comparison and full five-epoch runs remain pending.

## Second Muon confirmation at 700 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 51.5110
at step 700, improving from 56.6935 at step 600. Auxiliary 0.001
reached 53.1417 at the same step. Exposure is 68,812,800 tokens
and the initial-weight/data fingerprints match. Worker 380878
remains live with 300 updates left in this confirmation. Final
rate selection and five-epoch pretraining remain pending.

## Prepared shared five-epoch protocol

Prepared `records/2026-09-15_qwen3_scratch_five_epoch_protocol.json`
and validated its common arguments through the current CLI parser for
all three optimizers. It uses scratch initialization, seed 1337,
microbatch 12 with accumulation 4, five epochs, no maximum-update
cap, 100-update warmup, full cosine horizon 101,725, minimum rate
ratio 0.1, clipping 1, weight decay 0.1, and evaluation every 500
updates on the fixed 96 validation blocks. The warmup is retained
from the tested protocol; the decay horizon is extended for the full
run and its behavior remains unmeasured. Expected exposure is
9,999,974,400 tokens across 101,725 updates.

The shared label is `scratch_2b_5ep_tuned_v1_mb12a4`. Rate arguments
are deliberately unset until longer comparisons complete. Launch
through the training CLI so final checkpoints are retained, verify
device ownership first, and initialize anew rather than resume a
tuning checkpoint. After all formal runs, evaluate each checkpoint
on the complete heldout set and render the required step/time
comparison PNGs. No formal run has launched yet.

## Local AdamW confirmation at 700 updates

Local AdamW learning rate 0.001 reached perplexity 80.3513 at
step 700, improving from 92.9861 at step 600. Exposure is
68,812,800 tokens and elapsed time 15276.0991 seconds. All
scheduled metric rows through step 700 are present, with matching
initial weights and data fingerprints. Worker 3704283 remains live
and has 300 updates left; rates 0.003 and 0.006 remain queued.
Timing remains hardware-separated, and five-epoch runs are pending.

## Second Muown confirmation at 400 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 70.5942 at step 400, improving from 90.5436 at step 300.
Auxiliary 0.001 reached 72.8414 at step 400. Exposure is
39,321,600 tokens and initial-weight/data fingerprints match.
Worker 480344 remains live and continues toward 1,000 updates.
The remaining rate comparison and five-epoch runs are pending.

## Second Muon confirmation at 800 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 47.9727
at step 800, improving from 51.5110 at step 700. Auxiliary 0.001
reached 49.2899 at the same step. Exposure is 78,643,200 tokens
and the initial-weight/data fingerprints match. Worker 380878
remains live with 200 updates left. The five-epoch common protocol
is prepared, but final rate selection and formal launch await
completion of the longer comparison.

## Second Muown confirmation at 500 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 61.1710 at step 500, improving from 70.5942 at step 400.
Auxiliary 0.001 reached 62.6343 at the same step. Exposure is
49,152,000 tokens and the initial-weight/data fingerprints match.
Worker 480344 remains live and is halfway through this trial.
The auxiliary-0.01 comparison and five-epoch runs remain pending.

## Second Muon confirmation at 900 updates

Muon matrix 0.003 / auxiliary 0.003 reached perplexity 44.8711
at step 900, improving from 47.9727 at step 800. Auxiliary 0.001
reached 46.1721 at the same step. Exposure is 88,473,600 tokens
and the initial-weight/data fingerprints match. Worker 380878
remains active with 100 updates left. After successful completion,
compare both full traces, select the Muon rate, and launch its
prepared five-epoch run after verifying GPU ownership and capacity.
No formal run has launched yet.

## Second Muown confirmation at 600 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 54.7428 at step 600, improving from 61.1710 at step 500.
Auxiliary 0.001 reached 55.6538 at the same step. Exposure is
58,982,400 tokens and the initial-weight/data fingerprints match.
Worker 480344 remains live with 400 updates left in this trial.
Final rate selection and five-epoch runs remain pending.

## Muon rate selected and full five-epoch pretraining launched

Both Muon confirmations completed successfully. Auxiliary 0.003
ended at perplexity 42.5850, compared with 43.8401 for 0.001,
and was lower at every saved evaluation from step 400 through 1000.
Selected matrix rate 0.003 and auxiliary rate 0.003 as the best
observed tested setting. Small gaps and single-seed repeat variation
remain a limitation; this is not a global-optimality or significance
claim. Both complete result/trace sets and the zero-exit-code queue
summary were synchronized and checked. Local and ABA hashes match
for all five core source files.

After verifying the tuning worker and dispatcher exited, device 0
had no compute owner and 81,153 MiB free. Launched the training CLI
directly via nohup as PID 520877 with the prepared shared label
`scratch_2b_5ep_tuned_v1_mb12a4`. It uses fresh scratch initialization,
seed 1337, batch 12 with accumulation 4, five epochs, no maximum
update cap, warmup 100, cosine horizon 101,725, clipping 1, weight
decay 0.1, and evaluation every 500 updates. Final checkpoints are
retained. The launch record and complete command are in
`records/2026-09-15_qwen3_muon_formal_launch.json`.

Runtime verification: PID 520877 is live, A100 device 0 is at full
utilization with 65,253 MiB used, and step 0 reports perplexity
183646.8644 with the exact expected initial-weight/data hashes.
This verifies launch and scratch initialization, not a completed
epoch or a post-training evaluation. Muown PID 480344 remains active
on device 1; local AdamW continues its tuning queue. First full
pretraining run is now running; the full three-baseline objective
remains unfinished.

## Local AdamW confirmation at 800 updates

Local AdamW learning rate 0.001 reached perplexity 71.7019 at
step 800, improving from 80.3513 at step 700. Exposure is
78,643,200 tokens and elapsed time 17458.9152 seconds. All
scheduled rows through step 800 are present, with matching initial
weights and data fingerprints. Worker 3704283 remains active with
200 updates left; local rates 0.003 and 0.006 remain queued.
Muon formal pretraining PID 520877 and Muown tuning PID 480344
were verified live. The three-baseline goal remains unfinished.

## Second Muown confirmation at 700 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 50.2099 at step 700, improving from 54.7428 at step 600.
Auxiliary 0.001 reached 50.9258 at the same step. Exposure is
68,812,800 tokens and the initial-weight/data fingerprints match.
Worker 480344 remains live with 300 updates left in this trial.
Muon formal five-epoch pretraining remains live; local AdamW is
at step 800. Muown and AdamW final rate choices remain pending.

## Final figure preparation check

The existing baseline renderer reads all three optimizer traces under the
shared run label and produces both step and elapsed-hour PNG files. It
currently uses a linear perplexity axis. Scratch step-zero perplexity is
about 183,647, so the final rendering must use a clearly labeled logarithmic
perplexity axis or an additional post-initialization detail panel to keep
the trained curves readable while preserving the initial observation.
Do not render incomplete traces as a final comparison. Before final export,
verify all three result files report five completed epochs and 101,725
updates, matching initial weights, data, effective batch and validation
subset; retain full-heldout evaluations separately. Compare elapsed time
only among the formal ABA A100 runs. The local AdamW tuning time is not
part of the final timing comparison. No training source was changed during
this preparation check.

## Scratch baseline figure scale corrected

The baseline renderer now detects scratch initialization in all three
step-zero rows and uses a labeled logarithmic perplexity axis for both
step and elapsed-time plots. It keeps the initial observation and every
subsequent point. Legacy traces retain their linear axes. Three existing
renderer tests passed, and a no-file rendering check using the actual
AdamW, Muon and Muown scratch traces verified both logarithmic axes and
all three initial values above 180,000. No final comparison was exported.
Only rendering changed; running training processes were not restarted.
The source file SHA256 after this rendering-only change is `28889bdbbc501cbab8743f1a9eefa8a5acfdc0252c2aa6cca766acad7abae6e5`;
formal launch provenance retains the previous source hash intentionally.

## Second Muown confirmation at 800 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 46.9205 at step 800, improving from 50.2099 at step 700.
Auxiliary 0.001 reached 47.2947 at the same step. Exposure is
78,643,200 tokens; all scheduled rows are present and initial-weight/data
fingerprints match. Worker 480344 remains live with 200 updates left.
Muon formal pretraining and local AdamW tuning workers remain live.
Muown final rate selection awaits both this result and auxiliary 0.01.

## Second Muown confirmation at 900 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.003 reached
perplexity 43.9846 at step 900, improving from 46.9205 at step 800.
Auxiliary 0.001 reached 44.3721 at the same step. Exposure is
88,473,600 tokens; all scheduled rows are present and initial-weight/data
fingerprints match. Worker 480344 remains live with 100 updates left.
Muon formal pretraining and local AdamW tuning workers remain live.
Muown final rate selection awaits this completed result and auxiliary 0.01.

## Local AdamW confirmation at 900 updates

Local AdamW learning rate 0.001 reached perplexity 64.4700 at
step 900, improving from 71.7019 at step 800. Exposure is
88,473,600 tokens and elapsed time is 19641.7618 seconds. All
scheduled rows through step 900 are present, with matching initial
weights and data fingerprints. Worker 3704283 remains active with
100 updates left; local rates 0.003 and 0.006 remain queued.
The local trace is backed up only local-to-remote. Muon formal
pretraining and Muown tuning remain live; the full goal is unfinished.

## Second Muown confirmation completed; final auxiliary bracket started

The auxiliary-0.003 Muown trial completed 1,000 updates successfully
(exit code 0), ending at perplexity 41.8773 versus 42.2124 for
auxiliary 0.001. All 11 evaluations are finite and strictly improving;
initial weights, data fingerprint and 98,304,000-token exposure match.
Elapsed time was 7874.9439 seconds. Full result, trace and queue summary
were synchronized and audited. This is tuning, not five completed epochs.
The dispatcher automatically started its last trial, direction 0.01 /
gain 0.0003 / auxiliary 0.01, as worker 542684. Do not select the
final Muown rate until this bracket finishes. Muon formal pretraining
continues on device 0 and local AdamW continues its first confirmation.

## Muon formal pretraining: first evaluation at step 500

The selected Muon matrix/auxiliary rates 0.003/0.003 reached
perplexity 64.2762 after 500 updates and 49,152,000 tokens in
the full five-epoch run, down from random-initialization perplexity
183646.8644. Initial weights and data fingerprints match the shared
protocol. The first post-training evaluation is finite and verifies
learning under the full 101,725-update cosine schedule. This is
one early evaluation, not a completed epoch or evidence of final
convergence. PID 520877 remains live; the five-epoch run continues.

## Final Muown auxiliary-rate trial at 100 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.01 reached
perplexity 581.5502 at step 100. Auxiliary 0.003 reached 465.4763
and auxiliary 0.001 reached 498.7204 at the same step. The higher
rate is weaker at this early evaluation, but selection awaits the
complete 1,000-step trace because earlier Muon trials changed ordering.
Exposure is 9,830,400 tokens; initial weights and data fingerprints match.
Worker 542684 remains live. Muon full pretraining and local AdamW
tuning also remain live.

## First local AdamW confirmation completed; second rate started

Local AdamW rate 0.001 completed 1,000 updates successfully (exit 0),
ending at perplexity 59.6417. All 11 evaluations are finite and strictly
improving, with matching initial weights/data and 98,304,000-token
exposure. The complete trace includes the exactly restored step-500
row documented earlier. The dispatcher started rate 0.003 as PID
372709, and its scratch initialization fingerprints match. Rate 0.006
remains queued. Final AdamW rate selection awaits these comparisons;
local elapsed times will not enter the formal A100 speed comparison.

The final Muown auxiliary-0.01 trial reached perplexity 205.0605 at
step 200, improving from 581.5502 but still above auxiliary-0.003
(167.4201) and auxiliary-0.001 (179.3254) at the same step. Its
fingerprints and 19,660,800-token exposure were checked. Worker
542684 and formal Muon worker 520877 remain live.

## Final Muown auxiliary-rate trial at 300 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.01 reached
perplexity 98.1472 at step 300, improving from 205.0605 at step
200. Auxiliary 0.003 reached 90.5436 and auxiliary 0.001 reached
96.3512 at the same step. The higher rate remains weaker, though
the difference has narrowed. Full-trace selection remains pending.
Exposure is 29,491,200 tokens; initial weights and data fingerprints
match. Muown worker 542684, Muon formal worker 520877, and local
AdamW rate-0.003 worker 372709 remain live.

## Final Muown auxiliary-rate trial at 400 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.01 reached
perplexity 74.4917 at step 400, improving from 98.1472 at step
300. Auxiliary 0.003 reached 70.5942 and auxiliary 0.001 reached
72.8414 at the same step. The higher auxiliary rate remains weaker
at each saved evaluation so far. Full-trace selection remains pending.
Exposure is 39,321,600 tokens; initial weights and data fingerprints
match. Muown worker 542684, Muon formal worker 520877, and local
AdamW rate-0.003 worker 372709 remain live.

## Second local AdamW confirmation at 100 updates

Local AdamW learning rate 0.003 reached perplexity 839.1994 at
step 100, compared with 851.9674 for learning rate 0.001 on the
same local protocol. Exposure is 9,830,400 tokens and elapsed time
is 2178.3106 seconds. Initial weights and data fingerprints match.
The small early advantage does not establish the final rate choice.
Worker 372709 remains live; rate 0.006 remains queued.

## Final Muown auxiliary-rate trial at 500 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.01 reached
perplexity 63.3707 at step 500, improving from 74.4917 at step
400. Auxiliary 0.003 reached 61.1710 and auxiliary 0.001 reached
62.6343 at the same step. The higher rate remains weaker halfway
through the final comparison. Exposure is 49,152,000 tokens and
initial weights/data fingerprints match. Worker 542684 remains live;
Muon formal pretraining and local AdamW rate-0.003 tuning continue.

## Muon formal pretraining at step 1,000

Muon matrix/auxiliary rates 0.003/0.003 reached perplexity 45.5695
at 1,000 updates, improving from 64.2762 at step 500. Exposure
is 98,304,000 tokens; initial weights/data fingerprints match.
The full schedule retains a higher learning rate than the 2,000-step
tuning schedule, so its endpoint is not the same experiment as the
42.5850 tuning endpoint. The formal curve is improving so far;
1,000 of 101,725 planned updates are complete, with no complete
epoch yet. PID 520877 remains live and the five-epoch run continues.

## Final Muown auxiliary-rate trial at 600 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.01 reached
perplexity 55.9954 at step 600, improving from 63.3707 at step
500. Auxiliary 0.003 reached 54.7428 and auxiliary 0.001 reached
55.6538 at the same step. The higher rate remains weaker so far.
Exposure is 58,982,400 tokens; initial weights/data fingerprints
match. Worker 542684 remains live with 400 updates left. Muon
formal pretraining and local AdamW rate-0.003 tuning continue.

## Final Muown auxiliary-rate trial at 700 updates

Muown direction 0.01 / gain 0.0003 / auxiliary 0.01 reached
perplexity 50.9949 at step 700, improving from 55.9954 at step
600. Auxiliary 0.003 reached 50.2099 and auxiliary 0.001 reached
50.9258 at the same step. The higher rate remains weaker so far.
Exposure is 68,812,800 tokens; initial weights/data fingerprints
match. Worker 542684 remains live with 300 updates left. Muon
formal pretraining and local AdamW rate-0.003 tuning continue.

## Local AdamW rate-0.003 confirmation at 200 updates

Learning rate 0.003 reached perplexity 398.5812 at step 200,
improving from 839.1994 at step 100. The matched local
rate-0.001 trial reached 400.4505 at step 200; this small early
gap does not establish the selected rate. Exposure is 19,660,800 tokens;
initial weights and data fingerprints match. Worker 372709 remains
live with 800 updates left; rate 0.006 remains queued. Local timing
is excluded from the eventual A100 formal timing comparison.

## User-required effective batch 64 supersedes batch-48 campaign

See `records/2026-09-15_qwen3_batch64_transition.md` for the live-state audit, preserved historical results, memory probes and replacement launch records. The batch-48 Muon formal run was stopped after its saved step-5500 evaluation (29.0822). New scratch finalist comparisons now use batch 64. Five-epoch final curves must come from the new compliant protocol; earlier batch-48 results are historical tuning evidence.
