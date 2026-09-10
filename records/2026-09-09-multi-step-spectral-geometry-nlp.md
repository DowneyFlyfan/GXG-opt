# Multi-Step Spectral Geometry on matched GPT2-12x512

## Scope

Evaluate the Multi-Step Spectral Geometry idea on the existing cached NLP
dataset and the same five-epoch, effective-batch-48 protocol used by the
tuned AdamW and Muon baselines.  This record is deliberately incomplete until
the active experiment has produced an epoch metric.

## Latest-branch comparison

`git fetch origin --prune` was run before the local trial.  No commit is ahead
of the current checkout.  The detailed remote comparison is retained in
`records/2026-09-08-multi-step-spectral-geometry-remote-comparison.md`.

The current implementation retains the fetched updating rule:

\[
M_t=\beta M_{t-1}+(1-\beta)G_t,\qquad
W_{t+1}=W_t-\eta s(W_t)T_{p_t}(M_t),
\]

where \(p_t\in\{2,4,8,\infty\}\) is selected by a four-step,
positive-semidefinite secant-curvature rollout.  The local trial uses the
reduced transform (eight Newton--Schulz steps and degree-12 polynomial), a
500-step warm-up, 100-step policy interval, horizon weights \((1,.8,.6,.4)\),
history and basis rank eight, ridge \(10^{-4}\), and momentum \(.95\).

## Resource check and active screen

The initial 12 microbatch x 4 accumulation configuration exhausted the local
16 GB GPU on its first backward pass: it needed a further 1.46 GB while only
0.74 GB remained.  It produced no metric and is not an experiment result.

The first, equivalent effective-batch screen was:

| Setting | Value |
| --- | ---: |
| Label | `lr003_b8_a6_screen` |
| Spectral learning rate | `0.003` |
| Auxiliary AdamW learning rate | `0.0003` |
| Microbatch | 8 |
| Gradient accumulation | 6 |
| Effective batch | 48 |
| Epoch budget | 1 screen epoch |
| GPU memory during run | about 14.9 GB / 16.3 GB |

The screen must finish with a finite epoch metric and valid policy diagnostics
before choosing a five-epoch configuration.  Final comparisons require both
metric-steps and metric-time plots against the selected AdamW and Muon
baselines.

## Five-epoch matched run: epoch 1

The selected 0.0003 configuration completed epoch 1 (2,034 optimizer steps)
in 2,127.21 seconds at **0.68489075** validation next-token accuracy.  It
matches the independent 0.0003 screen exactly, confirming reproducibility.
The single final trainer wrote a 5.3 GB resumable checkpoint after the epoch;
it remains active for epochs 2--5.  Peak allocated memory remains within the
validated 13,367.38 MB envelope.

## Five-epoch matched run: epoch 2

Epoch 2 completed at optimizer step 4,068 in 4,264.53 cumulative seconds,
reaching **0.71515465** validation next-token accuracy.  The policy remained
at \(p=\infty\) for all 48 spectral matrices without any switch.  This is
above the tuned Muon one-epoch screen value (0.71167119); it does not by itself
establish a final comparison, which still requires the matched five epochs.
The trainer checkpoint was refreshed and the same process proceeded to epoch 3.

## Five-epoch matched run: epoch 3

Epoch 3 completed at optimizer step 6,102 in 6,402.17 cumulative seconds,
reaching **0.72976303** validation next-token accuracy.  The terminal policy
again selected \(p=\infty\) for every one of the 48 spectral matrices with no
switch.  The checkpoint was renewed and the original trainer proceeded to
epoch 4.

## Five-epoch matched run: epoch 4

Epoch 4 completed at optimizer step 8,136 in 8,536.72 cumulative seconds,
reaching **0.73907471** validation next-token accuracy.  The recorded policy
remained \(p=\infty\) for all 48 matrices with no switch.  The checkpoint was
renewed and the same trainer began its fifth and final matched epoch.

## Completed matched result

The final epoch completed at optimizer step 10,170 in 10,672.76 seconds
(2.965 hours), with **0.74419785** validation next-token accuracy.  The run
finished normally within the four-hour cap, used 13,367.38 MB peak allocated
GPU memory, and retained \(p=\infty\) on every one of the 48 spectral matrices
at each recorded epoch (no endpoint policy switches).

| Optimizer | Epoch-5 metric | Wall time (hours) |
| --- | ---: | ---: |
| AdamW (1.5e-4) | 0.72374725 | 2.7155 |
| Muon (2.5e-3) | 0.75503159 | 2.7462 |
| Multi-Step Spectral Geometry (0.0003) | 0.74419785 | 2.9647 |

Thus the method beats the tuned AdamW baseline by 0.02045059 accuracy, but it
trails tuned Muon by 0.01083374 and takes about 8% longer than Muon.  The
evidence establishes a stable, competitive-with-AdamW approximation under the
matched five-epoch protocol; it does not establish a new best optimizer over
Muon.  The verified plots are
`results/nlp/multi_step_spectral_lr0003_b8_a6_final_metric_steps.png` and
`results/nlp/multi_step_spectral_lr0003_b8_a6_final_metric_time.png`.

## First screen result: learning rate 0.003

The first screen completed one epoch (2,034 optimizer steps) in 2,120.55
seconds with peak allocated memory 13,367.38 MB.  Its validation next-token
accuracy was **0.28080559**.  It is finite and resource-valid, but it is not a
competitive configuration: the matched Muon learning-rate sweep reaches
0.71167119 at its 0.0025 setting after the same epoch, while even the smallest
AdamW sweep reaches 0.60662587.

The final policy selected \(p=\infty\) for 42 of 48 spectral matrices,
\(p=2\) for five, and \(p=4\) for one, with 22 policy switches.  The next
screen lowers the spectral learning rate tenfold to 0.0003; this changes only
the step magnitude while retaining the same formula, effective batch, seed,
and policy configuration.

## Second screen result: learning rate 0.0003

The second screen completed the same one epoch and 2,034 optimizer steps in
2,126.44 seconds, with peak allocated memory 13,367.38 MB.  Its validation
next-token accuracy was **0.68489075**.  Thus the tenfold reduction restores
stable learning and is much stronger than the 0.003 screen, but it remains
below the matched one-epoch Muon result (0.71167119) and the selected AdamW
result (0.65027873 is surpassed).

All 48 spectral matrices selected \(p=\infty\) at the recorded endpoint, with
no policy switches.  The next and final one-epoch screen uses \(\eta=0.0006\),
which is a conservative geometric interpolation towards the failed 0.003
setting.  It retains every other setting, including the effective batch 48.

## Third screen result: learning rate 0.0006 and selection

The 0.0006 screen completed one epoch in 2,128.76 seconds with the same peak
allocated memory, 13,367.38 MB.  Its validation next-token accuracy was
**0.67240334**, below the 0.0003 screen by 0.01248741.  The endpoint selected
\(p=\infty\) for 47 matrices and \(p=4\) for one, with one switch.

The three-screen sweep therefore selects **0.0003** for the exact five-epoch
trial: it is finite, uses the matched effective batch 48 without out-of-memory
failure, outperforms the selected AdamW screen at epoch one, and is the best
of the tested Multi-Step rates.  The final run will generate the required
metric-step and metric-time comparisons against the tuned AdamW and Muon
baselines.
