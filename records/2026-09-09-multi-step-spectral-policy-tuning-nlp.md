# Multi-Step Spectral Geometry policy-tuning pass on GPT2-12x512

## Motivation

The completed learning-rate sweep selected learning rate 0.0003 and produced
0.74419785 after five epochs, but its recorded spectral policy stayed at
\(p=\infty\) for every one of the 48 matrices.  This pass tests whether a
non-infinite initialization and earlier, neutral policy decisions can improve
the formula-compatible approximation.

## Controlled protocol

All screens retain the cached NLP dataset, seed 1337, GPT2-12x512, microbatch
8, gradient accumulation 6 (effective batch 48), learning rate 0.0003,
momentum 0.95, reduced transform, and one epoch (2,034 optimizer steps).
The first screen changes only the policy controls: initial \(p=4\), warm-up
50 steps, policy interval 25 steps, and zero switch margin.  It is compared
against the prior one-epoch default-policy result 0.68489075.

Only a finite, resource-valid screen that exceeds the prior default-policy
screen can be considered for a matched five-epoch trial.  Final comparison
still requires both metric-step and metric-time plots against tuned AdamW and
Muon.

## Screen 1: early policy initialized at p=4

The p=4 screen completed one epoch in 2,182.46 seconds with the same
13,367.38 MB peak allocated memory.  It reached **0.67283630**, below the
default-policy screen by 0.01205444.  Its endpoint used p=4 on 47 matrices
and p=2 on one, with three cumulative switches, demonstrating that the new
controls changed the policy rather than being ignored.

This variant is not eligible for a five-epoch run.  The next screen keeps the
same early-decision schedule but initializes at p=8, which tests a less
aggressive departure from the prior p=infinity solution.

## Screen 2 launch blocker: p=8

The p=8 screen was launched only after the p=4 process exited, but it failed
before the model reached the GPU and wrote no metric or result artifact.
`nvidia-smi` reported an NVML driver/library version mismatch, and PyTorch
raised `NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_()` while moving the
model to CUDA.  This is an external local-GPU runtime blocker, not a
finite-training result; no retry is permitted until both NVML and a minimal
PyTorch CUDA allocation are healthy.

The intended p=8 retry was then intentionally cancelled by the user before a
metric, in favor of a more aggressive learning-rate screen.  It is therefore
not a tuning result.

## Screen 2: aggressive learning rate 0.0015

With every default policy setting restored, learning rate 0.0015 completed one
epoch in 2,125.81 seconds at **0.55752182**.  The endpoint selected
\(p=\infty\) for 45 matrices and \(p=4\) for three, with three switches; peak
allocated memory remained 13,367.38 MB.  This is substantially below the
0.0003 default-policy screen (0.68489075), and, together with the prior
0.0006 (0.67240334) and 0.003 (0.28080559) screens, confirms 0.0003 as the
best tested rate.  No repeat five-epoch run is justified because that exact
configuration already has a completed five-epoch result.

## Screen 3: lower-side learning rate 0.00015

With the default policy and the identical one-epoch protocol, learning rate
0.00015 completed in 2,127.43 seconds at **0.67853165**, 0.00635910 below
the 0.0003 screen. Peak allocated memory was 13,367.38 MB, and all 48
matrices remained at \(p=\infty\) with zero switches. Thus lowering the rate
does not improve the incumbent. The final local confirmation is 0.0004: it
lies immediately above 0.0003, while the already completed 0.0006 screen is
also lower. It uses the same seed, batch, accumulation, epoch count, and
default policy; only learning rate changes.

## Screen 4: upper-side learning rate 0.0004

Learning rate 0.0004 completed the matched one-epoch screen in 2,126.76
seconds at **0.68567467**. This exceeds the prior 0.0003 incumbent
(0.68489075) by 0.00078392 and is the best tested rate. Peak allocated memory
was 13,367.38 MB. The policy selected \(p=\infty\) for 45 matrices, \(p=4\)
for two, and \(p=2\) for one, with three switches, so the optimized screen is
also not an inert copy of the prior all-\(p=\infty\) run.

The broad aggressive and local bracket is now complete: 0.00015=0.67853165,
0.0003=0.68489075, 0.0004=0.68567467, 0.0006=0.67240334,
0.0015=0.55752182, and 0.003=0.28080559. A five-epoch final run at 0.0004 is
required before calling it the best completed configuration; it will preserve
the screen's default policy, microbatch 8, accumulation 6, seed 1337, and
cached GPT2-12x512 protocol.

## Final matched run: LR 0.0004, micro-batch 8, accumulation 6

The final fresh five-epoch run completed successfully with 10,170 optimizer
steps, 54,682,624 parameters, and a 13,367 MB peak allocation.

| Epoch | Validation next-token accuracy | Elapsed seconds |
| --- | ---: | ---: |
| 1 | 0.68567467 | 2127.79 |
| 2 | 0.71498299 | 4264.87 |
| 3 | 0.72948456 | 6401.99 |
| 4 | 0.73872566 | 8534.81 |
| 5 | 0.74449921 | 10669.80 |

This is the best tested Multi-Step Spectral Geometry rate in the completed
one-epoch screens. Its final metric exceeds tuned AdamW (0.72374725) by
0.02075195, but remains below tuned Muon (0.75503159) by 0.01053238. Both
final PNG traces were rendered and inspected after completion.
