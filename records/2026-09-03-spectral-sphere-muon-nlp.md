# Spectral-Sphere Muon on the retained NLP baseline

## Purpose

Evaluate the optimizer in ["Muon + spectral sphere" (article 11241)](https://kexue.fm/archives/11241) on the retained `nlp_gpt_12x512` language-model baseline at the same wall-clock budget as the completed Muon baseline.

## Protocol

- Model: `gpt_12x512`, 54,682,624 parameters.
- Dataset, split, initialization seed, scheduler, and auxiliary AdamW parameter group: unchanged from the retained Muon baseline.
- Constrained group: precisely the matrices selected by the project Muon eligibility policy; embeddings, normalization parameters, and biases remain in AdamW.
- Article update: preserve each selected matrix's initial spectral norm.  Form the top-singular tangent `Theta = u_1 v_1^T`, solve the article's ten-step scalar fixed point for `lambda`, set `Phi = msign(G + lambda Theta)`, then spectrally normalize the candidate update.
- `msign`: the project's five-step Newton--Schulz approximation, applied at every one of the article's ten scalar fixed-point iterations.  The article uses SVD only in its NumPy verification snippet; this scalable version follows Muon's approximate primitive.  Its tangent residual is necessarily approximate rather than the exact-SVD residual.
- Physical batch: 12; gradient accumulation: 4; effective batch: 48.  Peak measured allocation is 12,295 MiB of 16,303 MiB, leaving a safety margin without an Out-of-Memory failure.
- Equal wall-clock budget: 10,418.2065 seconds, exactly the recorded completion time of the retained Muon baseline.

## Short calibration

The earlier four-rate table used an exact-SVD `msign` and is invalid for this
experiment, so it is deliberately excluded from the result.  Calibration is
being repeated with the five-step Newton--Schulz implementation, the same
physical/effective batch, seed, and validation procedure.

The approximation-matched 12-step calibration selected `learning_rate=0.003`:
the validation metrics were 0.272781, 0.284005, 0.281949, and 0.282721 for
learning rates 0.001, 0.003, 0.006, and 0.010 respectively.  The formal run
uses `momentum=0.95`, ten fixed-point iterations, and ten power iterations for
spectral normalization.

With effective batch 48, five epochs finish earlier than the 10,418.2065 s
Muon baseline.  The formal loop therefore permits six epochs and uses a
six-epoch cosine horizon, but stops at the exact baseline wall-clock budget.
This prevents an early candidate completion from invalidating the equal-time
comparison.

## Formal result

The formal run reached 8,710 optimizer steps.  Training-time accounting was
10,411.21 s at its last full interval and 10,425.73 s after evaluating the
last completed update; the latter 7.53 s overrun is terminal validation time,
which is retained in accordance with the experiment rule to measure the final
completed step.  Peak allocated memory was 12,297 MiB.

| Optimizer | Wall-clock training budget (s) | Final validation next-token accuracy |
|---|---:|---:|
| AdamW baseline | 10,418.21 | 0.711754 |
| Muon baseline | 10,418.21 | 0.752193 |
| Spectral-Sphere Muon, Newton--Schulz | 10,418.21 | 0.692193 |

The candidate reached 0.692710 at 10,411.21 s, then 0.692193 after terminal
evaluation.  Thus the conclusion does not depend on the terminal-evaluation
overrun: the strict fixed spectral-norm constraint is worse than both retained
baselines on this task.  It is not evidence that Muon is inferior; it is
evidence that constraining every eligible Transformer matrix to its initial
spectral sphere is too restrictive here.

Artifacts:

- `metrics/nlp/nlp_gpt_12x512__spectral_sphere_muon_ns5_equal_time_b12_a4_lr0003.jsonl`
- `results/nlp/nlp_gpt_12x512__spectral_sphere_muon_ns5_equal_time_b12_a4_lr0003.json`
- `results/nlp/spectral_sphere_muon_ns5_equal_time_b12_a4_lr0003_metric_steps.png`
- `results/nlp/spectral_sphere_muon_ns5_equal_time_b12_a4_lr0003_metric_time.png`
