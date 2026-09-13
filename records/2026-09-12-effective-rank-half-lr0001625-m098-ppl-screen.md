# Fixed-rank-half learning-rate 0.001625 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 0, concurrently shared with an unrelated process.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5 throughout.
- Learning rate: 0.001625; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 2,228.315 seconds. Validation negative log likelihood
was 1.06155869, corresponding to perplexity **2.89087345**. The strict
constraint accepted 60,474 matrix updates, skipped none, and retained its
fixed rank floor of 0.5. Newton--Schulz supplied 77,292 directions with no
fallback or joint-Newton step.

This is worse than the otherwise matched LR-0.00175 screen (2.82134636) by
0.06952709 PPL, so LR 0.001625 is rejected. Together with the rejected
LR-0.001675 and LR-0.002 screens, the completed local search selects LR 0.00175
and momentum 0.98 for the five-epoch strict-rank-half final.
