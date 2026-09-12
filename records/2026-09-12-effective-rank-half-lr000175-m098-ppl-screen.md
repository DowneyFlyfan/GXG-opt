# Fixed-rank-half learning-rate 0.00175 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5 throughout.
- Learning rate: 0.00175; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,946.718 seconds. Validation negative log likelihood
was 1.03721420, corresponding to perplexity **2.82134636**. The strict
constraint accepted 61,651 matrix updates, skipped none, and reported an exact
rank floor of 0.5. Newton--Schulz supplied 77,292 directions with no fallback
or joint-Newton step.

This is the current best strict-rank-half epoch-one result. It improves the
otherwise matched LR-0.0015 screen (2.83628843) by 0.01494207 PPL and the
LR-0.00125 screen (2.87833246) by 0.05698610 PPL. The next screen increases
learning rate to 0.002 while retaining momentum 0.98 and the fixed 0.5 floor.
