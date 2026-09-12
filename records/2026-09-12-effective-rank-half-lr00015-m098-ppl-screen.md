# Fixed-rank-half learning-rate 0.0015 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5 throughout.
- Learning rate: 0.0015; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,948.203 seconds. Validation negative log likelihood
was 1.04249631, corresponding to perplexity **2.83628843**. The strict
constraint accepted 61,969 matrix updates, skipped none, and reported an exact
rank floor of 0.5. Newton--Schulz supplied 77,292 directions with no fallback
or joint-Newton step.

This improves the matched fixed-rank setting at learning rate 0.00125 and
momentum 0.98 (perplexity 2.87833246) by 0.04204403 PPL. It is the current
best strict-rank-half epoch-one screen, but remains above tuned Muon
(2.62528750). The next screen retains momentum 0.98 and tests the higher
learning rate 0.00175.
