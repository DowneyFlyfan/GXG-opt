# Fixed-rank-half momentum 0.968 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; the rank floor stayed fixed at 0.5 for all updates.
- Learning rate: 0.00125; momentum: 0.968; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,931.304 seconds. Validation negative log likelihood
was 1.07136998, corresponding to perplexity **2.91937626**. The strict
constraint accepted 66,756 matrix updates and skipped none; the rank floor was
reported as exactly 0.5. Newton--Schulz supplied 77,292 directions; no fallback
or joint-Newton step was used.

This is worse than both the matched momentum-0.95 result (2.91308164) and the
current strict-rank best at momentum 0.98 (2.87833246). It rejects the proposed
quadratic-interpolation value 0.968. The next screen holds momentum at 0.98 and
tests a more aggressive learning rate of 0.0015.
