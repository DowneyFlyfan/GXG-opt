# Fixed-rank-half learning-rate 0.001675 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5 throughout.
- Learning rate: 0.001675; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,955.937 seconds. Validation negative log likelihood
was 1.06477242, corresponding to perplexity **2.90017889**. The strict
constraint accepted 60,478 matrix updates, skipped none, and maintained its
fixed rank floor of 0.5. Newton--Schulz supplied 77,292 directions with no
fallback or joint-Newton step.

This is worse than the otherwise matched LR-0.00175 screen (2.82134636) by
0.07883253 PPL, so LR 0.001675 is rejected. Its regression also invalidates
the prior quadratic interpolation as a guide for this non-smooth constrained
update; the independent LR-0.001625 screen remains the only live refinement.
