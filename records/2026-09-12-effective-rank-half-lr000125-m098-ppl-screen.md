# Fixed-rank-half momentum 0.98 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; the rank floor stayed fixed at 0.5 for all updates.
- Learning rate: 0.00125; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,932.361 seconds.  Validation negative log likelihood
was 1.05721112, corresponding to perplexity **2.87833246**.  The strict
constraint accepted 65,226 matrix updates and skipped none; the rank floor was
reported as exactly 0.5.

This is an improvement over the otherwise matched momentum-0.95 screen
(perplexity 2.91308164) and the momentum-0.8 screen (3.28425592).  It remains
above the tuned Muon epoch-one baseline (2.62528750), so it is evidence for a
better fixed-rank setting, not a claim of baseline superiority.  The next
screen tests momentum 0.99 on the freed second A100.
