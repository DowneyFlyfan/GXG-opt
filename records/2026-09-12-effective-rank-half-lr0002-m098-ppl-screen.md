# Fixed-rank-half learning-rate 0.002 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5 throughout.
- Learning rate: 0.002; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,963.657 seconds. Validation negative log likelihood
was 1.05872462, corresponding to perplexity **2.88269212**. The strict
constraint accepted 57,773 matrix updates, skipped none, and retained its exact
0.5 rank floor. Newton--Schulz supplied 77,292 directions with no fallback.

This is worse than the LR-0.00175 strict-rank leader (2.82134636), so learning
rate 0.002 is rejected. Fitting a quadratic to the three matched screens at
0.0015, 0.00175, and 0.002 places the local minimum near 0.001674. The next
screen tests 0.001675 with unchanged momentum and constraint.
