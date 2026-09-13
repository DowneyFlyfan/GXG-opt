# Fixed-rank-half learning-rate 0.001775 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5, learning rate
  0.001775, momentum 0.98, and weight decay 0.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,954.270 seconds. Validation negative log likelihood
was 1.06598010, corresponding to perplexity **2.90368348**. It preserved the
exact 0.5 rank floor, accepted 60,677 constrained matrix updates, and used no
skip, projection, or fallback step.

This final interpolation candidate is worse than the LR-0.00175 incumbent
(PPL 2.82134636) by 0.08233712 PPL. Together with the rejected LR 0.001725
and LR 0.0018 screens, it establishes LR 0.00175 with momentum 0.98 as the
best screened fixed-rank-half setting. The completed five-epoch confirmation
at that setting remains the fixed-rank-half result to compare with baselines.
