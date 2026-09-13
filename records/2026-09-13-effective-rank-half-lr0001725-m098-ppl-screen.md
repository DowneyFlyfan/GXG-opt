# Fixed-rank-half learning-rate 0.001725 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5, learning rate
  0.001725, momentum 0.98, and weight decay 0.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,958.062 seconds. Validation negative log likelihood
was 1.06382777, corresponding to perplexity **2.89744054**. The exact rank
floor remained 0.5; 60,954 constrained matrix updates were accepted and there
were no skips, projections, or fallback steps.

This lower-side neighbor is far worse than the LR-0.00175 incumbent (PPL
2.82134636), by 0.07609418 PPL. It is rejected. The final local interpolation
candidate is LR 0.001775, between the incumbent and the already-rejected
LR-0.0018 screen.
