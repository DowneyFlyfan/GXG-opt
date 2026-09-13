# Fixed-rank-half learning-rate 0.0018 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5, learning rate
  0.0018, momentum 0.98, and weight decay 0.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,957.588 seconds. Validation negative log likelihood
was 1.03752046, corresponding to perplexity **2.82221053**. The exact 0.5
rank floor was preserved: 60,774 constrained matrix updates were accepted,
there were no skipped, projected, or fallback steps, and Newton--Schulz
supplied 77,292 directions. Peak memory was 8,962.83 MiB.

The result is worse than the selected LR-0.00175, momentum-0.98 screen (PPL
2.82134636) by 0.00086418 PPL. It is rejected; the completed five-epoch
LR-0.00175 rank-half run remains the best confirmed fixed-rank-half result.
