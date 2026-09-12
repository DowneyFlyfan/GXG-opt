# Fixed-rank-half momentum 0.99 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), ABA A100 device 1.
- Optimizer: `effective_rank_half`, fixed rank floor 0.5, learning rate 0.00125,
  momentum 0.99, and zero weight decay.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 updates, and 64 validation batches.

## Result and decision

The screen completed in 1,949.782 seconds with validation negative
log-likelihood 1.08486321 and perplexity **2.95903501**.  It accepted 60,586
matrix updates, skipped none, and retained the reported rank floor of 0.5.

Momentum 0.99 is worse than 0.98 (2.87833246) and 0.95 (2.91308164), so it is
rejected.  A quadratic interpolation through the measured points 0.95, 0.98,
and 0.99 has its vertex near 0.968; the next strict-fixed-rank screen tests
that predicted region rather than extrapolating farther upward.
