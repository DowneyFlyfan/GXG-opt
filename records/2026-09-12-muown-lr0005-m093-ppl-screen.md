# Muown momentum 0.93 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 0.
- Optimizer: Muown; learning rate 0.005, momentum 0.93, and weight decay 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,932.765 seconds. Validation negative log likelihood
was 0.96470776, corresponding to perplexity **2.62402069**.

This is worse than the otherwise matched momentum-0.95 result (2.61343183)
and therefore rejects the interpolated momentum 0.93. Momentum 0.95 remains
the best screened Muown setting. The next screen holds it fixed and tests
learning rate 0.00475, between earlier 0.0045 and 0.005 screens.
