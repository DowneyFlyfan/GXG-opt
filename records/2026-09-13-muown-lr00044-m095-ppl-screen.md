# Muown learning-rate 0.0044 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 0.
- Optimizer: Muown with learning rate 0.0044, momentum 0.95, and weight decay 0.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,688.451 seconds. Validation negative log likelihood
was 0.96137390, corresponding to perplexity **2.61528716**.

This lower-side interpolation is worse than LR 0.0045 (PPL 2.61440876) and
the best recorded Muown screen, LR 0.005 with momentum 0.95 (PPL 2.61343183).
It is rejected. The LR sweep provides no evidence to replace the existing
five-epoch Muown result.
