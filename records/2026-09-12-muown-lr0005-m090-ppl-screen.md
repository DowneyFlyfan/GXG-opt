# Muown momentum 0.90 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), ABA A100 device 0.
- Optimizer: Muown, learning rate 0.005, momentum 0.90, and zero weight decay.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 updates, and 64 validation batches.

## Result and decision

The screen completed in 1,929.709 seconds.  Validation negative log-likelihood
was 0.96180865 and perplexity was **2.61642440**.

This is slightly worse than the otherwise matched momentum-0.95 reference
(2.61343183), so momentum 0.90 is rejected.  The next screen holds learning
rate 0.005 and tests momentum 0.97, probing the opposite side of the incumbent
without conflating momentum with another learning-rate change.
