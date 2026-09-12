# Muown learning-rate 0.005125 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task on ABA A100 device 0.
- Optimizer: Muown with its original momentum 0.95 and zero weight decay.
- Learning rate: 0.005125.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result and decision

The screen completed in 1,955.328 seconds.  Validation negative log likelihood
was 0.96669808 and perplexity was **2.62924855**.

This is worse than the otherwise matched learning-rate-0.005 reference
(perplexity 2.61343183), so 0.005125 is rejected.  The next Muown screen holds
the better learning rate at 0.005 and changes the exposed optimizer momentum to
0.90; this tests a different optimization dynamic rather than another
near-duplicate learning rate.
