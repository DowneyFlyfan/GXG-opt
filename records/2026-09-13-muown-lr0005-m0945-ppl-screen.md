# Muown momentum 0.945 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 0.
- Optimizer: Muown with learning rate 0.005, momentum 0.945, and weight decay 0.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,689.598 seconds. Validation negative log likelihood
was 0.96079557, corresponding to perplexity **2.61377510**. Peak memory was
9,096.17 MiB.

This does not improve the best recorded Muown epoch-one screen (PPL
2.61343183 at learning rate 0.005 and momentum 0.95); it is worse by
0.00034327 PPL. The tested setting is therefore rejected and is not promoted
to the required five-epoch confirmation.
