# Muown momentum 0.94 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 0.
- Optimizer: Muown; learning rate 0.005, momentum 0.94, and weight decay 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,927.481 seconds. Validation negative log likelihood
was 0.96276859, corresponding to perplexity **2.61893720**.

This is worse than the otherwise matched Muown momentum-0.95 reference
(2.61343183), so momentum 0.94 is rejected. The momentum scan has not
improved the 0.95 reference; its learning-rate scan also places 0.005 at the
observed local minimum. Further Muown screens should change a parameter with
new evidence rather than repeat nearby momentum values.
