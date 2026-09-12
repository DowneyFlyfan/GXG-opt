# Muown momentum 0.97 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 0.
- Optimizer: Muown; learning rate 0.005, momentum 0.97, and weight decay 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,928.249 seconds. Validation negative log likelihood
was 0.96400813, corresponding to perplexity **2.62218551**.

This is worse than the otherwise matched Muown momentum-0.95 reference
(2.61343183), so momentum 0.97 is rejected. Combining the screens at momentum
0.90 (2.61642440), 0.95 (2.61343183), and 0.97 (2.62218551), the quadratic
interpolant has a local minimum near 0.929. The next independent screen tests
momentum 0.93 at the unchanged learning rate.
