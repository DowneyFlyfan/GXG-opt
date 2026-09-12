# Muown 0.0045 Perplexity Screen

## Protocol

- Task/model: GPT2-12x512, 54,682,624 parameters.
- Hardware: ABA NVIDIA A100 80GB, device 0.
- Optimizer: Muown; learning rate 0.0045; weight decay 0.
- Effective batch size: 48 (micro-batch 8, six gradient accumulations).
- Budget: 2,034 optimizer updates (one matched epoch); validation: 64 batches.

## Result

Validation negative log-likelihood was 0.96103798 and perplexity was
**2.61440876** after 1,690.97 seconds.  This is slightly worse than the
matched LR-0.005 Muown reference (2.61343183), so 0.0045 is rejected.

Together with the completed 0.005 and 0.006 screens, the local quadratic
interpolation has its vertex near 0.00488.  Device 0 therefore immediately
continues with a fresh 0.004875 screen; device 1 continues the independent
fixed-effective-rank-0.5 momentum screen.
