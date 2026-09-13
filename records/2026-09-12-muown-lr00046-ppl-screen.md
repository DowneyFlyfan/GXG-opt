# Muown 0.0046 Perplexity Screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: local NVIDIA GeForce RTX 5070 Ti.
- Optimizer: Muown with learning rate 0.0046, momentum 0.95, auxiliary AdamW
  learning rate 0.0003, and zero weight decay.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 updates, and 64 validation batches.

## Result

The screen completed in 1,954.461 seconds with validation negative
log-likelihood 0.96199478 and perplexity **2.61691143**. It is 0.00347960
worse than the matched Muown 0.005 reference (2.61343183), so learning rate
0.0046 is rejected. The independent 0.00625 A100 screen remains queued on
Nautilus; no completed learning-rate screen has improved the 0.005 reference.
