# Muown 0.00625 Perplexity Screen

## Protocol

- Model/task: GPT2-12x512 next-token prediction, 54,682,624 parameters.
- Hardware: ABA NVIDIA A100 80 GB, CUDA device 0.
- Optimizer: Muown; learning rate 0.00625, momentum 0.95, and weight decay 0.
- Matched screen: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 updates, and 64 validation batches.

## Result and decision

The screen completed in 1,671.489 seconds, using a measured peak of 9,096.17
MiB.  Its validation negative log likelihood was 0.96527376 and perplexity was
**2.62550631**.  This is 0.01207448 worse than the matched Muown learning-rate
0.005 reference (2.61343183), so 0.00625 is rejected.  Together with the
completed screens at 0.005125, 0.006, 0.0075, and 0.010, it confirms that
increasing the learning rate above 0.005 does not improve the one-epoch
matched protocol.

The authoritative machine-readable evidence is
`metrics/nlp/nlp_gpt_12x512__aba_a100_muown_lr000625_m095_e1__muown.ppl.jsonl`
and its paired result summary.
