# Muown 0.010 Perplexity Screen on ABA A100

## Objective

Screen a learning rate above the completed Muown setting while retaining the
matched GPT2-12x512 perplexity protocol.  The completed reference used
learning rate 0.005 and obtained epoch-one validation perplexity 2.61343183.

## Configuration

- Host and device: ABA NVIDIA A100 80 GB, CUDA device 0.
- Optimizer: Muown.
- Learning rate: 0.010.
- Weight decay: 0.0.
- Auxiliary AdamW learning rate: 0.0003.
- Micro-batch size: 8; gradient accumulation: 6; effective batch: 48.
- Screen budget: 2,034 optimizer updates (one matched epoch).
- Validation: 64 batches; metric is token-weighted next-token perplexity.

## Result

The screen completed in 1,694.813 seconds.  Its epoch-one negative
log-likelihood was 0.96786224 and its validation perplexity was **2.63231118**.
This is worse than the matched 0.005 reference by 0.01887935 perplexity, so
0.010 is rejected.  The next screen uses 0.0075 on the same A100 while the
fixed effective-rank-half experiment runs on the other A100.
