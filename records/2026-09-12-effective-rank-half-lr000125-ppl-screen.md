# Fixed Effective-Rank-Half 0.00125 Perplexity Screen on ABA A100

## Objective

Evaluate the finite-step certified optimizer with its effective-rank floor held
at one half throughout training.  This screen intentionally does not use the
linear 0.2-to-0.8 schedule.

## Configuration

- Host and device: ABA NVIDIA A100 80 GB, CUDA device 1.
- Optimizer: `EffectiveRankHalf`; rank floor: 0.5 at every update.
- Learning rate: 0.00125; weight decay: 0.0; auxiliary AdamW learning rate:
  0.0003.
- Micro-batch size: 8; gradient accumulation: 6; effective batch: 48.
- Screen budget: 2,034 optimizer updates and 64 validation batches.

## Result

The screen completed in 1,914.482 seconds.  Validation negative
log-likelihood was 1.06921150 and perplexity was **2.91308164**, which is not
competitive with the matched 2.61343183 Muown epoch-one reference.  The
diagnostics prove `effective_rank_floor = 0.5`, 77,292 Newton--Schulz
directions, and zero joint-Newton/fallback steps.  The base solver is
therefore rejected; the next screen uses the static fixed-half joint-Newton
solver, not a linear schedule.
