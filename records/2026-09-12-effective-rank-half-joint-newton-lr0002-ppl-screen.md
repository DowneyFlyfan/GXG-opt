# Fixed Effective-Rank-Half Joint-Newton 0.002 Perplexity Screen

## Protocol

- Task/model: GPT2-12x512, 54,682,624 parameters.
- Hardware: ABA NVIDIA A100 80GB, device 1.
- Optimizer: `EffectiveRankJointNewton`; the effective-rank floor was exactly
  0.5 at every update (no linear schedule).
- Learning rate: 0.002; weight decay: 0; effective batch size: 48
  (micro-batch 8, six gradient accumulations).
- Budget: 2,034 optimizer updates (one matched epoch); validation: 64 batches.

## Result

The run completed in 1,941.14 seconds with validation negative log-likelihood
1.11705913 and perplexity **3.05585410**.  This is worse than the static
joint-Newton 0.00125 screen (2.90923607), so 0.002 is rejected.

The constraint diagnostics prove the fixed floor: `effective_rank_floor=0.5`.
It accepted 52,630 matrix updates and skipped 21,512.  The smooth joint-Newton
branch accepted zero updates; 55,780 updates used the certified fallback with
Newton--Schulz directions.  Thus the result is valid evidence for the
finite-step-certified optimizer, but not evidence that the smooth fast path
accelerated this workload.

## Follow-up

Device 1 was immediately reused, without interrupting the independent Muown
screen on device 0, for a static fixed-half certified run at learning rate
0.00125 and momentum 0.8.  This isolates momentum from the prior 0.95 default.
