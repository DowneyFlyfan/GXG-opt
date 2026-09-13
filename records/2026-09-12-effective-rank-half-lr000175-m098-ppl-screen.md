# Fixed-rank-half learning-rate 0.00175 PPL screen

## Protocol

- Model: GPT2-12x512 (54,682,624 parameters), NLP task.
- Hardware: ABA A100 device 1.
- Optimizer: `effective_rank_half`; rank floor fixed at 0.5 throughout.
- Learning rate: 0.00175; momentum: 0.98; weight decay: 0.
- Matched budget: micro-batch 8, gradient accumulation 6 (effective batch 48),
  2,034 optimizer updates, and 64 validation batches.

## Result

The screen completed in 1,946.718 seconds. Validation negative log likelihood
was 1.03721420, corresponding to perplexity **2.82134636**. The strict
constraint accepted 61,651 matrix updates, skipped none, and reported an exact
rank floor of 0.5. Newton--Schulz supplied 77,292 directions with no fallback
or joint-Newton step.

This is the current best strict-rank-half epoch-one result. It improves the
otherwise matched LR-0.0015 screen (2.83628843) by 0.01494207 PPL and the
LR-0.00125 screen (2.87833246) by 0.05698610 PPL. The next screen increases
learning rate to 0.002 while retaining momentum 0.98 and the fixed 0.5 floor.

## Five-epoch confirmation

The selected setting completed the required five matched epochs on ABA A100
device 1 as `aba_a100_effective_rank_half_lr000175_m098_final_e5`. Validation
perplexity improved monotonically across the recorded epoch ends:

- step 2,034: 2.82134636
- step 4,068: 2.57333840
- step 6,102: 2.46951440
- step 8,136: 2.40079624
- step 10,170: **2.37412280**

The complete run took 9,783.713 seconds and reached validation negative
log-likelihood 0.86462802. It preserved the exact 0.5 rank floor throughout,
accepted 277,260 constrained matrix updates, skipped none, and recorded zero
projection, certified-fallback, or joint-Newton steps. The final perplexity
improves the previous strict rank-half five-epoch result of 2.386942 by
0.012819, but it does not beat the matched Muon baseline (2.280087).
