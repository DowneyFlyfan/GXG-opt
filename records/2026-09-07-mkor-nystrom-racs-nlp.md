# GPT-2 12x512 curvature-approximation study

## Scope

This study implements and tests three approximation families on the retained
54,682,624-parameter GPT-2 12x512 Natural Language Processing task, using the
cached WikiText-103 tensors.  The retained comparison protocol uses micro-batch
12 with gradient accumulation 4 for the matrix optimizers.  The matched,
five-epoch reference curves are AdamW at learning rate 0.00015 and Muon at
learning rate 0.0025.

The implemented candidates are:

| Candidate | Approximation used in this repository |
| --- | --- |
| Momentum-Enabled Kronecker-Factor-Based Optimizer (MKOR) | Forward and backward hook statistics for eligible linear layers, rank-one Sherman-Morrison inverse-factor updates, factor refresh every 10 optimizer steps, and AdamW on non-matrix parameters. |
| Rank-Adaptive Covariance Scaling (RACS) | Matrix-only two-sided fixed-point preconditioning with exponential moving-average factors and a bounded scaling limiter; AdamW remains on auxiliary parameters. |
| Nyström Generalized Gauss-Newton (Nyström-GGN) | Matrix-free sampled-column Generalized Gauss-Newton operator, four sampled coordinates, Woodbury inverse action, damping 0.1, curvature batch 64, and refresh every four outer steps. |

## Correctness checks before GPU runs

The new focused unit tests covered Sherman-Morrison stability, the RACS
fixed-point update, dense Woodbury agreement for Nyström inversion, and the
positive-intersection guard.  The focused test invocation completed with
8 passing tests.  The initial MKOR screen exposed an unstable additive
rank-one inverse recurrence before any epoch metric; that run is retained in
`records/2026-09-07-mkor-tune-lr0003-b12-a4.log`.  Replacing it with the
subtractive Sherman-Morrison recurrence and factor reset rule produced the
stable retry below.

## Screening results

| Candidate | Settings | Completed work | Validation metric | Wall time | Peak GPU memory | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MKOR | matrix/auxiliary learning rate 0.0003 | 1 epoch / 2,034 steps | 0.297208 | 2,113.21 s | 12,856.05 MiB | Stable screen; not promoted. |
| RACS | matrix learning rate 0.02, auxiliary learning rate 0.0003, scale 0.05 | 1 epoch / 2,034 steps | 0.348836 | 1,975.61 s | 12,148.07 MiB | Stable screen; not promoted. |
| Nyström-GGN | physical batch 1, outer effective batch 3,904, curvature 64, sampled columns 4, damping 0.1, refresh 4 | 1 outer step | 0.670380 after warmup | 178.85 post-warmup s | 8,505.31 MiB | Stable screen; promoted to bounded formal trace. |

The MKOR and RACS values are one-epoch tuning screens, not five-epoch final
comparisons and not evidence of an advantage over the tuned baselines.

## Nyström-GGN warmup and formal result

The historical shared AdamW warmup checkpoint was absent, so it was regenerated
rather than silently changing initialization.  It processed 54,685,696 tokens
against a 54,682,624-token target in 1,107.79 seconds and reached validation
metric 0.656700.

The selected Nyström-GGN configuration then completed all eight requested
outer steps.  It consumed 62,464 training sequences, took 1,322.38 seconds
post-warmup (2,430.17 seconds including the measured warmup), and peaked at
8,713.90 MiB.  Its final metric was 0.670380.

Every held-out line search selected the zero step.  Thus the reported
post-warmup metric remained flat, rather than showing a curvature-driven
improvement.  Four columns were sampled at each refresh; the positive spectral
intersection retained two directions on steps 1--4 and three directions on
steps 5--8.  This is a rank-deficiency observation of the sampled curvature,
not a claim that a full rank-four factor was realized.

## Artifacts

- `metrics/nlp/nlp_gpt_12x512__mkor_tune_lr0003_retry_b12_a4.jsonl`
- `metrics/nlp/nlp_gpt_12x512__racs_tune_lr002_s005_b12_a4.jsonl`
- `metrics/nlp/nlp_gpt_12x512__nystrom_ggn_tune_r4_c64_d01_s4.jsonl`
- `metrics/nlp/nlp_gpt_12x512__nystrom_ggn_formal_r4_c64_d01_s4.jsonl`
- `results/nlp/nlp_gpt_12x512__mkor_tune_lr0003_retry_b12_a4.json`
- `results/nlp/nlp_gpt_12x512__racs_tune_lr002_s005_b12_a4.json`
- `results/nlp/nlp_gpt_12x512__nystrom_ggn_formal_r4_c64_d01_s4.json`
- `results/nlp/nystrom_ggn_formal_r4_c64_d01_s4_metric_steps.png`
- `results/nlp/nystrom_ggn_formal_r4_c64_d01_s4_metric_time.png`

## Conclusion

All three requested approximation families were implemented and exercised on
the requested model without an out-of-memory failure.  Under the tested
settings, neither one-epoch matrix screen justified a long final run, and the
formal Nyström-GGN trace found no nonzero held-out update.  The artifacts are
therefore negative but reproducible evidence; they do not support a performance
claim over the tuned AdamW or Muon baselines.
