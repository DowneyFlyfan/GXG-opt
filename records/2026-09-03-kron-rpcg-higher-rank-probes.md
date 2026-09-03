# Higher-rank recycled Kronecker-RPCG probes

## Question

Test whether increasing the recycled low-rank residual correction beyond the
previous rank-4 progressive Kronecker recursive preconditioned conjugate
gradient (Kron-RPCG) configuration improves the retained NLP experiment.

## Controlled protocol

All compared runs use the same warmed `gpt_12x512` model, seed, outer
effective batch 3,904, curvature batch 64, damping 0.1, 12 PCG iterations,
relative tolerance 0.05, correction refresh interval 4, bfloat16 basis,
line-search scale 4.0, and 260 screening sequences.  Only correction rank is
changed.

The historical rank-4 run used physical batch 12.  On the current local GPU,
rank-6 exhausted memory during the first full GGN curvature product even at
physical batches 12, 10, and 4.  This occurs before the residual correction
is built.  Physical batch 1 resolves the activation peak while preserving the
same accumulated outer and curvature sample counts, so the rank comparison
below uses physical batch 1.

## Results after one outer update

| Rank | Validation metric | Post-warmup seconds | Peak MiB | Curvature matvecs |
|---:|---:|---:|---:|---:|
| 4 historical (`b=12`) | 0.666389 | 483.631 | 12,768 | 18 |
| 6 (`b=1`) | 0.678513 | 257.952 | 10,384 | 20 |
| 8 (`b=1`) | 0.677765 | 270.130 | 10,593 | 22 |

Both higher-rank probes exceed the historical rank-4 first-step metric.  Rank
6 is 0.000748 above rank 8 while needing two fewer averaged GGN-vector
products and 208 MiB less peak memory, so rank 6 is the preferred next formal
candidate.  This is a one-step qualification result, not evidence of a
long-horizon advantage.

## Artifacts

- `metrics/nlp/nlp_gpt_12x512__kron_rpcg_progressive_r6_c64_pcg12_d01_s4_b1_probe.jsonl`
- `metrics/nlp/nlp_gpt_12x512__kron_rpcg_progressive_r8_c64_pcg12_d01_s4_b1_probe.jsonl`
- `results/nlp/kron_rpcg_progressive_r6_c64_pcg12_d01_s4_b1_probe_metric_steps.png`
- `results/nlp/kron_rpcg_progressive_r8_c64_pcg12_d01_s4_b1_probe_metric_steps.png`
