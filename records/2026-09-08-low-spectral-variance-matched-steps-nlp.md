# Matched-Step Low-Spectral-Variance GPT-2 12x512

## Contract

The run used the cached NLP corpus, GPT-2 12x512 with 54,682,624 parameters,
micro-batch 12, accumulation 4, and five epochs.  This gives 2,034 optimizer
updates per epoch and 10,170 updates total, matching the retained AdamW and
Muon baseline schedule exactly.

## Candidate settings

- Optimizer: Low-Spectral-Variance with 36 constrained matrix parameters.
- Learning rate: 0.003; momentum: 0.95; dual steps: 1; condition cap: 8.
- Other parameters: AdamW auxiliary group, as required when a matrix is not
  initially feasible for the spectral constraint.
- Peak allocated CUDA memory: 12,306.9 MiB of 16,303 MiB.

## Resume integrity

The first fifth-epoch invocation exhausted a cumulative 14,300-second guard at
step 8,532.  Its provisional metric was removed from the final JSONL.  The
runner was corrected to derive a partial-epoch cursor from the checkpoint and
skip the already-applied accumulation windows.  The restarted final invocation
continued from step 8,532 to 10,170 without repeating those windows.

## Observed metrics

| Optimizer | Final metric | Optimizer updates | Reported time |
| --- | ---: | ---: | ---: |
| AdamW (1.5e-4) | 0.723747 | 10,170 | 9,775.8 s |
| Muon (2.5e-3) | 0.755032 | 10,170 | 9,886.2 s |
| Low-Spectral-Variance | 0.663251 | 10,170 | 17,185.4 s |

The Low-Spectral-Variance metrics at epochs one through five were 0.396454,
0.555147, 0.612281, 0.640541, and 0.663251.  The optimizer trained and improved
throughout the matched schedule, but this experiment does not demonstrate a
gain over either tuned baseline and is slower in recorded wall-clock time.

## Artifacts

- `metrics/nlp/nlp_gpt_12x512__low_spectral_variance_matched_steps_c8_lr003_b12_a4.jsonl`
- `results/nlp/nlp_gpt_12x512__low_spectral_variance_matched_steps_c8_lr003_b12_a4.json`
- `results/nlp/low_spectral_variance_matched_steps_c8_lr003_b12_a4_metric_steps.png`
- `results/nlp/low_spectral_variance_matched_steps_c8_lr003_b12_a4_metric_time.png`
