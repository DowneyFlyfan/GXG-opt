# GPT-12x512 learning-rate tuning

## Fixed protocol

- Model: GPT-12x512 (54.68M parameters); WikiText byte stream cached under `.cache`.
- Full candidates use five epochs; high-end boundary probes may stop after one epoch. All use micro-batch 12, gradient accumulation 4, effective batch 48, and weight decay 0.01.
- Metric: validation next-token accuracy. No loss curves are used.
- Each listed candidate ran alone on the local GPU. All tuning metric/result data are retained; no tuning-sweep figures are generated.

## Completed candidates

| Optimizer | Learning rate | Epochs | Final validation accuracy | Seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| muon | 0.0025 | 5 | 0.755032 | 9886.2 | 12289.9 |
| muon | 0.0035 | 1 | 0.710566 | 1971.4 | 12289.9 |
| muon | 0.005 | 5 | 0.750230 | 9844.8 | 12289.9 |
| muon | 0.02 | 1 | 0.286447 | 1966.2 | 12289.9 |
| adamw | 0.0003 | 1 | 0.635850 | 1955.3 | 12436.7 |
| adamw | 0.0003 (full) | 5 | 0.721696 | 9776.2 | 12436.7 |
| adamw | 0.00015 | 1 | 0.650279 | 1956.0 | 12436.7 |
| adamw | 0.00015 (full) | 5 | 0.723747 | 9775.8 | 12436.7 |
| adamw | 0.000075 | 1 | 0.606626 | 1955.6 | 12436.7 |
| adamw | 0.0005 | 1 | 0.612691 | 1955.6 | 12436.7 |
| adamw | 0.0008 | 1 | 0.632266 | 1955.3 | 12436.7 |

## Selected settings

- muon: Muon lr=0.0025, final validation accuracy 0.755032.
- adamw: AdamW lr=0.00015 (full), final validation accuracy 0.723747.

## Incomplete candidates

- Muon lr=0.00125
- Muon lr=0.01

## Final-baseline figures

- `results/nlp/nlp_gpt_12x512_final_baselines_metric_steps.png`
- `results/nlp/nlp_gpt_12x512_final_baselines_metric_time.png`
