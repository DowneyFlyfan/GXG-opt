# AdamW and Muon learning-rate retune — retained NLP baseline

## Scope

- Task: cached WikiText stream, next-token validation accuracy.
- Model: `gpt_12x512`, 54,682,624 parameters.
- Optimizers: AdamW (Adam with decoupled weight decay) and Muon (Momentum Orthogonalized by Newton--Schulz).
- Training: five epochs, micro-batch size 4, gradient accumulation 2, effective batch size 8, weight decay 0.01.
- Device: local GPU.

The old baseline used learning rate `3e-4` for both optimizers.  The retune tested `6e-4` for both.  A two-epoch paired pilot established that neither optimizer diverged, then the same `6e-4` setting was run for five epochs per optimizer.

## Final validation accuracy

| Epoch | AdamW at `3e-4` | AdamW at `6e-4` | Muon at `3e-4` | Muon at `6e-4` |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.631744 | 0.682594 | 0.697289 | 0.714909 |
| 2 | 0.674274 | 0.705643 | 0.716557 | 0.731647 |
| 3 | 0.693958 | 0.718037 | 0.730698 | 0.743713 |
| 4 | 0.705212 | 0.728367 | 0.743656 | 0.752346 |
| 5 | 0.711754 | 0.734009 | 0.752193 | 0.756306 |

| Optimizer | Final gain from `3e-4` to `6e-4` | Five-epoch duration | Peak allocated memory |
| --- | ---: | ---: | ---: |
| AdamW | +0.022255 | 2 h 47 m 11 s | 4824.49 MiB |
| Muon | +0.004112 | 2 h 53 m 37 s | 4690.49 MiB |

At `6e-4`, Muon finished at 0.756306 and AdamW at 0.734009: a Muon advantage of 0.022297.  Every epoch improved for both runs.  Muon's measured duration was 3.85% greater than AdamW's, well inside the 20% timing constraint.

## Artifacts

- Formal metrics: `metrics/nlp/nlp_gpt_12x512__adamw.jsonl` and `metrics/nlp/nlp_gpt_12x512__muon.jsonl`.
- Formal result metadata: `results/nlp/nlp_gpt_12x512__adamw.json` and `results/nlp/nlp_gpt_12x512__muon.json`.
- Metric--steps: `results/nlp/nlp_gpt_12x512_metric_steps.png`.
- Metric--time: `results/nlp/nlp_gpt_12x512_metric_time.png`.  Evaluation occurs only at epoch boundaries, so each epoch-end point is placed proportionally across that run's measured total wall-clock duration.
- Paired pilot: artifacts with prefix `nlp_gpt_12x512_retune_lr0006_pilot`.

## Conclusion

Within the tested pair (`3e-4` and `6e-4`), `6e-4` is the retained learning rate for both baselines.  This result supports a stable, visibly stronger Muon baseline; it is not a claim that `6e-4` is globally optimal.
