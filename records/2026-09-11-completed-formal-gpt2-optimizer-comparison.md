# Completed formal GPT2-12x512 optimizer comparison

## Scope

This visualization compares only completed five-epoch formal GPT2-12x512
experiments already present in the repository. It is a retrospective plot: it
does not add a training result, and it excludes all one-epoch tuning screens
and the still-running effective-rank-third screen.

## Included evidence

| Optimizer | Final validation next-token accuracy | Wall-clock time |
| --- | ---: | ---: |
| AdamW (1.5e-4) | 0.72374725 | 2.72 h |
| Muon (2.5e-3) | 0.75503159 | 2.75 h |
| Muown (5e-3, zero decay) | **0.75617790** | 2.72 h |
| Multi-Step Spectral (4e-4) | 0.74449921 | 2.96 h |
| Effective-Rank-Half (1.25e-3) | 0.75524139 | 4.53 h |
| Hybrid Stiefel-Muon | 0.72645442 | 3.66 h |

The three baseline paths are resolved through `gpt_baseline_selection.py`, and
each candidate uses its formal completed result and JSONL metric file. All six
result files have `status: completed` and either explicit five epochs or the
equivalent 10,170 completed optimizer steps. The hybrid trace contains 44
intermediate measurements; all other included traces contain five epoch-end
measurements, which is intentionally preserved in the plot rather than
resampled.

## Artifacts

- `results/nlp/gpt2_12x512_completed_formal_optimizers_metric_steps.png`
- `results/nlp/gpt2_12x512_completed_formal_optimizers_metric_time.png`
- `results/nlp/gpt2_12x512_completed_formal_optimizers.json`

The figures establish Muown as the best completed final among these six runs
by 0.00093651 over Effective-Rank-Half and 0.00114632 over Muon. They do not
establish the yet-unmeasured one-third constraint as better or worse.
