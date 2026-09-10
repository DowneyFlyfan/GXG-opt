# Final-result baseline refresh

## Scope

All retained GPT-12x512 Natural Language Processing comparison figures were
redrawn against the best matched-batch AdamW and Muon baselines.  Tuning-only
figures were not regenerated and remain absent; their metric JSON Lines,
result JSON, and logs remain intact.

Computer Vision, Audio, and SmolLM2 figures were preserved unchanged.  Their
models and datasets differ from GPT-12x512, so replacing their curves with
these language-model baselines would make the comparison invalid.

## Selected matched-batch baselines

| Optimizer | Learning rate | Epoch-5 validation next-token accuracy | Measured duration |
| --- | ---: | ---: | ---: |
| AdamW | 0.00015 | 0.723747 | 9775.80 s |
| Muon | 0.0025 | 0.755032 | 9886.20 s |

These are the selected five-epoch winners in the common micro-batch 12,
gradient-accumulation 4 protocol. The paired baseline figures are:

- `results/nlp/nlp_gpt_12x512_final_baselines_metric_steps.png`
- `results/nlp/nlp_gpt_12x512_final_baselines_metric_time.png`

## Outputs refreshed

`src/regenerate_nlp_baseline_plots.py` regenerated 20 final Natural Language
Processing PNG files: 18 candidate Metric-Steps / Metric-Time figures and the
two aggregate Gauss--Newton figures.  The final paired baseline figures were
also redrawn from these selected traces.

## Verification

- The refreshed Hybrid Stiefel-Muon figure visibly contains AdamW (0.00015),
  Muon (0.0025), and the candidate trace.
- The final paired baseline figure visibly contains the selected matched-batch
  curves and labels.
- `pytest -q tests/test_regenerate_nlp_baseline_plots.py
  tests/test_lr_tuning_plot.py tests/test_finalize_literature_tuning.py
  tests/test_artifacts.py` passed: 10 tests.
