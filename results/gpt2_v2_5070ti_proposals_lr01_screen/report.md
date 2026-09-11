# GPT-2 optimizer 2.0 comparison

Fixed hyperparameter screen; these results do not establish superiority or novelty.

| Method | Seed | Epochs completed | Updates | Validation NLL | Peak GiB |
|---|---:|---:|---:|---:|---:|
| qk_defect_v1 | 0 | 0 | 33 | 8.072368 | 5.527 |
| routing_resistance_v1 | 0 | 0 | 33 | 8.072426 | 5.530 |
| tied_path_curvature_v1 | 0 | 0 | 33 | 8.072309 | 5.818 |
| proposal_notch_v1 | 0 | 0 | 33 | 8.072368 | 5.742 |
| feature_remap_cohort_v1 | 0 | 0 | 33 | 8.071451 | 5.807 |
| ln_response_v1 | 0 | 0 | 33 | 8.071860 | 5.530 |

This directory contains bounded smoke runs, not the five-epoch comparison.
