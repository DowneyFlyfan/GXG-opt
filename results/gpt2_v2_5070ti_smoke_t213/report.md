# GPT-2 optimizer 2.0 comparison

Fixed hyperparameter screen; these results do not establish superiority or novelty.

| Method | Seed | Epochs completed | Updates | Validation NLL | Peak GiB |
|---|---:|---:|---:|---:|---:|
| adamw | 0 | 0 | 33 | 9.641650 | 5.826 |
| muon | 0 | 0 | 33 | 10.842623 | 5.567 |
| qk_defect_v1 | 0 | 0 | 33 | 10.842623 | 5.530 |
| routing_resistance_v1 | 0 | 0 | 33 | 10.842601 | 5.530 |
| tied_path_curvature_v1 | 0 | 0 | 33 | 10.842613 | 5.818 |
| proposal_notch_v1 | 0 | 0 | 33 | 10.842623 | 5.742 |
| feature_remap_cohort_v1 | 0 | 0 | 33 | 10.842557 | 5.807 |
| ln_response_v1 | 0 | 0 | 33 | 10.842497 | 5.530 |

This directory contains bounded smoke runs, not the five-epoch comparison.
