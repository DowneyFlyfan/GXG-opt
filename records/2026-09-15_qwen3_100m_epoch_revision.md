# Qwen3-0.6B scratch baseline revision: 100 million tokens per epoch

## User-directed stop

At the user's request, the active batch-64 tuning workers were terminated on 2026-09-15 before restarting any trial:

- ABA A100 Muon worker 1348622 and dispatcher 1348615, last complete metric: step 500, perplexity 57.881917.
- ABA A100 Muown worker 1348623 and dispatcher 1348621, last complete metric: step 400, perplexity 65.355826.
- Local AdamW worker 1073992 and dispatcher 1073989, last complete metric: step 100, perplexity 810.922404.

Their traces remain preserved as interrupted tuning evidence and are not used to select final 100-million-token-epoch rates.

## Revised epoch definition

The full verified FineWeb-Edu cache remains the source. `--train-tokens-per-epoch 100000000` limits each epoch to its deterministic first 100,000,000 training tokens without copying a second 400 MB cache. With sequence length 2,048 and effective batch 64, one epoch contains 48,828 complete blocks, or 762 complete updates. The incomplete tail is deliberately excluded, leaving 99,876,864 tokens actually used in optimizer updates per epoch.

Five formal epochs therefore contain 3,810 updates and 499,384,320 trained tokens. The command line records the requested 100,000,000-token budget in every checkpoint and result configuration.

## Observed A100 timing projection

The stopped batch-64 trials used the same Qwen3 architecture, sequence length, micro-batch 8, and accumulation 8 as the revision. Their elapsed times include scheduled validation:

| Optimizer | Observed updates | Observed elapsed time | Projected 762-update epoch |
| --- | ---: | ---: | ---: |
| AdamW | 100 | 976.39 s | 7,444.10 s (2.07 h) |
| Muon | 500 | 5,030.21 s | 7,666.04 s (2.13 h) |
| Muown | 400 | 4,072.98 s | 7,759.03 s (2.16 h) |

All three projections are below four hours. AdamW's projection comes from a fresh A100 run with the revised `--train-tokens-per-epoch 100000000` configuration: 100 updates took 976.39 seconds and finished successfully. The older local RTX 5070 Ti observation projects to 6.14 h and is not used for the A100 protocol.

## Verification

- Added and passed a loader test that limits the training epoch while retaining validation coverage.
- Added and passed a trial test showing that the configured token budget produces the expected completed-update count.
- Targeted Qwen data and experiment tests: 27 passed.

## Revised tuning evidence

The first AdamW candidate (`learning_rate=0.001`) reached step 100 with validation perplexity 795.703380, down from scratch initialization perplexity 183646.368218. It has processed 13,107,200 tokens, uses the requested 100,000,000-token epoch cap, and remains active. This is an in-progress candidate result, not a final rate selection.

At step 200, the same candidate reached perplexity 366.090230 after 26,214,400 processed tokens, improving from its step-100 result. The worker remains active and selection is still pending all candidates.

## Superseded by the five-epoch two-hour constraint

The user clarified that the entire five-epoch training run must finish within two hours on one A100. The 100,000,000-token-per-epoch plan would require roughly ten hours across five epochs, so its live AdamW candidate was stopped after the verified step-200 checkpoint.

The replacement protocol uses 16,000,000 requested tokens per epoch: 122 complete batch-64 updates and 15,990,784 trained tokens per epoch. Five epochs are 610 updates and 79,953,920 trained tokens. Based on measured A100 timings that include scheduled validation, the projected full five-epoch times are 1.65 h (AdamW), 1.70 h (Muon), and 1.73 h (Muown). This gives a margin below the two-hour ceiling while preserving five true epochs.
