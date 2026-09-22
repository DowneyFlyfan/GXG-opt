# Qwen3 scratch baseline tuning: five epochs within two A100 hours

## Fixed formal budget

Each formal run uses scratch initialization, micro-batch 8, gradient accumulation 8, and five epochs of 16,000,000 requested tokens. The loader keeps 122 complete batch-64 updates per epoch, for 610 updates and 79,953,920 trained tokens across the five epochs. It evaluates every 100 updates for the required curves, but writes the 6.7-GiB checkpoint only at the final update.

## Timing evidence

The completed AdamW 0.001 screen took 3,041.076 seconds for 300 updates, including three checkpoint writes. A conservative linear scale predicts 6,183.521 seconds (1.718 hours) for 610 updates; final runs save only once and therefore have additional I/O margin. Earlier A100 calibrations project 1.70 hours for Muon and 1.73 hours for Muown. Each formal result must still demonstrate an actual duration below two hours.

## Rate screens

Rate screens use 300 updates with the formal 610-update warmup/cosine schedule and the same 16,000,000-token epoch boundary. AdamW tests learning rates 0.001, 0.003, and 0.006. Muon tests auxiliary learning rates 0.001 and 0.003 at matrix learning rate 0.003. Muown tests auxiliary learning rates 0.001, 0.003, and 0.01 at direction learning rate 0.01 and gain learning rate 0.0003.

The final rate for each optimizer will be chosen only after that optimizer's screens complete. Each selected final run restarts from the exact scratch weights and retains its checkpoint, complete metric trace, held-out evaluation, and matched step/time plots.

## In-progress evidence

AdamW learning rate 0.001 reached step 200 at validation perplexity 360.012015, down from 789.382990 at step 100 and scratch initialization perplexity 183646.368218. It processed 26,214,400 tokens and remains active. This is not a rate-selection result.
