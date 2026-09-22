# Qwen3 effective-batch-64 transition

The user replaced the project instructions: natural language processing now requires equivalent batch size 64. Earlier runs used 48 and are retained as historical evidence, not final compliant baselines.

## Current-state inspection

- Muon batch-48 formal worker 520877 was still active. Its latest saved evaluation was step 5500, perplexity 29.08216255386831. After verifying its exact command, SIGTERM was sent; device memory was released. Its trace and checkpoint are preserved.
- Muown batch-48 confirmation dispatcher and worker had exited. All three trials completed 1000 updates: auxiliary 0.001 -> 42.212409682333316; 0.003 -> 41.877291638048725; 0.01 -> 42.14951158547586.
- Local AdamW batch-48 dispatcher and worker were absent. Rate 0.001 completed (59.64168998757437). Rate 0.003 has a partial trace through step 400 (158.97029494108864), without a completed result. The reason for that earlier termination is not established. No recovery at batch 48 is planned.

## New protocol

Repeat the finalist learning-rate brackets from the same random seed and full data at batch 64. Do not resume any batch-48 checkpoint. Same sequence length 2048; 131072 tokens/update. Retain all old tuning data. Memory-check the A100 micro-batch configuration before launching long runs. Local fallback may use micro-batch 8, accumulation 8 with activation checkpointing, previously validated at the same micro-batch size. Final baseline timing comparisons must use the same A100 configuration.

Five full epochs contain 15258 complete batch-64 updates per epoch, 76290 total, with the final incomplete batch dropped consistently. Final formal selection will be based on the new batch-64 trials.

## Memory validation and launch

Two-update probes completed for both matrix optimizers at effective batch 64. Micro-batch 32, accumulation 2, activation checkpointing used 21240.27 MiB (Muon) / 22805.14 MiB (Muown), with total probe times 42.38 / 42.33 seconds. Micro-batch 8, accumulation 8, no activation checkpointing used 45493.82 / 47058.28 MiB, with total probe times 37.81 / 41.11 seconds. These short timings include evaluation/checkpoint overhead and are not throughput benchmarks.

Choose micro-batch 8 without activation checkpointing on A100: largest known safe divisor of 64 with this configuration; prior micro-batch 16 failed near the memory limit. Batch-64 divisibility prevents retaining micro-batch 12. Not claiming full memory saturation. Probe checkpoints were removed after successful results were validated; configuration, metrics and results remain.

New 1000-update comparisons use identical effective batch 64, 96 validation blocks, scratch seed 1337, warmup 100, cosine horizon 2000, clipping 1, weight decay 0.1. Muon dispatcher 1348615 uses device 0; Muown dispatcher 1348621 uses device 1. Local AdamW dispatcher 1073989 uses micro-batch 8, accumulation 8 with activation checkpointing. Local timing remains excluded from final A100 comparisons.

The formal protocol is records/2026-09-15_qwen3_batch64_five_epoch_protocol.json. No compliant five-epoch formal run has started yet. Fresh scratch starts will follow completed batch-64 parameter selection.

## First batch-64 tuning evaluations: step 100

- muon: perplexity 527.473209, elapsed 984.08 seconds.
- muown: perplexity 476.487874, elapsed 992.81 seconds.

Each run has processed 13,107,200 tokens. Scratch weight and data fingerprints match. These are initial auxiliary-rate-0.001 candidates, not selected final rates. All three trial workers remain active; local AdamW has not reached its first 100-update evaluation.

## Batch-64 tuning evaluations: step 200

- muon: perplexity 180.700880, elapsed 1971.42 seconds.
- muown: perplexity 155.879432, elapsed 1990.54 seconds.

Each run has processed 26,214,400 tokens. Fingerprints match and both validation perplexities improved since step 100. These auxiliary-rate-0.001 trials remain in progress; no final rate is selected.

## Local batch-64 AdamW rate 0.001: step 100

Validation perplexity improved from 183646.630927 at initialization to 810.922404 at step 100. Exposure is 13,107,200 tokens; elapsed time is 2898.96 seconds. Initial weights and data fingerprints match. Worker 1073992 remains active, with 900 updates left in this trial. Rates 0.003 and 0.006 remain queued; selection is pending. Local timings are not used for the final A100 timing comparison.

## Batch-64 muon auxiliary-rate-0.001 trial: step 300

Perplexity 89.963178; elapsed 2968.87 seconds. Exposure is 39,321,600 tokens; initial weights/data fingerprints match. Validation perplexity continues improving. Trial and rate selection remain incomplete.

## Batch-64 Muown auxiliary-rate-0.001 trial: step 300

Perplexity 84.701074; elapsed 3021.38 seconds. Exposure is 39,321,600 tokens; initial weights/data fingerprints match. Validation perplexity improved from 155.879432 at step 200. Trial and rate selection remain incomplete.

## Batch-64 Muon auxiliary-rate-0.001 trial: step 400

Perplexity 68.533591; elapsed 3991.12 seconds. Exposure is 52,428,800 tokens; initial weights/data fingerprints match. Validation perplexity improved from 89.963178 at step 300. Trial and rate selection remain incomplete.

## Batch-64 Muown auxiliary-rate-0.001 trial: step 400

Perplexity 65.355826; elapsed 4072.98 seconds. Exposure is 52,428,800 tokens; initial weights/data fingerprints match. Validation perplexity improved from 84.701074 at step 300. Trial and rate selection remain incomplete.

## Batch-64 Muon auxiliary-rate-0.001 trial: step 500

Perplexity 57.881917; elapsed 5030.21 seconds. Exposure is 65,536,000 tokens; initial weights/data fingerprints match. Validation perplexity improved from 68.533591 at step 400. The trial remains active, so this does not select a final rate.
