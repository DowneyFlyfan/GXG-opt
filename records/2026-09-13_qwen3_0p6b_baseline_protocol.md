# Qwen3-0.6B baseline protocol

## Status

The token cache and memory admission completed on ABA.  Baseline tuning and formal training have not started, so this record contains no optimizer-comparison result.

## Fixed workload

- Model: `Qwen/Qwen3-0.6B`, 596,049,920 parameters, loaded from `.cache/huggingface/models/Qwen3-0.6B`.
- Dataset: FineWeb-Edu `sample-10BT`, source revision captured at preparation time in the cache manifest.
- Token cache: 2,000,000,000 train tokens and 100,000,000 held-out validation tokens, packed as little-endian `uint32` under `.cache/qwen3_0p6b_fineweb_edu`.
- Sequence length: 2,048 tokens.  The loader drops only incomplete trailing blocks and shifts labels by one token.
- Formal budget: three equal data passes for each optimizer, contingent on memory admission and learning-rate screens.
- Metric: held-out next-token perplexity at completed optimizer steps and wall-clock time.

## Fair-comparison rules

- All candidates use the same model snapshot, token cache manifest, deterministic seed, sequence length, precision, scheduler, and effective token batch.
- AdamW updates all parameters.  Muon and Muown update only interior Qwen projection matrices in layers 1–26; embeddings/tied head, boundary layers, and one-dimensional tensors use AdamW.
- AdamW and Muon learning rates receive equal-budget screens.  Muown direction and gain rates are separately tuned; a shared-rate Muown run is not admissible.
- Weight decay is recorded and held consistent within each candidate family.  Checkpoints remain below `.cache/qwen3_0p6b/checkpoints`.

## Host preflight

- Host: ABA (`ecepxiegpu1.ucsd.edu`), two NVIDIA A100 80-GB devices.
- At launch preflight both GPUs were idle (4 MiB and 1 MiB allocated).  No process was terminated.
- A Qwen 2,048-token forward pass on GPU 0 allocated 5.552 GiB before backward/optimizer state.
- Root filesystem availability before model download was 63 GiB; model snapshot occupies 1.50 GB and remains within the project cache.

## Verified cache

- Completed in 1 hour 3 minutes using deterministic streaming selection.
- Manifest: `.cache/qwen3_0p6b_fineweb_edu/manifest.json` on ABA.
- Train stream: 2,000,000,000 tokens, 8,000,000,000 bytes, SHA-256 `8d019ccb133a030652f7bea00b4205d28ce8f7d5fe7a04be67dab0840505a866`.
- Validation stream: 100,000,000 tokens, 400,000,000 bytes, SHA-256 `2971d4b2c7418e841516f8ddca6cc42f0d858294772b488533f1ede7875dc917`.
- The verified temporary FineWeb cache was 4 KiB and was removed after digest verification.  No raw dataset shard is retained.

## A100 memory admission

All admission runs used the verified cache, sequence length 2,048, one optimizer update, one validation batch, and A100 80-GB hardware.  Their one-step perplexities are diagnostic only and must not be compared as trained-model quality.

| Optimizer | Microbatch | Peak allocated memory | One-update time | Outcome |
| --- | ---: | ---: | ---: | --- |
| AdamW | 1 | 9,249 MiB | 0.625 s | completed |
| AdamW | 8 | 65,944 MiB | 1.322 s | completed |
| AdamW | 9 | 74,048 MiB (72.31 GiB, 91.2% of 79.25 GiB usable) | 1.342 s | completed |
| AdamW | 16 | 72,040 MiB allocated, then failed requesting 18.55 GiB | — | CUDA out-of-memory |
| Muon | 9 | 74,048 MiB | 10.442 s | completed |
| Muown | 9 | 74,048 MiB | 10.722 s | completed |

The accepted common microbatch is **9**.  It satisfies the same-effective-batch constraint and approaches A100 capacity without the AdamW batch-16 failure.  Formal runs start with gradient accumulation one (effective batch nine sequences, 18,432 tokens/update); if a tuned schedule needs a larger effective batch, every optimizer uses the same accumulation multiplier.

The admission checkpoints, JSON traces, and logs were removed after this table was recorded; they were calibration artifacts, not trained model checkpoints.

## Initial screen protocol

1. Verify cache digests and remove temporary FineWeb download state after packing.
2. Admit the largest safe microbatch below 90% A100 allocation for every route; use accumulation to equalize effective batch.
3. Run equal-token learning-rate screens on the two idle A100 devices, then start the three matched formal baselines.
4. Record nonfinite/failed candidates as failures; do not replace them with a baseline result.
