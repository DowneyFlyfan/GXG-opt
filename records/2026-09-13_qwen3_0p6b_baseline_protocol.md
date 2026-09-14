# Qwen3-0.6B baseline protocol

## Status

The token cache, memory admission, and learning-rate selection completed on
ABA.  The matched three-epoch AdamW and Muon formal baselines are active on
separate A100 80GB devices.  Muown has not yet launched because both A100s are
occupied; it will use the already selected independent direction, gain, and
auxiliary rates when one device becomes free.  No formal baseline result is
claimed until all three runs have completed their equal three-epoch protocol.

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

The one-step batch-9 result was not stable: both a 20-step AdamW and a 20-step Muon run failed in backward with CUDA out-of-memory, requesting 10.43 GiB while roughly 78 GiB was already occupied.  The accepted common microbatch is therefore **8**, using `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.  It gives a stable 20-step peak of 68,226 MiB (AdamW) and 67,446 MiB (Muon), roughly 84% of usable device memory.  Formal runs start with gradient accumulation one (effective batch eight sequences, 16,384 tokens/update); any larger effective batch uses the same accumulation multiplier for every optimizer.

## First rate screen

Both 20-update screens used microbatch eight, the same cache manifest, one deterministic seed, and 64 fixed validation batches.  These are elimination measurements, not final baseline curves.

| Optimizer | Rate(s) | Validation perplexity after 20 updates | Time for 20 updates | Decision |
| --- | --- | ---: | ---: | --- |
| AdamW | learning rate `1e-4` | 19.759 | 17.551 s | lower-rate screen required |
| Muon | learning rate `2.5e-3`, AdamW auxiliary `3e-4` | 135.540 | 22.514 s | reject this rate; screen one order lower |

The cache's one-update diagnostic perplexity was approximately 13–17 depending on its initial deterministic batch.  It is not a 64-batch zero-update reference, so these 20-step values only establish that the selected high rates are unsuitable relative to lower-rate candidates; they do not quantify improvement from initialization.

The admission checkpoints, JSON traces, and logs were removed after this table was recorded; they were calibration artifacts, not trained model checkpoints.

## Second rate screen

Both 50-update screens retained the accepted common microbatch eight, the same cache manifest, deterministic seed, and 64 fixed validation batches.

The matched zero-update measurement is **22.720 perplexity**.  It used the unmodified cached Qwen snapshot, microbatch eight, and the same first 64 unshuffled validation batches.  It is the reference for all screens below.

| Optimizer | Rate(s) | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | --- | ---: | ---: | --- |
| AdamW | learning rate `3e-5` | 17.629 | 42.468 s | improves 5.090 perplexity from initialization; retain provisionally and test `1e-5` |
| Muon | learning rate `2.5e-4`, AdamW auxiliary `3e-5` | 24.519 | 50.354 s | degrades 1.800 perplexity; lower Muon direction rate required |

The two result files reported all 50 completed updates and finite validation perplexity.  These are short tuning screens only; neither is a formal baseline.  Their remote checkpoints (3.58 GB AdamW and 2.76 GB Muon) and logs are temporary calibration artifacts and are removed after recording.

## Third rate screen

Muon with a tenfold lower matrix-direction rate completed all 50 updates under the same accepted batch-eight protocol.

| Optimizer | Rate(s) | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | --- | ---: | ---: | --- |
| Muon | matrix learning rate `5e-5`, AdamW auxiliary `3e-5` | 17.689 | 50.078 s | improves 5.030 perplexity from initialization; retain provisionally and bracket near this rate |

This corrects the earlier Muon conclusion: the failure was an over-large matrix direction rate, not evidence against the routed Muon baseline.  The corresponding 2.76-GB temporary checkpoint and log are removed after recording.

## Fourth rate screen

The upper side of the retained Muon direction rate was checked with the same 50-update protocol.

| Optimizer | Rate(s) | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | --- | ---: | ---: | --- |
| Muon | matrix learning rate `1e-4`, AdamW auxiliary `3e-5` | 17.924 | 50.183 s | improves from initialization but is 0.235 worse than `5e-5`; retain `5e-5` |

The checkpoint, metric trace, result, and log for this bracket point are removed after the result is recorded.

## Fifth rate screen

The upper side of the retained AdamW rate was checked under the same 50-update protocol.

| Optimizer | Rate(s) | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | --- | ---: | ---: | --- |
| AdamW | learning rate `5e-5` | 17.671 | 43.383 s | improves from initialization but is 0.042 worse than `3e-5`; retain `3e-5` |

The checkpoint, metric trace, result, and log for this AdamW bracket point are removed after the result is recorded.

## First Muown two-rate screen

Muown was screened with independent tangent-direction and row-gain learning rates; it does not reuse a single scalar rate.

| Optimizer | Rate(s) | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | --- | ---: | ---: | --- |
| Muown | direction `5e-5`, gain `1e-5`, AdamW auxiliary `3e-5` | 17.689 | 52.106 s | improves 5.030 perplexity from initialization; retain direction and bracket gain |

This initial pair is within 0.00005 perplexity of the retained Muon screen, so gain-rate tuning is warranted.  The checkpoint, metric trace, result, and log are removed after recording.

## Muown gain-rate bracket

The Muown direction rate remained `5e-5` and the AdamW auxiliary rate remained `3e-5`; the two available A100 devices screened the gain rate concurrently.

| Optimizer | Direction rate | Gain rate | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Muown | `5e-5` | `3e-6` | 17.687 | 50.922 s | retain provisionally |
| Muown | `5e-5` | `3e-5` | 17.690 | 51.217 s | 0.003 worse than `3e-6`; reject |

Both values improve from the 22.720 initialization reference.  The gain effect is small at this horizon, but the evidence supports `3e-6` for the next independent direction bracket.  The two checkpoints, metric traces, results, and logs are removed after recording.

## Muown direction-rate bracket

The selected gain rate remained `3e-6` and AdamW auxiliary rate remained `3e-5`; the direction rate was bracketed concurrently.

| Optimizer | Direction rate | Gain rate | Validation perplexity after 50 updates | Time for 50 updates | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Muown | `3e-5` | `3e-6` | 17.867 | 51.628 s | reject |
| Muown | `5e-5` | `3e-6` | 17.687 | 50.922 s | retain |
| Muown | `7e-5` | `3e-6` | 17.701 | 51.735 s | reject |

The selected two-rate Muown setting is direction `5e-5`, gain `3e-6`, auxiliary `3e-5`.  It is finite and improves 5.033 perplexity from initialization.  Its checkpoint, metric trace, result, and log, and those of the rejected bracket points are temporary screening artifacts and are removed after recording.

## Selected formal settings and live evidence

The retained settings are AdamW learning rate `3e-5`; Muon matrix learning rate
`5e-5` with auxiliary AdamW rate `3e-5`; and Muown direction rate `5e-5`, gain
rate `3e-6`, and auxiliary AdamW rate `3e-5`.  The common formal command
contract is three epochs, microbatch eight, gradient accumulation one, 64
validation batches, evaluation every 1,000 completed updates, no data workers,
and seed 1,337.  The runner's default weight decay is `0.1`, held equal across
these formal baseline routes.

The active commands are:

```text
python src/run_qwen3_ppl.py run --optimizer adamw --learning-rate 3e-5 ...
python src/run_qwen3_ppl.py run --optimizer muon --learning-rate 5e-5 --auxiliary-lr 3e-5 ...
```

At the latest direct process inspection, both corresponding Python processes
were live and each A100 reported 100% utilization.  The latest durable metric
records were AdamW perplexity `15.5593` at step 13,000 and Muon perplexity
`15.6546` at step 11,000, both using manifest digest
`ced8d55d8e0849b22e5c6678194d0ca31fe42998b0760fea8c6d99ab00090649`.
These unequal step counts are ordinary elapsed-runtime progress, not different
training budgets.  The eventual Muown launch command must explicitly include
all three separate rates:

```text
python src/run_qwen3_ppl.py run --optimizer muown --direction-lr 5e-5 \
  --gain-lr 3e-6 --auxiliary-lr 3e-5 ...
```

## Final full-validation evidence

The periodic curves deliberately use a fixed 64-batch validation slice
(1,048,576 token predictions at microbatch eight) so that every 1,000-update
record has practical wall-clock cost.  This slice is not substituted for the
entire held-out cache in the final baseline decision.  The
`evaluate-checkpoint` command loads a completed checkpoint, verifies its
manifest digest, and evaluates every packed validation block, writing a
separate immutable JSON result beside the normal compact result:

```text
python src/run_qwen3_ppl.py evaluate-checkpoint --optimizer adamw \
  --run-label formal_3epoch_b8_v64_i1000 --device cuda
```

For this cache, full validation covers 48,828 complete 2,048-token blocks,
99,999,744 token predictions, and 6,104 loader batches at formal microbatch
eight (the final batch is smaller).  The evaluator counts packed blocks rather
than multiplying nominal batch count by microbatch size, so its reported token
count is exact.  It is intended to run separately for completed AdamW, Muon,
Muown, and any completed proposal checkpoint after the formal comparisons.

## Initial screen protocol

1. Verify cache digests and remove temporary FineWeb download state after packing.
2. Admit the largest safe microbatch below 90% A100 allocation for every route; use accumulation to equalize effective batch.
3. Run equal-token learning-rate screens on the two idle A100 devices, then start the three matched formal baselines.
4. Record nonfinite/failed candidates as failures; do not replace them with a baseline result.
