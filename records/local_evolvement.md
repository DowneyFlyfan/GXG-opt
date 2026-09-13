# Local evolvement

## Recommendation

Use a **randomly initialized, 63.82M-parameter GPT-style decoder with 12 layers, width 512, eight attention heads, and context length 512**, trained on a frozen sample of **FineWeb-Edu** with the GPT-2 BPE tokenizer. Give each optimizer and seed approximately **one hour on one RTX 5090**. Budget hyperparameter tuning separately. Keep AdamW and Muon as independently tuned baselines.

This is an experiment for developing transformer optimizers: detecting changes in learning speed, validation loss, stability, memory use, and implementation cost. It is not a claim that a one-hour model reaches full pretraining convergence or that its optimizer ranking transfers to large models. The design supports an honest “inconclusive” outcome when differences are below the experiment's resolution.

The configuration is `configs/experiments/local_evolvement.yaml`. It is explicitly an **experiment specification**, not a runnable configuration for the existing five-epoch trainer. A working, bounded calibration utility is included at `scripts/calibrate_local_evolvement.py`. The model has been instantiated and timed on the local RTX 5090; public dataset streaming has been verified. Full dataset preparation, one-hour real-data runs, and empirical confirmation of optimizer separation remain execution steps.

| Component | Selected setting |
| --- | --- |
| Model | Hugging Face `GPT2LMHeadModel`, custom 12×512 configuration, 63,823,360 unique parameters |
| Initialization | Random; every optimizer starts from the same weights within each seed |
| Training data | `HuggingFaceFW/fineweb-edu`, `sample-10BT`, frozen document-level sample |
| Tokenizer | GPT-2 BPE, 50,257 tokens; EOS after each document |
| Local training pool | Capacity for 536,870,912 prediction tokens, about 1 GiB of uint16 storage |
| Holdouts | Separate development and final audit pools, 4,194,304 prediction tokens each |
| Context / effective batch | 512 / 64 sequences = 32,768 prediction tokens per update |
| Physical batch / accumulation | Start at 32 / 2; reduce physical batch while preserving effective batch for larger optimizer states |
| Numerical settings | FP32 parameters and optimizer state; BF16 autocast; SDPA; TF32 enabled |
| Regularization | Dropout 0, weight decay 0, gradient clipping at global norm 1 |
| Primary budget | Freeze training state at 3,540 active seconds; reserve 60 seconds for final evaluation and saving |
| Primary outcome | Final held-out next-token negative log-likelihood, in nats per prediction token |
| Confirmation | Three fresh paired seeds; add two if uncertainty remains |

## Existing repository and compatibility

The inspected checkout contains a custom decoder in `src/models.py` and a byte-tokenized WikiText path in `src/data.py`. The latter caps training at 100M byte tokens. Its 32,000-class vocabulary and byte targets differ from standard GPT-2 BPE. A second profile, `configs/experiments/gpt2_v2_5090.yaml`, specifies random GPT-2 and WikiText-103; its local manifest reports 118,784,000 packed input tokens and five configured epochs. Those are observations from this checkout; the referenced “6G subset” could not be identified as a named, configured training corpus here. Gigabytes of files must not be treated as a token budget.

The new design is separate from both existing profiles. Historical byte-level losses, WikiText losses, and FineWeb-Edu BPE losses measure different tasks. They must have separate plots and result tables. In particular, changing a tokenizer changes the unit of perplexity, so a lower number across tokenizers is not evidence of a better optimizer.

The repository's `AGENTS.md` requires 10M–200M models, Adam(W)/Muon baselines, separate tuning, local execution, gradient accumulation, and metric-versus-step and metric-versus-time PNGs. This design follows those requirements. Its explicit one-hour experiment scope is an exception to the older instruction to run every experiment on the existing model for five epochs. The five-epoch profile itself remains intact.

The 12×512 Hugging Face model retains GPT-2's attention modules, fused QKV matrices, LayerNorm, residual connections, and tied embeddings. These are directly relevant to the six `optimizer_v2` ideas. Switching to a Llama-family model would also change the normalization, positional encoding, and module layout used by several probes. Retaining these structures reduces the number of simultaneous experimental changes.

Compatibility still needs an integration check. The current v2 trainer forces eager attention, assumes WikiText assets, applies a common baseline LR, and has different shifted-token accounting. It cannot implement this specification by accepting the new YAML alone. Baseline calibration uses stock optimizers and does not measure the v2 proposal adapter, its audits, or candidate-specific model passes.

## Research evidence and candidate choices

### Evidence relevant to optimizer discrimination

Semenov, Pagliardini, and Jaggi benchmark tuned optimizers across model sizes, batch sizes, and training horizons. Their 124M experiments include 512-token contexts and demonstrate that short-horizon rankings can change with longer training and different batch sizes. This supports controlling the batch and horizon, and limits what can be concluded from a one-hour proxy. It does not establish an effect size for the proposed 64M model.[^1]

The Polar Express study provides a closer GPT-style reference: GPT-2 Small on 1B FineWeb tokens, with BF16 training and separately selected learning rates. Its reported final validation losses are 4.197 for AdamW, 3.639 for the Jordan Muon variant, and 3.588 for Polar Express. That demonstrates that a web-text transformer task can reveal optimizer differences. Those losses belong to the paper's architecture, dataset, schedule, and hardware, and are not targets for this experiment.[^2]

Modded-NanoGPT's optimizer track is useful for its controlled comparison rules. Its multi-H100 speedrun time is not a runtime estimate for an RTX 5090. Reproducing a highly modified speedrun would also add architectural and implementation choices beyond the optimizer under study.[^3]

Sivaprasad and colleagues show why optimizer comparisons should account for the resources spent finding hyperparameters. Accordingly, the design records tuning cost, grants comparable search budgets, and does not use the same raw LR for all methods as a substitute for tuning.[^4]

### Dataset comparison

| Dataset and access | Strength for this task | Limitation | Decision |
| --- | --- | --- | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), public HF streaming | Natural web documents with educational filtering; sufficient unseen text for the hour; GPT-2-token sample available | A local subset and holdouts must be frozen; filtering changes the target domain | **Primary** |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), public HF streaming | Closest connection to several optimizer papers; broader web domain | A noisier and different domain from Edu; does not automatically yield better statistical power | Strong independent-domain confirmation alternative |
| [WikiText-103 raw](https://huggingface.co/datasets/Salesforce/wikitext), public HF download | Established article corpus with published train/validation/test splits; already cached in this checkout | The existing 118.8M-token cache would be revisited several times at the proposed hourly throughput; narrower article domain | Convenient secondary experiment or offline fallback |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), public HF download | Small-model language learning can be studied quickly; official train/validation data | Synthetic short stories and limited vocabulary make transfer to ordinary web-language training uncertain | Debugging and fast mechanism checks |

FineWeb-Edu's card describes educational filtering and provides a randomly sampled `sample-10BT` configuration. This is a source pool, not the amount to train on. Its published educational-benchmark gains are not evidence that it is inherently the most sensitive optimizer benchmark; selecting it here is a practical research judgment.[^5] FineWeb remains a useful alternative with direct benchmark precedent.[^6]

WikiText contains more than 100M original dataset tokens from selected Wikipedia articles and supplies fixed splits. That token count is not the count obtained with every BPE tokenizer.[^7] TinyStories was explicitly designed to make coherent language generation possible at very small model sizes. Its simpler distribution is useful for rapid checks, but success there alone would be weak evidence for the transformer optimizers' intended web-text setting.[^8]

No Kaggle competition or browser submission is needed for the selected dataset. Direct unauthenticated streaming of a row from the pinned FineWeb-Edu revision succeeded locally. An HF token can affect rate limits, but gated access was false at inspection.

### Model comparison

| Model option | What it would test | Assessment |
| --- | --- | --- |
| Custom GPT-2 12×512, 63.82M | A deep transformer with the current proposals' relevant structures and more tokens per parameter within the hour | **Selected** |
| GPT-2 12×768 with 512 positions, 124.05M | Closer width to the usual GPT-2 Small reference | Feasible, but materially slower in the local calibration |
| An 8-layer, narrower GPT-style model | Faster inexpensive mechanism exploration | Fewer layers or narrower matrices change the geometry being tested; unnecessary as the default after the 64M timing result |
| Random-init SmolLM2-135M configuration | Llama-style architecture with grouped-query attention and RMSNorm | Useful future transfer test; several current GPT-2-specific probes would need adaptation |
| Pretrained model fine-tuning / LoRA | Adaptation of existing weights, or a small trainable subspace | A different optimization problem from full-parameter training from scratch |

The SmolLM2-135M configuration has 30 layers, width 576, nine query heads, three KV heads, RMSNorm, and RoPE. Its name alone therefore does not imply interchangeability with a GPT-2-like 135M model.[^9]

The selected model's parameter count follows from GPT-2's tied output head and biased blocks:

`V*d + T*d + L*(12*d*d + 13*d) + 2*d = 63,823,360`,

where `V=50,257`, `T=512`, `L=12`, and `d=512`. The count was also verified by instantiating the model. It includes trainable embeddings and counts tied weights once. No pretrained weights are loaded. Both initialization standard deviation 0.02 and GPT-2's residual-projection initialization behavior must remain fixed across optimizers.

## RTX 5090 feasibility

NVIDIA specifies 32 GB of GDDR7 for the RTX 5090.[^10] The local CUDA device reported 32,607 MiB total through `nvidia-smi`, with approximately 4.5 GB already in use at the initial inspection. Available training memory therefore must be measured, not inferred from the advertised capacity.

The calibration used the installed **PyTorch 2.11.0+cu130 and Transformers 5.7.0**, FP32 model parameters, BF16 autocast, clipping, and real optimizer updates. It used random token IDs with dense LM forward/backward computation and pinned CPU-to-GPU batch transfer. It ran five warmup updates and timed 25 synchronized updates per configuration. Evaluation batch timing was measured separately. It used no `torch.compile` and no activation checkpointing.

| Model / attention / optimizer | Micro × accumulation | Mean update, s | Prediction tokens/s | Peak allocated / reserved, GiB |
| --- | ---: | ---: | ---: | ---: |
| 63.82M / SDPA / AdamW | 8 × 4 | 0.1240 | 132,157 | 5.40 / 5.52 |
| 63.82M / SDPA / AdamW | 32 × 2 | 0.2579 | 127,047 | 18.23 / 20.74 |
| 63.82M / SDPA / Muon + AdamW | 32 × 2 | 0.2726 | 120,226 | 18.10 / 20.73 |
| 124.05M / SDPA / AdamW | 16 × 4 | 0.3985 | 82,223 | 12.67 / 12.93 |
| 63.82M / eager / AdamW | 32 × 2 | 0.3339 | 98,143 | 22.73 / 25.38 |

The first row has a smaller **effective** batch and is a sizing observation, not a scientifically comparable optimization run. The other rows use 32,768 prediction tokens per update. The 124M row is not an exhaustive batch search for that model. Peak memory includes model/gradient/optimizer allocations in the PyTorch process; unrelated desktop use is additional.

On the tested path, the 64M model gives about 1.55 times the AdamW throughput of the tested 124M configuration. Muon adds about 5.7% update time relative to the matched 64M AdamW row. Eager attention adds about 29.5% update time. These are local short-probe observations, not universal GPU speed ratios.

At the measured mean speeds, **12,288 updates = 402,653,184 prediction tokens** take about **52.8 minutes for AdamW and 55.8 minutes for Muon**, before evaluation, checkpointing, initialization, and other runner costs. Allowing roughly 3,300 seconds for updates and using the observed 90th-percentile step times yields about 414M and 394M tokens, respectively. Thus approximately **390M–440M tokens per hour** is a defensible initial planning band for these baselines, subject to real-data calibration.

The short probe does not establish sustained clocks, thermal behavior, data-loader throughput, candidate-probe cost, or actual learning quality. Tokens were repeated within the shape probe; its loss values are intentionally not reported as quality evidence. Before locking a run contract, repeat timing for at least 200 steady updates on the frozen real dataset and exercise every candidate's expensive periodic work. A 200-step probe is a minimum: increase it if needed to cover several complete probe/refresh cycles.

### Memory policy

Start at physical batch 32 and accumulation 2. It already uses about 21 GiB reserved, leaving space for the desktop and additional optimizer state. Target no more than about 24 GiB process reservation while maintaining at least 3 GiB global headroom. Measure peaks after states have been allocated and after the most expensive curvature/probe update. A baseline-only forward pass is not an adequate memory check.

If a candidate needs more space, try physical batch 16 with accumulation 4, then 8 with accumulation 8. Preserve 64 sequences and 32,768 prediction tokens per update. For a strict algorithm comparison, use the largest physical batch that every included method can safely support, and rerun the baselines at that batch. A deployment-oriented time comparison can use method-specific physical batches, but must label and record that implementation difference.

Do not fill unused VRAM by increasing the effective batch after tuning has started. Batch size changes gradient noise and may change optimizer rankings. Additional reserve is justified for curvature optimizers; the largest safe allocation is not necessarily the fastest configuration.

## Dataset preparation and access

Pin the following source revisions before preparing data:

| Asset | Revision / access |
| --- | --- |
| FineWeb-Edu | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| GPT-2 tokenizer | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Source configuration | `sample-10BT`, upstream split `train` |
| Access page | [FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| Source files | [Pinned 10BT sample directory](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT) |

The inspected 10BT directory contains 14 Parquet shards totaling **28,518,193,415 bytes**. Downloading the whole source sample would defeat the small local-cache objective. Use streaming during preparation and stop once the local quotas are satisfied. Hugging Face supports iteration and buffered shuffling without first downloading a complete dataset; buffered shuffling should not be described as a uniform shuffle of the full corpus.[^11]

The following is a verified access recipe, **not the complete preparation pipeline**:

```python
from datasets import load_dataset

documents = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    "sample-10BT",
    revision="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    split="train",
    streaming=True,
)
```

The repository's `.venv-gpt2-v2/bin/python` has the `datasets` dependency used for the verified read. The base `/home/justin/miniconda3/bin/python` used for GPU calibration does not have that package. Avoid changing the installed CUDA stack just to obtain dataset utilities.

### Frozen sample contract

1. Traverse pinned shards in recorded source order. Record any preprocessing shuffle seed and buffer size. Apply stable NFKC and whitespace normalization for **deduplication keys**; tokenize the original document text so normalization does not silently change the modeled corpus.
2. Remove exact duplicate normalized-text hashes globally before admitting documents to any split. For retained documents, use a canonical URL as the split key when available, otherwise the normalized-text hash. Canonicalize URLs deterministically: lowercase scheme and host, remove the fragment, retain path and query. Preserve its implementation/version in the manifest.
3. Assign each retained document using `int(sha256(("local_evolvement_v1:" + key).encode()).hexdigest(), 16) % 1000`. Buckets 0–979 are training, 980–989 development, and 990–999 audit. Split **before tokenization/packing**. All retained versions of the same canonical URL stay in one split. Global exact deduplication prevents identical normalized content at different URLs from appearing in different splits.
4. Tokenize without automatic special tokens and append EOS 50256. Write a local training stream with one continuous buffer, carrying token remainders across source batches. Do not discard a remainder at every map batch or worker boundary.
5. Freeze capacity for 536,870,912 training targets. A stream producing that many targets needs 536,870,913 stored IDs. Stop appending to each split at its token quota; continue traversing only as needed to fill the other quotas. Record final-document truncation and unused tail tokens explicitly.
6. Freeze 4,194,304 scored targets for development and another 4,194,304 for audit. Keep document identities and window offsets. Select the quick-development set once from the development pool using a recorded deterministic order, with exactly 262,144 scored targets. Exclude its selection from candidate-dependent decisions.
7. Record source IDs, URL-key hashes, normalized text hashes, tokenizer files and hashes, token-file SHA-256s, counts, exact split-disjointness checks, and preprocessing revision. Reject empty or insufficient splits and never silently reuse development data as training data.

This procedure prevents exact-duplicate and same-canonical-URL leakage under the stated normalization. It does not prove absence of near duplicates, syndicated paraphrases, or all forms of domain overlap. Inspect cross-split near-duplicate samples before making strong quality claims. The experimental population is the frozen local sample; it is not asserted to be a uniform sample of all 1.3T source tokens.

The train token file occupies about 1 GiB; the two main held-out token pools add about 16 MiB before document indexes and padding. Source Parquet ranges, raw caches, manifests, and checkpoints require additional disk. Network bytes depend on Parquet row groups, filesystem caching, and streaming behavior and are not guaranteed to equal the final token-cache size. Prepare once on the WSL filesystem under `.cache/local_evolvement/`, then run entirely from local files.

For each run seed, generate one random permutation of the frozen stride-512 training windows using a dedicated CPU generator seeded with `run_seed + 1,000,003`. Record the permutation hash and consume it without replacement. If calibration suggests the pool will be exhausted before the time limit, enlarge and refreeze it for all methods before starting the campaign. An unexpected exhaustion is an incomplete run; do not silently cycle a smaller corpus to fill the hour.

### Loss and window definition

For training window `k`, take 513 consecutive IDs starting at `512*k`. Use the first 512 as model input and the last 512 as targets. Each target position is scored once. EOS separates packed documents; causal attention across a training document boundary is allowed and identical for every optimizer. The next window overlaps by one stored token to provide its first prediction context.

For development and audit, reset context per document. Split each document into input/target windows with the same one-token overlap. Pad only the final partial window and mask its padding targets. Retain EOS as a scored target. This gives a reproducible document-level evaluation and supports resampling entire documents. Training loss and held-out loss need not match because their boundary contexts differ, but all optimizers use the same definitions.

Compute `NLL = sum(valid target cross-entropies) / number of valid targets`, with FP32 cross-entropy accumulation and an FP64 dataset sum. Perplexity is `exp(NLL)`. Do not average batch means with unequal token counts. Auxiliary curvature/probe tokens are charged to cost, but do not inflate the count of primary training targets.

## One-hour comparison and equal-token comparison

It is impossible to guarantee all three of the following for arbitrary optimizers: exactly equal training tokens, exactly equal wall time, and unrestricted optimizer overhead. The design makes the choice explicit through two measurements.

### Primary: fixed wall clock

The primary question is: **Which optimizer produces the best held-out loss in roughly one hour on this GPU?** Start the active clock immediately before per-run model/optimizer initialization. Include initial evaluation, batch loading and transfers, forward/backward passes, updates, proposal calculations, periodic probes, validation, diagnostics, and checkpoint I/O. Dataset preparation and the separate tuning campaign are excluded and reported separately.

Freeze the final training state no later than 3,540 active seconds. Before launching another update, allow enough time for a conservative estimate of its complete cost, including scheduled expensive work. Never score a half-applied update or a partially accumulated gradient as a completed optimizer step. Use the remaining 60 seconds for full development evaluation and final saving. If this reserve is insufficient in calibration, enlarge it for **all** methods before the comparison; record any actual overrun.

Use a linear warmup for 200 completed updates, then a constant peak LR. At active time 2,655 seconds, begin a linear decay toward 1% of the peak at 3,540 seconds. Define the LR for an update from its start time; use the completed step index for warmup. Apply the same multiplier to the independently configured matrix and auxiliary LRs. A candidate too slow to finish warmup before decay is a throughput failure for this profile and needs a separately labeled smaller-scale experiment.

Evaluate the 262,144-target development sentinel every 300 active seconds and at every 1,024-update landmark during the constant-LR phase. Coalesce coincident evaluations. Full development evaluation is required at initialization and at the frozen endpoint. Charge evaluation time to the active budget. Once the run's settings are fixed, avoid adding method-specific diagnostic work that changes its available training time without reporting it.

At common update landmarks reached **before time-based decay by every method**, the curves also provide a controlled comparison at equal tokens. The same seed, initialization, batch sequence, clipping, warmup, and constant LR schedule apply; each optimizer still uses its separately tuned LR. Use actual recorded landmarks and phase labels, not interpolation that implies an unobserved checkpoint was evaluated. These prefix comparisons are exploratory and use the quick-development set.

### Secondary: fixed tokens for finalists

For a cleaner endpoint comparison of learning efficiency, rerun selected methods for **12,288 updates / 402,653,184 targets**, using exactly the same training order within each seed. Warm up for 200 updates, remain constant through update 9,216, and linearly decay to 1% by update 12,288. Evaluate the same development and final audit populations.

The baseline timing suggests a runtime close to an hour, but a costly curvature method can take longer. That is a reported result, not a reason to remove its overhead from the clock. If strict hourly duration is required even for this secondary experiment, calibrate and freeze a lower common step count across all finalists before tuning. Faster methods will finish earlier. Recalibrate whenever the architecture, attention implementation, effective batch, proposal frequency, or optimizer implementation changes.

Do not interpret an equal-token endpoint obtained after time-dependent cooldown as a fully controlled equal-token comparison: the schedule phases may differ. This is why the primary run's common-prefix analysis is restricted to the shared constant phase and the secondary mode has an explicitly step-based schedule.

## Optimizers and tuning

### Baseline definitions

**AdamW:** all trainable parameters, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay zero, fused implementation when supported consistently. At zero decay, this is the Adam update implemented through AdamW. Zero decay follows the repository's small-model preference and avoids making an untuned decay coefficient part of the initial comparison. A future regularization study should treat nonzero decay as a distinct, equally tuned protocol.

**Muon:** full two-dimensional hidden attention/MLP matrices only; exclude token embeddings, positional embeddings, tied output embeddings, LayerNorm parameters, and biases. Optimize excluded parameters with AdamW. Use momentum 0.95, Nesterov momentum, five Newton–Schulz iterations, coefficients `(3.4445, -4.775, 2.0315)`, and epsilon `1e-7`.

Pin `adjust_lr_fn="match_rms_adamw"`, whose shape factor is `0.2*sqrt(max(rows, columns))`. PyTorch also offers an `original` adjustment with a different formula; numeric LR values are not portable between the two. GPT-2's Conv1D matrix orientation and its full fused-QKV partition must be recorded. The selected symmetric factor matches the existing repository's shape-scaling convention, but does not make all optimizer implementations or momentum conventions identical.[^12]

For the six v2 ideas, identify the base optimizer family and preserve a separately tuned baseline without the intervention. Test each proposal in isolation before combining proposals. Ensure an embedding shared with the output head is updated only once. Count probe forward/backward calls, precision conversions, matrix solves, and rejected or retried proposals in both diagnostics and elapsed time.

### Search budget

Use these starting search ranges, not claims of optimal settings:

| Method | Peak learning-rate trials |
| --- | --- |
| AdamW | `1e-4`, `3e-4`, `6e-4`, `1e-3`, `2e-3` |
| Muon matrix LR with the pinned adjustment | `3e-4`, `1e-3`, `3e-3`, `1e-2`, `3e-2` |
| Muon auxiliary AdamW | Start with the independently selected AdamW LR |

Hold the listed betas, momentum, epsilon, clipping, and schedule constant during the initial LR sweep. This is a budget-limited baseline search, not exhaustive tuning. Candidate-specific parameters can consume the same allowed trial budget; report the actual search spaces rather than concealing manual iterations.

Run five 10-minute screens per method on seed 42. These screens use the opening portion of the full hourly schedule; they must not compress an hour's cooldown into ten minutes. Extend the two best settings to full-hour trials from identical initial conditions, then select by full development NLL. Total initial tuning allocation is **170 minutes per method**. Early screens can misrank slow-starting methods, so retain all curves and keep a documented, equal-budget extension route.

If the winning LR is at a boundary, extend the grid or explicitly label the baseline undertuned. Do not publish an improvement over an obviously undertuned baseline. If tuning auxiliary LR or a second momentum parameter, grant the same additional tuning budget to the competing methods and record its use. Keep all tuning logs, configurations, failures, and time costs; do not count seed 42 as a fresh confirmation seed.

## Detecting effects and interpreting uncertainty

Use one screened seed to discard clear divergence, excessive overhead, or no measurable benefit. For promising candidates and the tuned baselines, run fresh seeds **101, 202, and 303**. Within each seed, share a hashed initialization, the exact frozen data order, effective batch, and evaluation windows. Use distinct, recorded random generators for model initialization, data ordering, and candidate probes so consuming probe randomness cannot change the training batches.

Order runs in a balanced or randomized sequence across seeds so one method does not consistently run on a colder GPU or during a busier desktop period. Record GPU driver, runtime versions, clock/power settings, peak memory, and background contention. Pairing reduces nuisance variation; it does not guarantee bitwise reproducibility of all BF16 kernels. Do not force a slow attention implementation for only one method and call the resulting runtime an optimizer-only comparison.

Define the effect for seed `s` as `delta_s = NLL(candidate, s) - NLL(baseline, s)`. Negative is better. Report each paired delta, the mean, the standard deviation of paired deltas, and a 95% paired Student-t interval. With three seeds the interval is wide: its half-width is `4.303 * sd(delta) / sqrt(3)`. For example, a paired standard deviation of 0.01 nats gives a half-width of about 0.025 nats; three runs would not resolve a 0.01-nat gain reliably. Add seeds 404 and 505 when useful, and report inconclusive results if uncertainty remains.

Predeclare **0.01 nats per token** as the initial smallest practically interesting difference. It corresponds to a perplexity reduction of approximately `1 - exp(-0.01) = 0.995%`. This is a decision threshold, not a promised detection limit or a threshold derived from the public papers. A negative interval with a tiny effect is statistically different but may not be practically useful. A strong improvement claim requires the uncertainty bound to support the chosen practical threshold, not just a favorable mean.

Separately, use a paired document-cluster bootstrap of the held-out documents: resample document identities jointly across the compared models, then recompute token-weighted NLL using each resample's summed loss and token count. Use 10,000 resamples with a fixed bootstrap seed. This measures uncertainty due to the finite evaluation corpus conditional on the trained models. It must not be presented as uncertainty over training seeds, and millions of token positions must not be treated as millions of independent training replicates.

Keep the audit holdout unused until architecture, hyperparameters, finalists, and comparison endpoints are frozen. Then score final checkpoints on it. Extensive repeated development against the same validation data can overfit the benchmark; reserve a new document slice for a later campaign or confirm on FineWeb/WikiText before claiming broad utility. For multiple confirmatory candidates, predeclare a primary comparator or apply a multiple-comparison correction such as Holm to the family of superiority tests. Exploratory plots alone are not confirmatory tests.

### Required outcomes

| Observation | Interpretation |
| --- | --- |
| Lower loss at equal tokens and at one hour, replicated on fresh seeds | Evidence of learning-efficiency and practical runtime improvement on this profile |
| Lower loss at equal tokens, worse loss at one hour | Better token efficiency whose compute overhead is too high locally |
| Better one-hour result but no equal-token gain | Useful runtime improvement; separate implementation throughput from the optimizer's learning behavior |
| Better training loss but worse held-out loss | Faster fitting or a generalization change, not a held-out improvement |
| Overlapping paired uncertainty or an effect below the resolution | Inconclusive or practically negligible; preserve the result |
| Repeated OOM, nonfinite losses, or incomplete updates | Feasibility/stability failure for this configuration |

Before calling the benchmark sensitive, complete tuned AdamW/Muon hourly pilots, verify that both learn well beyond initialization, and inspect whether baseline and candidate differences exceed seed and evaluation noise. The two baselines need not have a statistically significant gap: a well-tuned tie is a valid outcome. What matters is the measured resolution and whether candidate effects exceed it. If resolution is inadequate, add seeds, enlarge the evaluation set when document noise dominates, or test a second domain/horizon. Do not select a dataset merely because it makes a favorite optimizer win.

## Reproducibility, artifacts, and execution sequence

Every completed experiment should retain resolved configuration, code revisions and hashes, model/optimizer routing, dataset manifest, initialization hash, RNG states, resume metadata, token and step counters, per-step or interval timings, validation NLL, and optimizer diagnostics. Save both **metric_steps.png** and **metric_time.png**, including tuned AdamW and Muon from the same protocol. Show NLL as the primary metric and optionally perplexity; use completed optimizer updates rather than epochs on the step axis.

In `summary.csv`, include method, seed, NLL/PPL, completed updates, primary training targets, auxiliary targets and forward/backward calls, active wall seconds, training-only seconds, evaluation/save time, tokens per second, peak allocated/reserved VRAM, LR settings, failure status, and experiment mode. Keep equal-time and equal-token endpoints identifiable. Save per-document loss sums and token counts for audit resampling.

Keep results under `results/local_evolvement/` and checkpoints under `.cache/local_evolvement/checkpoints/`. Retain the final model and a latest resumable checkpoint, with periodic atomic replacement rather than an unlimited copy at every metric point. Include the model, optimizer, schedule state, data cursor/order, RNG states, and active elapsed time in resumable state. Do not reset an hour's budget when resuming. Mark interrupted runs and record restart overhead.

The following sequence makes the design executable without altering the older benchmark:

1. Prepare and verify the pinned local sample and tokenizer with the split/window contract above. Dataset access has already been checked; the full freeze has not been performed.
2. Implement a separate training entry point for this YAML, with explicit optimizer/auxiliary LR groups, the two budget modes, document-aware evaluation, and the required output accounting. Reuse compatible model/probe code after checking assumptions; do not silently route this design through the five-epoch runner.
3. Check token accounting, loss masking, parameter routing, identical initialization, and batch order with tiny cases. Verify each candidate's disabled intervention matches its base optimizer to the expected numerical tolerance.
4. Run real-data calibration for baselines and selected candidates, covering memory peaks, full evaluation, checkpoint I/O, and multiple proposal-refresh cycles. Freeze the safe shared physical batch, timing reserve, and runtime versions.
5. Execute the equal-budget tuning campaign, then the one-hour screen and fresh-seed confirmation. Save the required PNGs for completed optimizer results. Evaluate the unused audit split only after freezing finalists.
6. Run the fixed-token finalist comparison and a second domain or larger architecture when a broader optimizer claim is intended.

To rerun the included **calibration utility** at the proposed batch:

```bash
cd /home/justin/GXG-opt
/home/justin/miniconda3/bin/python scripts/calibrate_local_evolvement.py \
  --micro-batch 32 --effective-batch 64 --method adamw \
  --output .cache/local_evolvement/calibration/adamw-new.json
/home/justin/miniconda3/bin/python scripts/calibrate_local_evolvement.py \
  --micro-batch 32 --effective-batch 64 --method muon \
  --output .cache/local_evolvement/calibration/muon-new.json
```

These commands run short synthetic shape probes; they do not start the hour-long experiment. Use new output names, because the utility refuses to overwrite measurements. Raw measurements for this design are in `records/local_evolvement/calibration/`. The baseline timings make the proposed model/data combination plausible on this exact GPU. The experimental controls make its effects interpretable. The eventual size and significance of those effects require the specified real-data runs.

## Sources

Public sources were accessed during the local design work on September 11, 2026, Pacific time. Local evidence consists of the inspected checkout, `.cache/gpt2-v2/manifest.json`, hardware/runtime queries, the successful streaming read, and the raw calibration JSONs accompanying this report.

[^1]: Andrei Semenov, Matteo Pagliardini, and Martin Jaggi. *Benchmarking Optimizers for Large Language Model Pretraining*, September 2025, version 1. [Paper, especially sections 3–4 and appendices D–E](https://arxiv.org/html/2509.01440v1). [Authors' implementation](https://github.com/epfml/llm-optimizer-benchmark). Used for the dependence of conclusions on training horizon, effective batch, and tuning; not for RTX 5090 throughput.

[^2]: Noah Amsel and colleagues. *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm*, 2025, version 2. [Section 5.2 and Figure 6](https://arxiv.org/html/2505.16932v2). Used for the 124M GPT-2/FineWeb optimizer comparison and its experimental conditions.

[^3]: Keller Jordan and contributors. *Modded-NanoGPT*, evolving repository. [Project and hardware scope](https://github.com/KellerJordan/modded-nanogpt); [optimizer-only track](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization). Used as a controlled-comparison reference, not a consumer-GPU runtime claim.

[^4]: Prabhu Teja Sivaprasad, Florian Mai, Thijs Vogels, Martin Jaggi, and François Fleuret. *Optimizer Benchmarking Needs to Account for Hyperparameter Tuning*, ICML 2020. [Paper, revised August 2020](https://arxiv.org/abs/1910.11758). Used for the need to include and disclose hyperparameter-search resources.

[^5]: Hugging Face FineWeb team. *FineWeb-Edu dataset card*, evolving release. [Dataset description, sample configurations, streaming, and limitations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Used for the source corpus, educational filtering, and 10BT sampling configuration. Revision and file sizes were independently checked through the HF API.

[^6]: Hugging Face FineWeb team. *FineWeb dataset card*, evolving release. [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb). Used for the broader-web alternative. The team's [technical report](https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/index.html) describes FineWeb/FineWeb-Edu and their release terms; source material remains subject to its applicable rights.

[^7]: Salesforce Research and dataset maintainers. *WikiText dataset card*, based on the 2016 dataset. [Dataset and split definitions](https://huggingface.co/datasets/Salesforce/wikitext). Used for the article corpus, original token-scale description, fixed splits, and raw configuration.

[^8]: Ronen Eldan and Yuanzhi Li. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, 2023. [Paper](https://arxiv.org/abs/2305.07759); [authors' dataset](https://huggingface.co/datasets/roneneldan/TinyStories). Used for the synthetic short-story corpus and its small-model purpose.

[^9]: HuggingFaceTB. *SmolLM2-135M model configuration*, repository artifact. [Configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json). Used for architecture comparison, not measured local performance.

[^10]: NVIDIA. *GeForce RTX 5090 specifications*, product documentation. [Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/). Used for advertised GPU memory; all training timings are local measurements.

[^11]: Hugging Face. *Datasets: Stream*, current documentation. [Streaming and buffered shuffling](https://huggingface.co/docs/datasets/stream). Used for bounded-access mechanics; it does not guarantee a particular network-byte total.

[^12]: PyTorch. *Muon*, version 2.11 documentation. [Constructor, parameter routing guidance, and LR adjustments](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Muon.html). The installed `torch.optim._muon` source was also inspected and hashed by the calibration utility.
