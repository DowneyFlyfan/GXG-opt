# Qwen3-0.6B optimizer study — design

## Decision and scope

This is a continued-pretraining optimizer benchmark, not pretraining Qwen3-0.6B from scratch.  The user explicitly selected the 0.6-billion-parameter model; that is an intentional exception to the repository's general 10–200-million-parameter rule.  The study compares tuned AdamW (Adam with decoupled weight decay), Muon (Momentum Orthogonalized by Newton–Schulz), and Muown (Muon directions with Adam-updated row gains), then evaluates the four active research proposals in `records/ideas`.  Proposals in `Done` and `Failed` are excluded.

## Reproducible workload

Use `Qwen/Qwen3-0.6B` in bfloat16, its native tokenizer, and FineWeb-Edu's `sample-10BT` split.  The loader writes only the following recoverable token arrays and manifest below `.cache/qwen3_0p6b_fineweb_edu/`:

| split | distinct Qwen tokens | fixed blocks of 2,048 tokens | storage (`uint32`) |
| --- | ---: | ---: | ---: |
| train | 2,000,000,000 | 976,562 | 7.45 GiB |
| validation | 100,000,000 | 48,828 | 0.37 GiB |

The preparation command streams documents, appends exactly one end-of-sequence token per document, and writes fixed-width blocks.  It records source revision, tokenizer revision, selected shard identifiers, seed, split rule, requested and written token counts, and SHA-256 digests.  The temporary Hugging Face download cache is below this project cache and is removed after manifest verification, so raw source shards do not consume persistent space.  Training data are selected deterministically; validation documents are disjoint before token packing.  The loader drops the final partial block and returns `(input_ids, labels)` where `labels` is the one-token-shifted sequence.

Every candidate starts from the same local model snapshot, sees the same deterministic epoch permutation, uses a sequence length of 2,048, bfloat16 autocast, the same scheduler, and the same effective token batch.  The initial target is three passes over the 2-billion-token train stream (6 billion token exposures, approximately 10 tokens per model parameter).  This is substantial enough to compare optimizer trajectories, but it is not evidence of fresh-pretraining convergence.

## Optimizer routing and tuning

AdamW updates all parameters.  Muon and Muown receive only interior two-dimensional transformer linear weights: attention query, key, value, and output projections plus multilayer-perceptron gate, up, and down projections in blocks 1–26.  Token embedding/tied language-model head, the first and last blocks, and all one-dimensional RMSNorm (root-mean-square normalization) scales remain in AdamW.  The tied embedding is a single physical parameter with one optimizer state.

Before a formal run, each baseline receives an equal-budget short learning-rate screen on an identical held-out prefix.  Muown's direction and gain rates are independently tuned; using a single shared rate is disallowed.  Weight decay is permitted because the requested model exceeds 200 million parameters, but its candidate grid and value are reported.  A memory admission run selects the largest safe microbatch; gradient accumulation produces the same effective batch for every optimizer.  Nonfinite loss, invalid optimizer state, or a failed auxiliary solve stops that candidate and records the failure rather than silently changing its algorithm.

The formal output records validation negative log likelihood and perplexity at matched **optimizer steps**, elapsed time, token exposure, throughput, peak memory, exact hyperparameters, seed, parameter-routing checksum, and checkpoint state.  Final candidates receive both required graphs: `metric_steps.png` and `metric_time.png`, each including the three tuned baselines.  Checkpoints, model snapshots, and temporary download files stay in `.cache`; only compact JSON/JSONL metrics, PNGs, and records are retained outside it.

## Proposal sequence

The active proposals are evaluated independently, never combined:

1. `routing_resistance_v1` — a sampled, positive-semidefinite query/key routing-metric filter.  First verify the categorical sampler, factorized Gram matrix, and Woodbury solve against tiny dense references; then apply the filter to one deterministic query/key head per event.
2. `tied_path_curvature_v1` — a paired-probe generalized Gauss–Newton sketch for the single tied embedding/head parameter.  Test its joint cross-path terms and checkpoint identity before a short screen.
3. `proposal_notch_v1` — the resonance selective filter.  Test its causal proposal history, skip behavior, and disabled equivalence before screening it.
4. `feature_remap_v1` — feature-drift momentum remapping.  Test that remapping is restricted to the selected linear weights and that the disabled setting exactly follows Muon.

For each proposal the order is: mathematical unit tests; 100-step safety screen; matched learning-rate screen; a three-epoch formal candidate only if it passes safety and is competitive on the screen.  A failed safety condition is a measured negative result and is recorded with the exact failure, not converted into a baseline result.

## Execution and safeguards

The preferred execution host is ABA because its two A100 80-GB devices are currently idle.  Code is changed and tested locally first, copied to `/home/yufan/New_Optimizer`, then launched with `nohup`; the remote process, log, checkpoint age, and GPU utilization are monitored.  Before every launch, inspect process ownership and stop only a positively identified filler—not unknown work, experiments, or a serving process.  A tuning run can occupy both devices; formal runs are one candidate per device, sharing no checkpoint path.

The repository is presently dirty and behind its remote.  This study will never reset, pull over, stage broadly, or commit unrelated deletions.  Each commit uses explicit paths; a push is attempted after each completed experiment and any non-fast-forward result is reported without rewriting history.

## Acceptance criteria

- A manifest proves the exact model, tokenizer, data source, split, and fixed-width token counts.
- AdamW, Muon, and Muown each complete equal three-epoch runs with separately tuned Muown rates.
- Every active proposal either has a verified screen and formal comparison or an evidence-backed recorded failure.
- Each final proposal has step- and time-based perplexity PNGs against the three baselines.
- All durable run metadata and results make a resumed process and an independent audit possible.
