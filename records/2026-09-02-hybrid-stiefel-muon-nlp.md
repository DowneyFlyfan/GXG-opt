# Hybrid Muon / Stiefel-Muon on GPT-12x512

## Question

Test the hypothesis that the first and last Transformer blocks should retain
ordinary Muon, while the ten interior blocks use the closed-form manifold
updates: the square orthogonal formula from
[article 11215](https://kexue.fm/archives/11215) and the rectangular Stiefel
formula from [article 11864](https://kexue.fm/archives/11864).

## Routing

The retained GPT-12x512 has four Muon-eligible matrices per block: QKV,
projection, feed-forward expansion, and feed-forward contraction.  The hybrid
route is therefore:

| Location | Matrix count | Optimizer |
|---|---:|---|
| `blocks.0` | 4 | ordinary Muon |
| `blocks.1` through `blocks.10` | 40 | scaled Stiefel-Muon |
| `blocks.11` | 4 | ordinary Muon |
| embeddings/tied output, normalization, biases | remaining | AdamW auxiliary |

For each interior matrix the physical tensor is \(W=sQ\), where
\(s=\lVert W_0\rVert_F/\sqrt d\) is fixed and \(Q\) remains on the appropriate
column- or row-Stiefel manifold.  Square matrices use the blog's Muon plus
orthogonal-parameter update; QKV and feed-forward rectangles use the Stiefel
closed form.  Edge blocks are neither projected at initialization nor
retracted during training.

## Tuning and resources

Local GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB.  Physical batch 12 plus
gradient accumulation 4 was stable, yielding effective batch 48 and 12,298 MiB
peak allocated memory.

| Sweep | Candidate | Validation accuracy |
|---|---|---:|
| Interior Stiefel learning rate, 12 steps | 0.001 | 0.229335 |
|  | 0.003 | **0.280928** |
|  | 0.006 | 0.279188 |
|  | 0.010 | 0.279526 |
| Edge Muon learning rate, 12 steps with interior 0.003 | 0.0001 | 0.280819 |
|  | 0.0003 | 0.280928 |
|  | 0.0006 | **0.281199** |
|  | 0.0010 | 0.279986 |

At 128 steps, edge Muon 0.0006 and interior Stiefel 0.003 reached 0.292868,
slightly above edge Muon 0.0003 / interior 0.003 at 0.292847.  The formal run
used the former, momentum 0.95, five Newton--Schulz iterations, seed 1337, and
the same cached WikiText-103/data order/five-epoch target as all retained
baselines.

## Final result

| Optimizer | Final validation next-token accuracy | Time |
|---|---:|---:|
| AdamW (retained) | 0.711754 | 2.787 h |
| all-Stiefel-Muon | 0.722632 | 3.825 h |
| hybrid Muon/Stiefel-Muon | 0.726454 | 3.663 h |
| ordinary Muon (retained) | **0.752193** | **2.894 h** |

The hybrid improves the all-Stiefel result by 0.003822 and AdamW by 0.014701,
so the edge-Muon hypothesis is directionally supported.  It remains 0.025739
below ordinary Muon and 26.6% slower, so it does **not** establish a replacement
for original Muon.

The final checkpoint has 40 constrained interior matrices.  Their normalized
Stiefel Gram-error range is \(3.25\times10^{-5}\) to
\(3.34\times10^{-3}\) in Frobenius norm, the residual of finite Newton--Schulz
polar retraction rather than unconstrained parameter drift.

## Interpretation

Leaving the edge blocks unconstrained materially helps relative to constraining
every block, but it does not remove the central limitation: hard Stiefel
optimization prevents the interior blocks from learning their singular values.
The comparison with historical Muon also remains initialization-confounded in
the interior: hybrid Stiefel-Muon projects those 40 matrices at initialization,
while ordinary Muon begins from Gaussian initialization.  A strict causal
ablation would initialize ordinary Muon with the same middle-block scaled
Stiefel matrices.

## Artifacts

- Metrics: `metrics/nlp/nlp_gpt_12x512__hybrid_stiefel_muon_b12_a4_mu0006_st003_formal.jsonl`
- Result: `results/nlp/nlp_gpt_12x512__hybrid_stiefel_muon_b12_a4_mu0006_st003_formal.json`
- Metric--Steps: `results/nlp/hybrid_stiefel_muon_b12_a4_mu0006_st003_formal_metric_steps.png`
- Metric--Time: `results/nlp/hybrid_stiefel_muon_b12_a4_mu0006_st003_formal_metric_time.png`
- Checkpoint: `.cache/nlp/checkpoints/nlp_gpt_12x512__hybrid_stiefel_muon_b12_a4_mu0006_st003_formal.checkpoint.pt`
