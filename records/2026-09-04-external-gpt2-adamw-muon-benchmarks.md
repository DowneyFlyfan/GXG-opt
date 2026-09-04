# External GPT-2 AdamW--Muon benchmarks

## Decision

Our current GPT baseline is 54.68M parameters on byte-level WikiText-103, so
external GPT-2/FineWeb losses must not be merged into our result graphs. They
are nevertheless strong evidence that the current Muon matrix learning rate
`6e-4` is undertuned: controlled GPT-2 papers select `0.005` to `0.02` for
Muon while selecting `0.0005` to `0.0006` for AdamW.

## Closest controlled AdamW--Muon result

[Amsel et al., *The Polar Express* (arXiv:2505.16932; ICLR
2026)](https://arxiv.org/abs/2505.16932) reports the closest direct comparison.

| Setting | Value |
| --- | --- |
| Model | GPT-2 Small: 124M, 12 layers, 12 heads, width 768 |
| Data / budget | FineWeb, one epoch, 1B tokens |
| Tokenizer / context | GPT-2 vocabulary 50,257; context 1,024 |
| Batch / precision / hardware | Batch 32, bfloat16, four H100 GPUs |
| Schedule | Constant for first 40% of updates, then linear decay |
| Weight decay | None |
| Muon parameter routing | 2D matrices only; embeddings, unembedding, and positional embeddings excluded |
| Polar approximation | Five bfloat16 iterations |

| Optimizer | Tuned learning rate | Final validation loss |
| --- | ---: | ---: |
| AdamW | `0.0005` | 4.197 |
| Original Muon (Jordan Newton--Schulz) | `0.01` | 3.639 |
| Muon with Polar Express | `0.005` | 3.588 |

Thus the standard Muon arm has a raw matrix learning rate 20 times AdamW's
rate and lowers the paper's validation loss by 0.558. Figure 4 and its exact
experimental description are in the [OpenReview paper](https://openreview.net/forum?id=yRtgZ1K8hO).

## Independent check: MARS

[Yuan et al., *MARS*](https://proceedings.mlr.press/v267/yuan25f.html) use
GPT-2 Small (125M) on FineWeb-Edu. Its Appendix F selects AdamW maximum
learning rate `6e-4`, betas `(0.9, 0.95)`, but Muon maximum learning rate
`2e-2` and momentum `0.95`.

| Training tokens | AdamW best validation loss | Muon best validation loss |
| ---: | ---: | ---: |
| 5B | 3.193 | 3.165 |
| 20B | 3.024 | 3.006 |
| 50B | 2.885 | 2.901 |

Muon wins at 5B and 20B, but loses at the 50B endpoint. This is direct
counter-evidence to an unconditional “Muon always wins” assumption. The study
uses 16 A100 GPUs for GPT-2 Small and is not a four-hour local target.

## Fully reproducible optimizer-only benchmark

[Modded-NanoGPT Track 3](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization)
fixes the GPT-124M architecture, FineWeb data, 0.5M-token batch, validation,
and one forward/backward pass per step. Its target is validation cross-entropy
3.28.

| Arm | Matrix learning rate | Steps to target | Qualification |
| --- | ---: | ---: | --- |
| Muon plus auxiliary AdamW | `0.02` | 3,600 | One seed; non-significant |
| Adam | `0.0015` | 5,625 | Authors label it likely undertuned |
| Muon plus auxiliary AdamW | `0.025` | 3,375 | 20 seeds; accepted |

The published baseline consumes about 2B tokens, so it is not credibly a
four-hour task on the local 16GB RTX 5070 Ti. It remains the best public code
reference for parameter groups and data/evaluation invariance.

## Local experiment implied by the evidence

The paper's 1B-token protocol is too large for a four-hour claim here. A
100M-token proxy is appropriate: retain GPT-2 Small, FineWeb, context 1,024,
the held-out split, bfloat16, parameter routing, and schedule shape; reduce
only the training budget. At global batch `32 x 1,024`, that is about 3,052
updates. It is a tuning proxy, not a published convergence result.

\[
\eta_{\mathrm{Muon}} \in \{0.0025, 0.005, 0.01, 0.02\},
\qquad
\eta_{\mathrm{AdamW}} \in \{0.0003, 0.0005, 0.0006, 0.001\}.
\]

The local Muon code already uses the conventional shape multiplier
\(0.2\sqrt{\max(m,n)}\), so this is a true pre-scaling matrix learning rate,
not an AdamW elementwise rate. Auxiliary parameters must remain on AdamW.

## Sources

1. [Polar Express paper](https://arxiv.org/abs/2505.16932).
2. [Muon official implementation and parameter-group recipe](https://github.com/KellerJordan/Muon).
3. [Modded-NanoGPT Track 3 rules, code, and logs](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization).
4. [MARS paper PDF, Figure 7 and Appendix F](https://raw.githubusercontent.com/mlresearch/v267/main/assets/yuan25f/yuan25f.pdf).
5. [SkyAI GPT-2 124M Muon recipe and caveats](https://github.com/northaxosky/sky-ai).
