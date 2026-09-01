# Paper Full Gauss-Newton reproduction on GPT-12x512

## Scope

This experiment reproduces the algorithm from *The Potential of Second-Order
Optimization for LLMs: A Study with Full Gauss-Newton* on the retained local
NLP baseline. It does not claim to reproduce the paper's C4, LLaMA, H100, or
120M--240M-token large-batch results. The controlled local task is the 54.68M
parameter GPT-12x512 model, cached WikiText-103 byte stream, sequence length
1024, fixed validation set, and validation next-token accuracy metric.

The source audit used the original PDF and the official code cached at
`.cache/reference/full-gauss-newton` (commit
`0c14e7ec429646f5693dc8cb02fe81e1f01075c2`).

## Reproduced algorithmic contract

- The inner objective gradient is evaluated matrix-free as
  \(g_0 + J_0^T H_L J_0(\theta-\theta_0)\). The Generalized Gauss--Newton
  matrix is never materialized.
- Muon optimizes eligible hidden matrices. AdamW handles embeddings, tied
  output head, normalization parameters, biases, and other non-matrix
  parameters.
- The inner parameters and optimizer state persist across outer updates. The
  next solve starts at the previous pre-line-search endpoint.
- Each outer update uses a fresh held-out batch to compare step sizes
  \(2^{-i/2}\), \(i\in\{0,1,2,3,4\}\), on the true nonlinear model.
- The 45M paper template is reproduced with 122 inner steps, inner statistical
  batch 32, and sequence length 1024: 3904 sequences or 3,997,696 tokens per
  outer direction. One physical sequence is used at a time; both the gradient
  and \(J^T H_L J\) curvature contribution are averaged across all 32
  microbatches.
- The initial inner hyperparameters are Muon momentum 0.95, global gradient
  clipping 1, Adam-routed weight decay 0.001, no Muon-matrix weight decay, and
  an inner cosine cycle. Learning rate is selected by local qualification.

## Common warmup

All paper Full Gauss--Newton trials start from one AdamW checkpoint after 5%
of the Chinchilla-optimal token count, matching the paper's initialization
control.

| Quantity | Value |
|---|---:|
| Target tokens | 54,682,624 |
| Processed tokens | 54,685,696 |
| AdamW updates | 6,676 |
| Wall time | 1,106.620 s |
| Validation next-token accuracy | 0.656700 |
| Checkpoint | `.cache/nlp/checkpoints/nlp_gpt_12x512__paper_common_adamw_warmup.pt` |

## Qualification results

| Label | Inner steps | Inner batch | Outer tokens | Inner LR | Outer steps | Final accuracy | Selected line-search step | Post-warmup time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `paper_full_gn_qual_n60_lr001` | 60 | 1 | 61,440 | 0.010 | 1 | 0.653870 | 0.2500 | 16.31 s |
| `paper_full_gn_pilot_n60_lr0001` | 60 | 1 | 61,440 | 0.001 | 10 | 0.679794 | 0.2500 | 120.35 s |
| `paper_full_gn_pilot_n122_lr0001` | 122 | 1 | 124,928 | 0.001 | 5 | 0.679932 | 0.2500 | 114.74 s |
| `paper_full_gn_template_n122_b32_lr001` | 122 | 32 | 3,997,696 | 0.010 | 1 | 0.677887 | 0.2500 | 635.41 s |
| `paper_full_gn_template_n122_b32_lr0003` | 122 | 32 | 3,997,696 | 0.003 | 1 | 0.684097 | 0.7071 | 634.47 s |

An additional 60-step, batch-one trial at learning rate 0.003 was rejected:
accuracy fell from 0.672058 to 0.637497 over six outer updates while the inner
gradient norm rose to 14.33. Its trace is retained as a failed qualification,
not reported as a completed result.

The full-template learning rate 0.003 was selected because it improved the
one-step metric by 0.621 percentage points over 0.01 and did not pin the first
line search to the smallest candidate. A three-step persistence check produced
accuracies 0.684097, 0.685104, and 0.687241, so the run was resumed rather than
discarding its durable inner state.

## Interpretation boundary

The paper reports its largest iteration-complexity gains at 120M--240M tokens
per outer update and explicitly warns that preconditioning gains may not appear
at small batch size. The local four-hour budget permits the official 4M-token
template, but not even one 120M-token outer step at the measured throughput.
Results below therefore establish algorithm fidelity and local behavior, not a
replication of the paper's headline 5.4x or 16x iteration-reduction numbers.

## Formal result

The formal run stopped after completing and evaluating outer update 21. It did
not start a partial update after reaching the time limit.

| Quantity | Value |
|---|---:|
| Completed outer updates | 21 |
| Final validation accuracy | 0.715515 |
| Post-warmup wall time | 13,189.760 s |
| Total wall time including common warmup | 14,296.380 s (3.971 h) |
| Model-training tokens after warmup | 83,951,616 |
| Held-out line-search tokens | 83,951,616 |
| Final selected line-search step | 0.353553 |
| Peak allocated GPU memory | 9,327.795 MiB |

All 21 evaluated points improved monotonically from 0.684097 to 0.715515.
The result exceeded the retained AdamW final accuracy, 0.711754, by 0.003761
(0.376 percentage points). It did not exceed the retained Muon final accuracy,
0.752193; the gap is 0.036678 (3.668 percentage points). In update count, Full
Gauss--Newton exceeded Muon's first-epoch accuracy of 0.697289 at outer update
8, but remained 0.001041 below Muon's second-epoch accuracy of 0.716557 at the
four-hour limit.

The AdamW and Muon curves are retained historical runs from random
initialization, whereas this Full Gauss--Newton run includes the paper-required
common AdamW warmup. Therefore the plots are useful local references, not a
strict time-to-metric controlled comparison. A controlled claim requires
rerunning both baselines from the same warmup checkpoint and token stream.

The durable metric stream is
`metrics/nlp/nlp_gpt_12x512__paper_full_gn_template_n122_b32_lr0003.jsonl` and
the 691 MiB checkpoint is kept under `.cache/nlp/checkpoints`.

## Conclusion

The local run reproduces the paper's matrix-free Generalized Gauss--Newton
operator, persistent Muon inner solve, full gradient-and-curvature accumulation,
and held-out nonlinear line search. It reproduces strong per-outer-update
progress and beats the local AdamW endpoint, but it does **not** reproduce the
paper's headline advantage over Muon. The most direct reason is scale: this run
uses 3.998M tokens per direction, while the headline results appear at
120M--240M tokens per direction and substantially longer H100 runs. Claiming
the headline result from this four-hour local template would be unsupported.
