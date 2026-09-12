# GPT2-12x512 four-optimizer final perplexity comparison

## Scope

This record closes the matched five-epoch NLP GPT2-12x512 comparison among
AdamW, Muon, Muown, and the scheduled effective-rank joint-Newton optimizer.
Every trace contains five end-of-epoch validation measurements and 10,170
completed optimizer updates.  Perplexity is `exp(mean next-token negative log
likelihood)`, so lower is better.

## Common protocol

- Model: GPT2-12x512.
- Training duration: five epochs.
- Micro-batch size: 8; gradient accumulation: 6; effective batch size: 48.
- Validation: 64 batches at each epoch boundary.
- Optimizer parameters were the selected formal GPT2-12x512 settings:
  AdamW `lr=1.5e-4, weight_decay=0.01`; Muon `lr=2.5e-3,
  weight_decay=0.01, auxiliary_lr=5e-4`; Muown `lr=5e-3,
  weight_decay=0, auxiliary_lr=3e-4`; effective rank `lr=1.25e-3,
  weight_decay=0, auxiliary_lr=3e-4`.

## Final measured results

| Optimizer | Final PPL | Updates | Measured training time |
| --- | ---: | ---: | ---: |
| AdamW | 2.530269 | 10,170 | 9,672.480 s |
| Muon | 2.280087 | 10,170 | 9,744.465 s |
| Muown | 2.296619 | 10,170 | 9,769.596 s |
| Effective rank (0.2 to 0.8, joint Newton) | 2.386942 | 10,170 | 19,380.782 s |

The effective-rank trace was run on ABA's NVIDIA A100 80GB PCIe; the three
baseline traces were measured locally on an NVIDIA GeForce RTX 5070 Ti.  The
time plot therefore preserves actual wall-clock evidence but is not a
controlled cross-GPU throughput claim.  The step plot is the appropriate
optimization-quality comparison.

The fifth effective-rank epoch reached its scheduled floor of `0.8`.  Its
diagnostics recorded 488,160 accepted effective-rank updates, 488,160
certified-fallback updates, 841 fast-projection updates, and zero rejected or
skipped updates.  Thus this is an implemented effective-rank constrained
method, but the certified fallback handled the overwhelming majority of
accepted updates; it must not be described as a pure fast joint-Newton result.

## Artifacts

- Metric-versus-step figure:
  `results/nlp/gpt2_ppl_aba_a100_ppl_formal_b8_a6_joint_newton_four_way_steps.png`
- Metric-versus-time figure:
  `results/nlp/gpt2_ppl_aba_a100_ppl_formal_b8_a6_joint_newton_four_way_time.png`
- Local baseline JSONL evidence:
  `metrics/nlp/nlp_gpt_12x512__aba_a100_ppl_formal_b8_a6_joint_newton__{adamw,muon,muown}.ppl.jsonl`
- Effective-rank JSONL evidence:
  `metrics/nlp/nlp_gpt_12x512__aba_a100_ppl_formal_b8_a6_joint_newton_fast_v5__effective_rank_linear_joint_newton.ppl.jsonl`

The renderer is `write_labeled_ppl_comparison_plots` in
`src/gpt2_ppl_experiment.py`.  It keeps the original label attached to every
trace, allowing the ABA effective-rank continuation to be combined with the
local baselines without renaming, copying, or fabricating evidence.
