# Generalized Gauss--Newton reference retuning

## Scope

This record replaces an incorrect K-FAC interpretation. The retrained NLP
baseline is the 54,682,624-parameter `gpt_12x512` causal language model on the
cached WikiText-103 task. The earlier layer-wise Kronecker approximation is an
ablation, not the original Gauss--Newton solver; the active reference path is
matrix-free full generalized Gauss--Newton (GGN).

## Sources used

- Nicol Schraudolph, *Fast Curvature Matrix-Vector Products for Second-Order
  Gradient Descent* (2002): the positive-semidefinite generalized GGN used for
  classification losses.
- James Martens, *Deep Learning via Hessian-Free Optimization* (2010): damped
  curvature solves and Levenberg--Marquardt actual-versus-predicted reduction.
- `ltatzel/PyTorchHessianFree`, a public Martens-style reference implementation:
  adaptive damping, CG warm starts, backtracking, and line search.

The original sources do **not** prescribe K-FAC's factored damping or its
gradient-space momentum. Those changes are excluded from the reference path.

## Exact full-GGN feasibility and initialization check

An all-parameter matrix-free operator now computes \(Gv = J^T H_\ell Jv\)
with forward- and reverse-mode automatic differentiation. Its small linear
least-squares test matches the analytic Hessian exactly. On the 54.68M model,
a one-sequence, 128-token curvature batch completed one exact product in
0.453 seconds with 1,857 MB peak allocated GPU memory, so an exact
Hessian-free implementation is feasible on the local 16 GB GPU.

The same probe found that the original decoder initialization gives logit
standard deviation 22.756 and an exactly zero numerical GGN product for a
unit gradient direction: the softmax Hessian is saturated at the start. A
standard GPT-style initialization changes those two measurements to 0.455 and
46.139 respectively. The runner now applies that initialization to each fresh
local GPT baseline, so future AdamW, Muon, and GGN traces can be compared from
the same nonsaturated state. The existing AdamW/Muon files were produced
before this correction and are not a fair comparator for the corrected GGN
trace.

## Full-GGN batch-size qualification

The older table below this section belongs to the rejected Kronecker ablation;
it must not determine the exact full-GGN batch size.  Full GGN has a much
larger matrix-free curvature footprint, so each candidate was tested with a
complete damped-CG update on the local 16,303 MB GPU.

| Batch x sequence length | Tokens / GGN update | Peak allocated memory | Result |
|---:|---:|---:|---|
| 2 x 512 | 1,024 | 9,108 MB | stable, but leaves excessive memory unused |
| 2 x 768 | 1,536 | 13,000 MB | stable, slower than 2 x 512 |
| 4 x 512 | 2,048 | 12,807 MB | stable |
| 4 x 600 | 2,400 | 15,175 MB (93.1%) | stable; selected |
| 4 x 640 | 2,560 | 15,260 MB before a further 210 MB allocation | out of memory |

The selected full-GGN configuration is therefore batch size four and sequence
length 600.  It uses the largest observed stable configuration, leaving about
1.13 GB of physical-device headroom for transient allocator requests.  This
selection follows the requested near-full-GPU policy without treating an OOM
as an acceptable normal operating condition.

## Pure full-GGN retuning evidence

All values below are NLP validation token accuracy.  Each full-GGN candidate
uses the same cached WikiText-103 stream and Armijo/Levenberg--Marquardt logic.

| Initialization | Curvature update | Damping | Metric evidence | Decision |
|---|---|---|---|---|
| original | 4 x 600, CG 1 | 200 then floor 20 | 0.26485 at 256; 0.27463 at 512 | rejected: saturated curvature and slow progress |
| GPT-style | 4 x 600, CG 1 | 1 then floor 0.001 | 0.19261 at 32, 64, and 96 | rejected: one CG iterate does not change predictions |
| GPT-style | 4 x 600, CG 2 | 1 then floor 0.001 | 0.28365 at 256; 0.28709 at 512 | rejected: only +0.00344 in 268.6 seconds |
| GPT-style | 4 x 600, CG 2 | 1 then floor 0.1 | 0.19236 at 32; 0.24235 at 64 | rejected: stronger damping is worse |
| GPT-style | 4 x 512 x 2 sequential windows, CG 2 | 1 then floor 0.001 | 0.19261 at 32 | rejected: lower throughput and no early gain |
| GPT-style | 4 x 600, four-loader-batch gradient/GGN accumulation, CG 2 | 1 then floor 0.001 | 0.26110 at 69 outer steps / 240.61 s | rejected: 9,600-token statistics but slower time-to-metric |

The two-CG case has accepted Armijo steps and faithful curvature prediction
(reduction ratios 0.861 at step 256 and 0.988 at step 512), so this is not a
line-search failure.  It is a time-to-metric failure of exact full GGN on this
54.68M vocabulary-softmax language model. It cannot honestly be called better
than the retained pre-correction Muon metric of 0.75219; moreover that Muon
trace must be rerun after the initialization correction before a fair claim is
possible.

The accumulation implementation holds the GGN matrix fixed across each CG run
and averages four distinct loader mini-batches sequentially. It reached 15.80
GB device usage (96.9%) without an OOM. Thus the negative result is not caused
by unused GPU memory; it shows that this four-batch accumulation alone is
insufficient.

## Required next decision

The pure-GGN tuning space has exhausted the supported batch, CG, damping, and
curvature-window variants without evidence it can win inside four hours.
Continuing requires either an explicit switch to a GGN-guided first-order
method or permission to report the negative pure-GGN result. Neither choice is
made by this record.

## Sources

- https://nic.schraudolph.org/pubs/Schraudolph02.pdf
- https://www.cs.utoronto.ca/~jmartens/docs/Deep_HessianFree.pdf
- https://github.com/ltatzel/PyTorchHessianFree
