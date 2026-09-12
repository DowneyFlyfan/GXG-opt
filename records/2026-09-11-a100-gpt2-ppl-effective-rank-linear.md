# A100 GPT2-12x512 perplexity comparison

## Scope

Historical NLP metric figures were next-token accuracy plots. Accuracy does not
identify token negative log likelihood, so perplexity cannot be reconstructed
from those curves. The recoverable optimizer-2.0 NLL traces were rendered as
exact PPL figures, and a new matched five-epoch comparison is running for the
four requested methods only: AdamW, Muon, Muown, and the scheduled
effective-rank optimizer.

All durable checkpoints are under `.cache/nlp/checkpoints`; no checkpoint is
written below `results/`.

## A100 protocol

- Host: `ABA`, GPU 0, NVIDIA A100 80GB PCIe. GPU 1 was occupied by another
  user and was never used.
- Model/data: retained 54,682,624-parameter GPT2-12x512 and cached
  WikiText-103 byte-token stream.
- Deterministic attention configuration; micro-batch 64, gradient
  accumulation 1, four loader workers, zero weight decay, and 64 validation
  batches per measurement.
- The batch was selected by an actual deterministic AdamW step (63.7 GiB
  peak). Larger batch 96 exhausted memory in cross-entropy; it produced no
  metric artifact.
- The final protocol is five epochs, same seed and data order for every
  method, and writes PPL at actual optimizer-step counts and measured elapsed
  wall-clock time.

## Screen selection

Each candidate was tested for 128 matched updates. Lower PPL is better.

| Method | Candidate LR | Validation PPL | Selected |
|---|---:|---:|:---:|
| AdamW | 0.0003 | 23.8939 | |
| AdamW | 0.0010 | 23.0013 | yes |
| Muon | 0.0100 | 11.6878 | yes |
| Muon | 0.0300 | 11.9441 | |
| Muown | 0.0100 | 11.6267 | yes |
| Muown | 0.0300 | 11.7388 | |
| Effective rank, linear 0.2 to 0.8 | 0.0100 | 11.6018 | yes |
| Effective rank, linear 0.2 to 0.8 | 0.0300 | 11.8789 | |

The selected scheduled-rank screen finished at floor 0.8 with 6,144 accepted
matrix updates, zero skipped matrix updates, and 2,851 rank-equalization
projections. The finite-step certificate is applied after each projection.

## Implementation and verification

`EffectiveRankLinear` schedules the lower bound from 0.2 on its first update
to 0.8 on its final update. When a matrix is below the current floor, its
singular values are minimally interpolated toward an equal spectrum before the
existing certified partial-polar update. This makes the high-rank stage
actionable instead of silently skipping updates. The checkpoint persists the
global schedule position.

Targeted tests passed: 15 tests covering the rank certificate, scheduler
checkpoint restoration, infeasible-matrix projection, the PPL renderer, and
the PPL comparison artifacts.

## Historical checkpoint PPL recovery

The archived accuracy records do not contain token losses and therefore cannot
be transformed into PPL curves. Instead, every available GPT2-12x512
checkpoint was loaded unchanged and evaluated with the same token-weighted NLL
definition on 64 held-out batches. This produced 26 endpoint records under
`results/nlp/checkpoint_perplexity/` and the clearly labelled endpoint-only
figure `results/nlp/gpt2_checkpoint_recovery_perplexity.png`; it does not
misrepresent a single endpoint as a training trajectory.

Selected recoveries, with lower PPL better, were Muown final: 2.30315,
Multi-Step Spectral final at LR 0.0004: 2.37578, Multi-Step Spectral final at
LR 0.0003: 2.37986, Nystr\u00f6m-Adam final: 2.56112, and Nystr\u00f6m-GGN
formal: 3.21231. Historical AdamW, Muon, and Effective-Rank-Half final weights
were not present, so their accuracy-only figures remain labelled as historical
accuracy rather than being falsely converted.

## Final-run status

The first A100 configuration was invalidated before completion; only the fresh
formal-rate protocol below can provide final PPL figures and a conclusion.

## Nautilus two-A100 request

On 2026-09-11, a bounded two-A100 Nautilus Job was submitted as
`new-optimizer-gpt2-dual-a100` using
`configs/experiments/nautilus_gpt2_dual_a100_reservation.yaml`.  The namespace
quota permits nine A100 requests and had only one active request before this
submission.  The scheduler accepted the Job, but no node with two schedulable
A100s was assigned.  Its pod remained Pending and the Job reached its
15-minute active deadline with zero completions; the terminal Job was then
deleted.  The manifest remains as the reproducible request evidence.  The live
ABA comparison remains single-GPU because its time-axis comparison is
intentionally matched; it was not modified or duplicated for the Nautilus
request.

## First completed A100 epoch

The invalidated effective-rank linear method completed epoch 1 at optimizer step 1,526:
validation PPL **10.63258** after **3,938.28 seconds** (1.094 hours).  Its
scheduled floor advanced to 0.32002; 73,248 matrix updates were accepted, zero
were skipped, and 33 rank-equalization projections were applied.  The process
then continued into epoch 2 under the same protocol.

## Invalidated high-rate A100 attempt

The label `aba_a100_ppl_final_b64` used micro-batch 64, accumulation 1, and
rates 0.01 (effective rank, Muon, Muown) and 0.001 (AdamW).  On 2026-09-12 it
was stopped after the user identified that these disagree with the completed
formal GPT2 comparison: effective-rank 0.00125, Muon 0.0025, Muown 0.005, and
AdamW 0.00015.  Its two PPL records and checkpoint remain as
invalid-protocol diagnostics only; they must not appear in a final comparison.

## Replacement formal-rate protocol

The replacement has the fresh label `aba_a100_ppl_formal_b8_a6_joint_newton`,
effective batch 48 (micro-batch 8, accumulation 6), and the four formal
learning rates above. Muown and the scheduled effective-rank method use zero
weight decay; AdamW and Muon retain formal 0.01 weight decay. The
effective-rank entry is the 0.2-to-0.8 linear schedule with the joint-Newton
efficient-solver path and certified fallback diagnostics. It starts from a new
model state and does not overlap with the invalid checkpoint.

## Fast scheduled projection revision

The joint-Newton direction solve uses matrix products, but the former schedule
recovery invoked a full singular-value decomposition whenever a weight fell
below the current floor. That recovery is outside the direction solve and made
an end-to-end runtime claim inappropriate.

Commit `ee90416` adds a full-rank fast path. It computes the Newton--Schulz
polar factor \(P\), searches the equalizing path \(W+aP\), rescales the accepted
matrix to retain \(\lVert W\rVert_F\), and accepts it only after the original
effective-rank predicate passes. A rank-deficient, nonconverged, or otherwise
unbracketed path explicitly falls back to the prior exact singular-value
projection. The final diagnostic records both `fast_projection_steps` and
`projection_fallback_steps`; hence a future result can distinguish an actual
fast path from a safety fallback.

The focused regression first failed against the old code because its fast-path
counter and norm-preserving recovery did not exist. After the change,
`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_effective_rank_half.py
tests/test_gpt2_ppl_experiment.py` passed 21 tests. This establishes the
certificate-preserving branch behavior; it is not an end-to-end GPU speed
claim. The active V2 A100 process had already imported the old source, so it
remains an isolated solver-quality trace. The committed revision is staged on
ABA for the next clean effective-rank run and does not mutate the live process.
