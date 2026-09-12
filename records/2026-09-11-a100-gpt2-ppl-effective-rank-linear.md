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

The selected five-epoch A100 protocol started with the effective-rank method.
This record will be completed with its epoch metrics, the remaining three
methods, final PPL figures, and an evidence-backed conclusion after all four
runs finish.

## Nautilus two-A100 request

On 2026-09-11, a bounded two-A100 Nautilus Job was submitted as
`new-optimizer-gpt2-dual-a100` using
`configs/experiments/nautilus_gpt2_dual_a100_reservation.yaml`.  The namespace
quota permits nine A100 requests and had only one active request before this
submission.  The scheduler accepted the Job but it is currently `Pending`; no
node with two schedulable A100s has been assigned.  It expires after 15 minutes
if not admitted, so it cannot hold idle GPUs.  The live ABA comparison remains
single-GPU because its time-axis comparison is intentionally matched; it was
not modified or duplicated for the Nautilus request.
