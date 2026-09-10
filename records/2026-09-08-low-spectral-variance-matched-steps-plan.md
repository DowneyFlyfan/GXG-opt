# Matched-step Low-Spectral-Variance continuation plan

## Goal

Run Low-Spectral-Variance on GPT-2 12x512 with the same micro-batch 12,
gradient accumulation 4, five-epoch, 10,170-optimizer-step schedule used by
the retained AdamW and Muon references.

## Constraint and execution

The measured direct-Cayley implementation needs about 7,200 seconds per
matched-schedule epoch.  A five-epoch invocation would exceed the repository's
four-hour experiment cap.  Execute five continuations at completed epoch
boundaries instead: targets 1, 2, 3, 4, and 5.  The checkpoint contains the
model, optimizer, scheduler, cumulative optimizer steps, and cumulative timing;
therefore each continuation starts at the next full epoch without replaying a
partial epoch.

## Fixed settings

- GPT-2 12x512 and cached NLP data
- micro-batch 12, gradient accumulation 4
- condition cap 8, one Newton-Schulz dual iteration, momentum 0.95
- previously selected Low-Spectral-Variance learning rate `0.003`
- 14,300 second cap for each one-epoch continuation

## Evidence to retain

Retain the JSONL trace after each continuation, final result JSON, both required
metric-step and metric-time comparisons against tuned AdamW/Muon, and a record
that distinguishes matched optimizer-step count from prior batch-144 results.
