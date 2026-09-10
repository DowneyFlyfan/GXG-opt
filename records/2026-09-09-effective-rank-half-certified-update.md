# Effective-Rank-Half Certified Update

## Implemented core

- Source: `records/ideas/effective_rank_half_finite_step_certified_update.md`.
- `src/effective_rank_half.py` implements the finite-step condition
  \(\mathcal E(W)=\|W\|_F^4/(r\|W^\top W\|_F^2)\ge1/2\).
- The direction is the rank-aware partial polar factor of the gradient, so
  its spectral norm is at most one.
- If a full step fails the exact finite-step condition, bisection selects the
  largest scalar in `[0, 1]` that retains feasibility.  The implementation
  does not claim to solve the paper note's full dual inner optimization.

## Verification

- `tests/test_effective_rank_half.py` verifies a feasible update and a
  boundary case that requires backtracking.
- CPU result: 2 passed.
- The actual 54,682,624-parameter GPT2-12x512 model has 48 Muon-eligible
  matrices; 37 satisfy the initial effective-rank-half condition and will use
  the certified optimizer, while the remaining parameters stay in AdamW.

## Experiment status

- The requested A100 allocation remained Pending. On 2026-09-10 the user
  explicitly authorized a local fallback, so the single aggressive screen
  `lr0005_b8_a6_screen` was launched on the local GPU with learning rate
  0.005, micro-batch 8, accumulation 6, zero weight decay, and one epoch.
- The screen must finish with a valid metric and finite certificate behavior
  before a fresh five-epoch matched local final is launched. No duplicate
  trainer is permitted.

## Screen 1: aggressive learning rate 0.005

`lr0005_b8_a6_screen` completed one epoch in 3,293.14 seconds with 8,481 MB
peak memory and zero weight decay. It produced validation next-token accuracy
**0.64654541**, which is 0.00373332 below the retained AdamW epoch-one result
(0.65027873) and 0.06512578 below Muon (0.71167119). The aggressive rate is
therefore rejected; it is finite and completed cleanly, but it is not eligible
for five-epoch promotion. The next bounded screen is the lower 0.00125 rate.

## Screen 2: lower learning rate 0.00125

`lr000125_b8_a6_screen` completed one epoch cleanly in 3,292.52 seconds with
8,481 MB peak memory and zero weight decay. It produced validation next-token
accuracy **0.70956039**: +0.05928167 relative to the selected AdamW epoch-one
baseline (0.65027873), -0.00211080 relative to Muon (0.71167119), and
-0.00535393 relative to Muown (0.71491432). It is finite, preserves the
configured constraint path, and is sufficiently competitive to promote the
exact batch-8, accumulation-6, learning-rate-0.00125 configuration to the
required fresh five-epoch GPT2-12x512 comparison. The completed screen left no
checkpoint under `results/nlp`; final checkpoints are confined to `.cache`.

## Promoted five-epoch final

The fresh matched final `lr000125_b8_a6_final` started from a new model state
with the same learning rate, micro-batch, accumulation, and zero decay. Its
first epoch completed in the expected 55-minute range and exactly reproduced
the screen metric: **0.70956039**. The live 515,807,577-byte resume checkpoint
is located only at
`.cache/nlp/checkpoints/nlp_gpt_12x512__lr000125_b8_a6_final__effective_rank_half.checkpoint.pt`;
no checkpoint exists under `results/nlp`. Epochs 2--5 remain in progress.

### Live epoch evidence

Epoch 2 completed at **0.73014259**, a finite +0.02058220 increase from the
reproduced epoch-1 value (0.70956039). The sole trainer continued directly
into epoch 3; its resume checkpoint was refreshed only under `.cache` at
2026-09-10T12:29:08, and a fresh scan again found no matching checkpoint below
`results/nlp`. This remains an in-progress matched final, not a final
comparison claim.

Epoch 3 then completed at **0.74310493**, a further +0.01296234 gain over
epoch 2. The sole trainer advanced directly into epoch 4. Its 515,807,577-byte
resume checkpoint was refreshed at 2026-09-10T13:23:30 under `.cache` only;
the matching `results/nlp` checkpoint scan remained empty.

Epoch 4 completed at **0.75121689**, a further +0.00811195 gain over epoch 3.
The sole trainer advanced directly into epoch 5. Its 515,807,577-byte resume
checkpoint was refreshed at 2026-09-10T14:17:44 under `.cache` only; the
matching `results/nlp` checkpoint scan remained empty. This is 0.00185585
below the matched Muown epoch-four metric (0.75307274), so it is promising but
not yet a final-baseline improvement claim.

## Five-epoch final result

The promoted `lr000125_b8_a6_final` run completed all five matched epochs:
**0.70956039**, **0.73014259**, **0.74310493**, **0.75121689**, and
**0.75524139**. The final metric is 0.00020981 above the selected Muon result
(0.75503159), but 0.00093651 below the selected Muown result (0.75617790).
It therefore establishes a competitive certified candidate rather than a new
best baseline. The run completed in 16,315.66 seconds with 8,481.33 MiB peak
allocated memory. Its completed checkpoint was automatically removed from
`.cache`, and the matching `results/nlp` checkpoint scan remained empty.

The final four-way step and wall-clock plots are
`results/nlp/effective_rank_half_lr000125_b8_a6_final_metric_steps.png` and
`results/nlp/effective_rank_half_lr000125_b8_a6_final_metric_time.png`.

## Screen 3: upper-neighbour learning rate 0.00175

`lr000175_b8_a6_screen` completed one fresh epoch in 3,294.61 seconds with
the matched micro-batch 8, accumulation 6, and zero weight decay configuration.
Its validation next-token accuracy was **0.70899963**, which is 0.00056076
below the 0.00125 incumbent's epoch-one result (0.70956039). The run completed
cleanly, removed its `.cache` checkpoint, and is rejected: raising the rate from
0.00125 to 0.00175 is not an improvement.
