# Joint-Newton effective-rank solver: GPT2-12x512 screen

## Implementation

`effective_rank_linear_joint_newton` retains the existing linearly scheduled
effective-rank floor from 0.2 to 0.8.  For a smooth full-rank matrix gradient,
it uses a matrix-product-only Newton--Schulz polar derivative inside the joint
Newton solve.  The solver accepts a step only after its Schur solve, original
finite-step effective-rank check, spectral normalization, and dual-gap check.

Wide matrices are solved through the transpose-invariant tall formulation.
An eight-iteration Newton--Schulz preflight rejects ill-conditioned gradients
before the full twelve-iteration solve.  A rejected preflight uses the
previous bisection-certified partial-polar update for that matrix step.

## Matched local screen

- Hardware: NVIDIA GeForce RTX 5070 Ti, 16 GB.
- Model: GPT2-12x512, 54,682,624 parameters.
- Data: cached NLP data; seed 1337.
- Both methods: effective-rank floor 0.2 to 0.8 over 16 updates, learning rate
  0.01, micro-batch 8, gradient accumulation 6, no weight decay.
- Validation: 8 batches after update 16.

| Method | PPL at update 16 | Train + validation seconds | Joint Newton / fallback matrix steps |
| --- | ---: | ---: | ---: |
| Existing linear certified solver | 102.9218 | 36.5393 | 0 / 0 |
| Linear joint-Newton solver | 102.9115 | 36.5391 | 0 / 768 |

The new path did not claim a speedup.  At the first GPT update, selected
matrix-gradient singular-value ratios were approximately `1e-8` to `1e-6`;
the Newton--Schulz preflight orthogonality residual remained much larger than
the `1e-8` smoothness requirement.  The fallback therefore preserved the
finite-step guarantee and reproduced the established trajectory, rather than
applying a regularized but uncertified joint step.

## Current decision

Keep the live ABA A100 five-epoch comparison unchanged.  It is measuring the
already selected linear effective-rank optimizer against AdamW, Muon, and
Muown.  The joint-Newton method needs a separately validated regularized or
stable-subspace extension before it can replace that protocol on GPT2.
