# Spectral Unit-Ball Full-GGN Local Screen

## Scope

This record implements and screens
`records/ideas/spectral_unit_ball_gauss_newton_derivation.md` on the retained
54,682,624-parameter GPT2-12x512 WikiText task.  It is a short local-GPU
screen, **not** a five-epoch final comparison.

The implementation restricts the full Generalized Gauss--Newton (GGN) action
to the 48 eligible matrix blocks (37,748,736 parameters).  For every projected
gradient iteration it computes `B delta = J^T C J delta` through one
matrix-free Jacobian--vector product (JVP) and vector--Jacobian product (VJP).
Thus cross-block terms among every selected layer remain present.  The fixed
batch, parameters, and curvature operator are not changed within the inner
solve.  Each block is projected exactly by singular-value clipping to
`||X_l||_2 <= 1`; measured accepted norms were at most 0.1736 and later were
about 0.005.

The nonmatrix 16,933,888 parameters are optionally updated by separate AdamW;
this does not replace or alter the restricted full-GGN curvature system.

## Environment

- Device: local NVIDIA GeForce RTX 5070 Ti, 16,303 MiB.
- Model: GPT2-12x512, 54,682,624 parameters.
- Curvature/update micro-batch: 1; no gradient accumulation.
- Held-out validation: four batches after each update.
- Seed: 1337.
- Weight decay: zero for the spectral and auxiliary paths.
- Peak allocation: 2,857--3,343 MiB.

## One-step trust-scale bracket

All entries start from the same fresh model and batch.  `beta` is the projected
quadratic step denominator; smaller values are more aggressive.

| initial beta | accepted beta | auxiliary AdamW LR | PPL after update 1 |
|---:|---:|---:|---:|
| 1 (backtracked) | none through 64 | 0 | 32642.96 |
| 128 | 128 | 0 | **6735.49** |
| 144 | 144 | 0 | 6778.03 |
| 192 | 192 | 0 | 6890.12 |
| 256 | 256 | 0 | 7025.50 |
| 512 | 512 | 0 | 7570.11 |
| 1024 | 1024 | 0 | 8996.70 |

`beta=128` was the best pure-GGN one-step setting.  The `beta=1` case made no
update because every projected candidate increased the **fixed GGN quadratic**;
no unsafe direction was accepted.

## Twenty-update matched local screen

The controls use the same GPT2 model, seed, device, micro-batch, data stream,
20 optimizer updates, and four validation batches.  They are short screens,
not substitutes for the separately completed five-epoch A100 records.

| method | settings | PPL at update 20 | optimizer seconds |
|---|---|---:|---:|
| AdamW | LR 1.5e-4 | 278.82 | 0.82 |
| Muon | matrix LR 2.5e-3, auxiliary LR 5e-4 | **17.60** | 1.59 |
| spectral GGN, no auxiliary | beta 128, one inner iteration | 1865.97 | 45.45 |
| spectral GGN + AdamW | beta 128, one inner iteration, auxiliary LR 3e-4 | 57.12 | 47.20 |
| spectral GGN + AdamW | beta 256, one inner iteration, auxiliary LR 5e-4 | 25.18 | 47.19 |
| spectral GGN + AdamW | beta 256, **two inner iterations**, auxiliary LR 5e-4 | 23.59 | 68.80 |

The best spectral screen accepted all 20 updates.  It is materially better
than the pure spectral-only path, but is 5.99 PPL above matched Muon and about
43 times slower at this early horizon.  It therefore is not promoted to a
five-epoch baseline or mixed-device time graph.  The next justified action is
to extend this best candidate over a longer local screen and compare its curve
with matched local controls before committing a five-epoch run.

## Verification

`PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_spectral_unit_ball_gn.py tests/test_full_ggn.py`

passed: 12 tests.  The new tests prove independent singular-value projection,
joint cross-block curvature use on a coupled quadratic, and preservation of
the language-model logits shape by the analytic softmax cross-entropy Hessian
product.

## Artifacts

- Code: `src/spectral_unit_ball_gn.py`,
  `src/spectral_unit_ball_gn_experiment.py`, and
  `src/run_spectral_unit_ball_gn.py`.
- Best screen metric:
  `metrics/nlp/nlp_gpt_12x512__local_5070ti_spectral_unit_ball_full_ggn_aux5e4_s20_i2_b1_beta256__spectral_unit_ball_full_ggn.ppl.jsonl`.
- Best screen result:
  `results/nlp/nlp_gpt_12x512__local_5070ti_spectral_unit_ball_full_ggn_aux5e4_s20_i2_b1_beta256__spectral_unit_ball_full_ggn.ppl.json`.
- Matching local controls:
  `metrics/nlp/nlp_gpt_12x512__local_5070ti_b1_a1_s20_spectral_control__{adamw,muon}.ppl.jsonl`.
