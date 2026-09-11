# Optimizer 2.0 migration and bounded validation

The six proposals from `/home/justin/new-optimizer/optimizer/2.0` are now native
GXG-opt modules under `src/optimizer_v2/`, with no runtime dependency on that
workspace. This follows the spectral migration (`4c5eb1a`) by bringing the design
contracts, implementation, configurations, tests, and runnable GPT-2 comparison
together. Migration started from `938829c0c4cd2ff251fd4088ac09f2a2b323e8be`.

## Scope and preservation

- Methods: `qk_defect_v1`, `routing_resistance_v1`, `tied_path_curvature_v1`,
  `proposal_notch_v1`, `feature_remap_cohort_v1`, and `ln_response_v1`.
- All six mathematical modules (`attention`, `layernorm`, `linalg`, `optimizer`,
  `probes`, `temporal`) are byte-identical to the source workspace. Every original
  adapter class/function is AST-identical; the existing parameter-selection
  helper is colocated to remove the source-workspace import.
- The native experiment module retains the approved five-epoch, eight-method,
  random GPT-2/WikiText-103 screen. Fixed settings and complete epochs take
  precedence over generic repository tuning/time-limit defaults for this profile.
- Native imports, configuration paths, source snapshots, exports, and the CLI
  replace the source workspace's legacy-runner dependencies. A local step/time
  renderer implements GXG-opt's experiment reporting convention.
- The existing spectral runner, optimizer implementations, A100 configurations,
  and repository-wide requirements are unchanged. Asset preparation adds an
  opt-in random-initialization path; its default pretrained behavior and manifest
  are preserved.
- The isolated v2 environment uses PyTorch 2.11.0+cu130 and Transformers 5.7.0.
  Integration tests skip the incompatible PyTorch version in other profiles.
  Only missing declared dataset/plot dependencies are installed. Existing packed
  assets were materialized locally for validation; no symlink is required.

## Validation

```bash
PYTHONPATH=src:scripts .venv-gpt2-v2/bin/python -m pytest -q \
  tests/test_optimizer_v2_math.py tests/test_optimizer_v2_integration.py \
  tests/test_gpt2_spectral_comparison.py tests/test_multi_step_spectral_geometry.py
```

**66 passed.** Coverage includes independent FP64 mechanisms, CPU/CUDA stock
optimizer parity, disabled equivalence, tied aliases/fused QKV, cohort/filter
resume, exact epoch coverage/partial accumulation, auxiliary failure isolation,
nonfinite-training checkpoint recovery, full validation, completed-run checksum
verification, both asset-initialization modes, and generated plot artifacts.

Shell syntax and Python compilation passed. The v2 dry run reports eight methods,
36,250 updates per method, five epochs, seed 0, effective batch 32, and
593,920,000 input tokens per method. The existing spectral dry run also passed.

The full repository suite stops during collection because the unchanged
`tests/optimizers/integration/test_optimizer_equivalence.py` imports
`build_optimizer`, which is absent from legacy `src/optimizers.py`. The identical
failure was reproduced in a clean archive of the pre-migration commit under the
same interpreter. It is outside this migration and was not changed.

The original source workspace's 121-test result and GPU checks are historical
source verification; the 66-test result above was executed in GXG-opt itself.

A clean archive of the staged source (without datasets, a local environment, or
source-workspace files) passed the standalone CLI dry run and all **24 independent
mathematical tests**. This checks the native import/configuration layout in a
fresh extraction rather than relying on the working checkout's search path.

## Native full-size GPU smoke

```bash
bash scripts/run_gpt2_v2_5090.sh --output results/gpt2_v2_migration_smoke --steps 33
```

All eight methods completed 33 updates on the RTX 5090 with production auxiliary
schedules, batch 8, accumulation 4, sequence length 512, and random 124M GPT-2.
Each processed 540,672 input tokens from matching initialization. Full validation
ran initially and at smoke completion. There were no numerical fallbacks.

| Method | Updates | Peak allocated GiB |
|---|---:|---:|
| adamw | 33 | 9.441 |
| muon | 33 | 9.173 |
| qk_defect_v1 | 33 | 9.124 |
| routing_resistance_v1 | 33 | 9.124 |
| tied_path_curvature_v1 | 33 | 9.417 |
| proposal_notch_v1 | 33 | 9.344 |
| feature_remap_cohort_v1 | 33 | 9.414 |
| ln_response_v1 | 33 | 9.124 |

Both metric-versus-step and metric-versus-time PNGs were generated, include all
eight methods, and are labelled as bounded smoke checks. The time plot was also
visually checked. Repeating the same launcher command exited successfully, skipped
all eight verified completed runs, and left every final checkpoint unchanged.

These short warmup runs validate execution and migration, not training quality.
They do not complete the 128-update feature prediction diagnostic or natural
notch activation; those transition paths have separate bounded tests.

All run artifacts remain under the ignored smoke directory. Neither the full
five-epoch comparison nor any remote/cloud run was started. The commit contains
source/configuration/design/test files and this record, without environments,
datasets, model weights, or checkpoints. Original workspace files were preserved.

To start or resume the full native comparison:

```bash
cd /home/justin/GXG-opt
bash scripts/run_gpt2_v2_5090.sh --output results/gpt2_v2_5090_seed0_5epochs
```

## Integration with rewritten remote main

Before pushing, remote `main` had been replaced by an unrelated history ending
at `7e0395a` (effective-rank-third work). The unpublished v2 commit was replayed
onto that current history. The original local tip `385c779` is retained on the
local backup branch `backup/main-before-v2-sync-20260910`.

The README conflict was resolved by preserving the remote text and appending the
v2 section. All optimizer, experiment, launcher, configuration, and test files
remain byte-identical to the verified v2 commit. Upstream effective-rank code,
results, repository instructions, and removals were retained.

Post-integration verification: **76 focused tests passed**, covering the v2,
spectral, and incoming effective-rank suites. Both experiment dry runs and shell
syntax passed. The full repository collection limitation documented above is
outside this focused check. Full training remains unlaunched.
