# Fixed-frequency GN-guided AdamW

This opt-in package implements the fixed Phase-4 prototype from
`gn_guided_adam_implementation_plan.md`. AdamW remains primary. Each eligible
weight tensor receives a low-rank, gradient-seeded Krylov GGN subspace; Adam acts
only on its implicit orthogonal complement and GGN acts only inside it.

The rough-experiment schedule is a deterministic epoch duty cycle. The checked-in
fixed config guides for one epoch, runs AdamW-only for one epoch, and repeats. The
first optimizer step after each off-to-on transition forces a curvature refresh;
the ordinary fixed step interval still applies within on epochs.

Adaptive block selection and adaptive refresh are not implemented. Setting
`adaptive.enabled: true` is a configuration error until the fixed MVP passes its
GPU research gate.

## Minimal integration

```python
from gn_guided_adam import FunctionalBatch, GuidedStepContext, GuidedAdamConfig
from gn_guided_adam import gn_guided_adamw

config = GuidedAdamConfig.from_yaml(
    "src/gn_guided_adam/configs/gn_guided_fixed.yaml"
)
optimizer = gn_guided_adamw(model, config)

loss.backward()  # real training gradients populate Adam moments
result = optimizer.step(GuidedStepContext(
    curvature_batch=FunctionalBatch(curvature_args, curvature_loss_fn),
    acceptance_batch=FunctionalBatch(acceptance_args, acceptance_loss_fn),
    gradient_accumulation=accumulation,
    tokens=tokens_in_step,
    epoch=epoch,
))
optimizer.zero_grad(set_to_none=True)
```

The caller owns all loaders and must independently sample the training, curvature,
and acceptance data where practical. Curvature/acceptance reuse flags are logged.
Candidates include their learning-rate factor but exclude decoupled weight decay;
weight decay is applied exactly once before the chosen direction and only for models
with at least 100M parameters, following repository policy.

`gn.enabled: false` needs no step context and is numerically equivalent to AdamW
with the same supported single parameter group. On a scheduled refresh, missing or
nonfinite curvature, solve, trust, or acceptance data falls back to that same AdamW
candidate. Adam moments are always updated only from the real `.grad` tensors.

Useful local verification commands:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
py -3 -m pytest tests/test_gn_guided_config_blocks.py tests/test_gn_guided_ggn_krylov.py tests/test_gn_guided_subspace_acceptance.py tests/test_gn_guided_optimizer.py tests/test_gn_guided_reporting.py -q
py -3 -m pyflakes src/gn_guided_adam tests/test_gn_guided_config_blocks.py tests/test_gn_guided_ggn_krylov.py tests/test_gn_guided_subspace_acceptance.py tests/test_gn_guided_optimizer.py tests/test_gn_guided_reporting.py
```

## Evidence and experiments

CPU tests validate operator identities, Krylov and reduced solves, subspace
orthogonality, trust/staleness, stateless candidate evaluation, fallback parity, and
checkpoint resume. They are not research results. No Gate A/B/C status, time-to-
target improvement, GPU-hour reduction, run log, or plot is claimed until real GPU
runs exist. The reporting module produces paired AdamW/GN-guided PNGs versus wall
time, optimizer steps, and tokens after real JSONL experiment logs are available.
