# `gxg_opt_adam`

This package implements the callback-driven `gxg_optimizer` described by the
repository coding plan. The first rough-experiment controller uses a fixed epoch
duty cycle; adaptive probe-based switching is intentionally deferred. Layer-wise
generalized Gauss--Newton, AdamW, and controller state remain independent and
serializable.

## Construction

```python
from gxg_opt_adam import GXGConfig, GXGOptimizer

config = GXGConfig.from_yaml("src/gxg_opt_adam/configs/gxg_optimizer.example.yaml")
gxg_optimizer = GXGOptimizer(model=model, config=config, metric_fn=validation_metric)
```

The caller owns loaders and supplies the zero-based current `epoch` and already-
selected data through `StepContext`. `duty_cycle.gn_epochs` and
`duty_cycle.adam_epochs` repeat in the configured `start_phase` order. A GN step
needs a `FunctionalBatch` for curvature and a disjoint `reference_loss_closure` for
line search. During GN, `shadow_microbatch_gradients` must be real gradients; their
mean squares, not the square of their mean, update Adam's shadow variance. Bridge
steps additionally need a larger reference-gradient mapping.

`nominal_budget_exhausted=True` routes the controller to
`FINAL_QUALITY_CHECK`. Call `evaluate_quality(EvalContext(...))` with a validation
metric to enter conservative recovery or finish. The API rejects a `test` split for
all controller decisions.

## Checkpointing

`state_dict()` contains all three optimizer banks, transition history, bridge state,
best validation state, and RNG state. `checkpoint_payload()` additionally packages
the model and optional scheduler, scaler, and sampler. `save_checkpoint()` writes the
combined payload with atomic replacement.

## Scope and safety

- No production path materializes a full Hessian or GN matrix.
- GN proposals never enter Adam moment buffers.
- Weight decay is disabled for models below 100M parameters, following the repository
  rule that it is only used for big models.
- The included tests are small deterministic CPU checks. They are not training
  experiments and do not make speed or validation-quality claims.
- Gradient-noise, alignment, and gain-per-second helpers are observability-only in
  this version; they cannot trigger a phase transition.
- `SingleProcessLayerAdapter` is implemented. A Megatron full-layer owner can satisfy
  `DistributedLayerAdapter`; this package intentionally does not pretend a generic
  process group can reconstruct tensor-parallel logical layers without framework
  metadata.
