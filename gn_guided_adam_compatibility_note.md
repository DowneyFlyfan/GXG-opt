# GN-Guided AdamW Compatibility Note

## Repository ownership

- `src/training.py` owns loader advancement, gradient accumulation, backward,
  optimizer/scheduler stepping, CUDA BF16 autocast, validation, and result artifacts.
- `src/optimizers.py` exposes the existing AdamW and Muon baselines. Its Muon kernel
  is decorated with `torch.compile` at import time; PyTorch 2.0.1 on Windows cannot
  collect that baseline module, so the new package must not import it.
- There is no common scaler, checkpoint, DDP/FSDP, fused-optimizer, activation-
  checkpointing, or optimizer-callback abstraction. Models are ordinary PyTorch
  modules; NLP output weights are tied functionally to the embedding tensor.
- Formal training is CUDA-only and uses task-dependent accumulation. No short
  baseline training job is run in this CPU-only implementation pass. The existing
  selected CPU tests pass before changes (47 tests).
- The existing `src/gxg_opt_adam` package contains matrix-free JVP/HVP/VJP work, but
  GN-guided AdamW has different block, subspace, acceptance, and state semantics. It
  remains a separate implementation rather than silently reusing its controller.

## New package boundary

`src/gn_guided_adam` is an opt-in package. The caller performs a normal backward
pass and supplies already-selected curvature and independent acceptance batches in
a step context. The optimizer never advances a loader and never reads validation or
test data. With guidance disabled, it requires no context and follows AdamW.

The fixed prototype uses one eligible weight tensor per block. Biases,
normalization parameters, embeddings, tied tensors, sparse tensors, output heads,
and tensors below the configured size threshold are unguided and remain ordinary
AdamW parameters. Adaptive block selection and adaptive refresh are deliberately
deferred.

## Numerical and state conventions

- Adam candidates include the learning-rate factor but exclude decoupled weight
  decay. Weight decay is applied once immediately before the selected direction.
- Adam first and second moments are updated once from the real training gradients
  before candidate construction. Curvature, Krylov, oracle, or rejected candidate
  data never enter those moments.
- Per-block GGN products use functional calls with cloned buffers and
  JVP/output-loss-HVP/VJP. Production code never materializes a Jacobian or GGN
  matrix.
- A fixed epoch duty-cycle switches guidance on and off. Fixed-rank bases refresh
  on each off-to-on transition and at a fixed step interval during on epochs.
  Between refreshes, deterministic age/drift staleness limits apply. No adaptive
  refresh policy is active.
- Candidate comparison is functional and restores module modes and RNG state. A
  nonfinite build, solve, scale, or acceptance result selects the AdamW candidate.
- Checkpoints combine model, Adam state, bases, reduced matrices, damping,
  cooldowns, telemetry, RNG, and optional scheduler/scaler/sampler state.

## Verification scope

Only deterministic CPU unit and tiny numerical integration tests are authorized in
this pass. No GPU training, dataset download, time-to-target claim, Gate A/B/C
decision, run log, or experiment plot is fabricated. Configurations and benchmark
commands are provided for a later GPU experiment; research gates remain explicitly
unevaluated until those runs exist.
