# Kronecker GGN optimizers

This repository now contains two separately selectable optimizers without changing the existing AdamW/Muon training path:

- `baseline_kronecker_ggn.KroneckerGGN`
- `low_rank_corrected_kronecker_ggn.LowRankCorrectedKroneckerGGN`

They share the registry, factors, exact joint-eigenvalue damping, spectral actions, fallback implementation, and curvature closure contract in `kronecker_ggn_common`. The corrected optimizer only adds the signed low-rank relative-residual estimate and corrected inverse action. Setting `correction_rank: 0` follows the baseline direction path exactly.

## Curvature definitions

`exact_ggn` means matrix-free products of `J.T @ H_loss @ J`. The reference implementation uses `torch.func` JVP/VJP transforms and includes analytical output-Hessian products for MSE and softmax cross-entropy. `mc_ggn` means likelihood-compatible Monte Carlo curvature tangents. `empirical_fisher` means outer products of observed gradients; it is deliberately named separately and is never relabeled as GGN.

For a linear layer, factors are `A = mean(a a.T)` and `B = mean(C)`. The baseline assumes independence and uses `A tensor B + damping * I`. It does not independently damp `A` and `B`. All production actions retain the weight's `[output_dim, input_dim]` shape and use `B @ X @ A.T`, avoiding flattening-convention ambiguity.

The corrected optimizer estimates signed eigenpairs of

`E = M^-1/2 @ (G + damping * I) @ M^-1/2 - I`.

Negative eigenvalues matter because the Kronecker model can overestimate curvature as well as underestimate it. Safety requires `1 + d > 0`; values below `-1 + correction_eigenvalue_margin` are clipped before the inverse action. Invalid, stale, poorly converged, unreliable, or over-budget corrections fall back to the shared baseline for that layer.

## Construction and curvature closure

The existing `src/optimizers.py` remains a module for the AdamW/Muon harness. A singular registry builder was added without converting that module into a package:

```python
from optimizers import build_optimizer
from kronecker_ggn_common import CurvatureUpdate

optimizer = build_optimizer(
    name="low_rank_corrected_kronecker_ggn",
    model=model,
    config=optimizer_config,
)

def curvature_closure(model, curvature_batch, registry):
    # Factor construction is owned by the training/curvature backend so that
    # batches, autocast, distributed synchronization, and MC sampling stay explicit.
    return CurvatureUpdate(
        curvature_mode="exact_ggn",  # or mc_ggn / empirical_fisher
        factors={layer_id: (activation_factor, output_factor)},
        ggn_operators={layer_id: exact_or_mc_operator},
    )

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = training_loss(model, batch)
    loss.backward()
    if optimizer.should_update_curvature():
        optimizer.update_curvature(curvature_closure, batch=curvature_batch)
    optimizer.step()
```

The closure signature is exactly `(model, batch, registry) -> CurvatureUpdate`. A plain mapping from layer ID to `(A, B)` is also accepted and is tagged with the optimizer's already resolved mode. Supplying `CurvatureUpdate` is preferred because it makes mislabeling detectable. The graph-producing closure is never called from `step()`.

`KroneckerFactorEstimator.from_output_curvatures` consumes exact/reference per-sample `C_i` matrices. `from_mc_tangents` and `from_empirical_gradients` are separate entry points. Factor and spectral refresh schedules are independent; corrected configurations add warm-up, correction refresh, maximum age, active-layer selection, optional cross-batch validation, and early refresh through `report_realized_decrease`.

## Support and storage

The MVP supports untied `nn.Linear.weight` tensors. Biases use a separate deterministic AdamW/SGD fallback. Embeddings, convolutions, tied weights, normalization tensors, fused/expert-specific ownership, and other parameters are explicitly registered as fallback rather than silently treated as supported. Optimizer construction logs every fallback parameter and reason.

Dense correction storage costs approximately `rank * output_dim * input_dim * bytes_per_element`. Both retained state and Lanczos workspace are checked before allocation. `dense_reference` and `selected_layers` are supported; compressed correction matrices are intentionally deferred until the core hypothesis is validated.

Optimizer `state_dict()` includes the resolved configuration, `curvature_mode`, factors, spectral ages, correction ages, fallback moments, and correction tensors. It excludes closures, operators, hooks, and autograd graphs. The same model registry and curvature mode are required when loading.

## Reproduction

Run the mathematical and integration suite:

```bash
PYTHONPATH=src pytest -q tests/optimizers
```

Generate the deterministic reference residual-spectrum diagnostic:

```bash
python scripts/diagnose_kronecker_residual.py
```

Benchmark bounded inverse-action overhead:

```bash
python scripts/benchmark_optimizer_overhead.py --rows 512 --columns 512 --rank 4
```

Every training comparison must include separately tuned AdamW and Muon runs, use a 10M–200M parameter model, and produce a combined metric-vs-step PNG:

```bash
python scripts/compare_optimizers.py \
  --adamw metrics/run__adamw.jsonl \
  --muon metrics/run__muon.jsonl \
  --kronecker-ggn metrics/run__kronecker_ggn.jsonl \
  --low-rank-corrected metrics/run__low_rank_corrected_kronecker_ggn.jsonl \
  --output results/run_metric_steps.png
```

The checked-in optimizer configurations are under `configs/optimizer/`. Tiny MLP/Transformer files under `configs/experiments/` are mathematical reference fixtures, not formal experiments. Formal model experiments remain subject to the repository's 10M–200M size limit, single-GPU rule, OOM checks, and four-hour cap.
