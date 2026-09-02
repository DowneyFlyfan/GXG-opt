# Recycled low-rank Gauss--Newton experiment plan

Implement and evaluate a compute-efficient approximation to the paper-faithful
Full Gauss--Newton (GN) baseline on the retained 54.68M-parameter GPT-12x512
WikiText-103 task.  The controlled run must start from the existing common
AdamW warmup checkpoint, use the same model, tokenizer, sequence length, data
order, validation metric, and held-out line-search protocol, and receive the
same post-warmup wall-clock budget as the completed Full GN run.

The method approximates the inverse action of the damped system

\[
  \left(G+\lambda I\right)p=-g
\]

without constructing either \(G\) or its inverse. It combines a cheap
factored-diagonal Kronecker base with a signed low-rank approximation of the
whitened relative residual. Each outer update accumulates exactly one full-batch
curvature product along the normalized gradient. That same product calibrates
the Kronecker base and contributes one secant direction to a recycled rank-four
projected residual. The approximate inverse is then applied directly; repeated
full-batch preconditioned conjugate-gradient scans are retained only as an
ablation because they are computationally impractical at equal batch size.

| Stage | Deliverable | Verification gate |
|---|---|---|
| Numerical core | Factored-diagonal Kronecker base, signed residual basis, SPD inverse application, optional PCG | dense toy systems match direct solves and remain positive definite |
| Recycling | Persist and update the low-rank projected secant subspace | rank grows to four and remains bounded |
| Integration | Same warmup, gradient batch, curvature subset, held-out line search, checkpoint/resume | accounting and resume tests pass |
| Qualification | Local-GPU memory and one-complete-outer-step probes | no OOM or non-finite direction; completed step has a metric |
| Formal run | Up to the same 13,189.76-second post-warmup budget as Full GN | stop only after a completed outer step and validation |
| Reporting | Metric-step and metric-time comparisons with AdamW, Muon, and Full GN | PNGs, JSON/JSONL, detailed record, exact wall time and Gv counts |

## Initial controlled configuration

- Model: GPT-12x512, 54,682,624 parameters.
- Task and data: cached WikiText-103, context length 1024, unchanged seed and
  training stream.
- Start point: `.cache/nlp/checkpoints/nlp_gpt_12x512__paper_common_adamw_warmup.pt`.
- Outer gradient and line-search batches: 3,904 sequences each, matching the
  completed paper-template Full GN run.
- Physical batch: 2, the largest safe shared batch for exact full-model GGN
  products on the local 16.3 GiB GPU. Gradient and curvature each accumulate
  1,952 physical batches to the same 3,904-sequence effective batch.
- Curvature batch: all 3,904 outer-gradient sequences, accumulated with exact
  sample weighting. The final physical batch is 2 for both statistics.
- Low-rank residual: signed projected-secant rank 4, updated by one full-batch
  curvature product per accepted outer update.
- Direct inverse action: damping 0.1 and held-out step scales 0.1, 0.0707,
  0.05, 0.0354, and 0.025. The one-step qualification selected 0.0707 and
  improved validation accuracy from 0.656700 to 0.661888.
- Large checkpoints and cached tensors: `.cache/nlp/checkpoints` only.
- Time limit: 13,189.76 post-warmup seconds; the last started outer update is
  completed and evaluated before stopping.

## Acceptance and comparison rules

1. Unit tests are written before implementation and must cover exact toy-system
   recovery, positive definiteness, eigenvalue clipping, recycling, and line
   search restoration.
2. Formal comparisons use validation metric versus completed outer step and
   validation metric versus elapsed post-warmup time.  Loss curves are not
   reported.
3. Performance is reported honestly.  The controlled result is judged against
   the retained AdamW, Muon, and paper Full GN traces at their recorded elapsed
   time; no target metric is treated as guaranteed.
4. Qualification runs may use smaller batches only when clearly labelled;
   they cannot be substituted for the controlled formal result.
5. Only files belonging to this method are staged, so unrelated user changes
   in the existing dirty worktree remain untouched.

## Execution checklist

- [x] Add failing numerical and experiment-contract tests.
- [x] Implement the Kron-RPCG numerical core.
- [x] Integrate the paper-task data, warmup, line search, metrics, and resume path.
- [x] Run CPU tests and local-GPU memory/numerical qualifications.
- [x] Run the controlled same-duration local-GPU experiment.
- [x] Draw metric-step and metric-time PNGs and write the experiment record.
- [x] Run final verification and push only the method-specific changes/results.
