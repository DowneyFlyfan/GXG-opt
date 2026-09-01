# Paper-faithful Full Gauss-Newton reproduction plan

Run the retained 54.68M GPT-12x512 language model on cached WikiText-103 with
the algorithm from *The Potential of Second-Order Optimization for LLMs: A
Study with Full Gauss-Newton*. This is a scaled reproduction on the current
baseline task, not a claim to reproduce the paper's C4/LLaMA/H100 numbers.
Existing AdamW and Muon traces remain historical comparisons; the controlled
comparison starts all three methods from one AdamW warmup checkpoint.

| Phase | Deliverable | Gate |
|---|---|---|
| Integration | Matrix-free quadratic gradient and Muon inner solver | each inner gradient equals \(g_0+G_{GN}(\theta-\theta_0)\) |
| Paper controls | Persistent inner optimizer, pre-line-search warm start, held-out line search | behavior matches the official implementation |
| Qualification | One outer step with memory/time and exact batch accounting | no OOM; effective batch is at least 60 sequences |
| Runs | Paper-GN, AdamW, and Muon records from one warmup checkpoint | each run stops and checkpoints at four hours |
| Reporting | Metric-step and metric-time PNGs plus a `records` report | all traces and wall-clock measurements are reproducible |

The production path uses only data and checkpoints under `.cache`. It keeps
the model, tokenizer, validation metric, sequence length, seed, and data order
fixed. The GGN matrix is never materialized.

## Paper algorithm contract

At outer parameters \(\theta_0\), each fresh inner mini-batch supplies the
matrix-free gradient of the quadratic Taylor model,
\(g_0+J_0^T H_L J_0(\theta-\theta_0)\). Muon updates eligible hidden matrices;
AdamW updates embeddings, normalization parameters, position parameters, and
the tied language-model head. Inner optimizer state persists across outer
steps, and the next inner solve starts from the previous pre-line-search end
point.

After the inner loop, evaluate step sizes
\(1,1/\sqrt{2},1/2,1/(2\sqrt{2}),1/4\) on entirely new mini-batches and accept
the one with minimum true nonlinear loss. Apply outer exponential weight
averaging only as a separately labelled ablation. Use the paper's initial
inner settings as the first qualification point: Muon learning rate 0.01,
momentum 0.95, weight decay 0.001 on Adam-routed parameters and zero on Muon
matrices, gradient clipping 1, and an inner cosine schedule.

The paper's 45M template uses 122 inner steps with 32 sequences per step,
roughly four million tokens per outer update. That exact batch is not expected
to fit a four-hour RTX 5070 Ti run, so qualification first measures the largest
safe physical mini-batch and then increases the outer batch with inner steps
and gradient accumulation. A result is called paper-faithful only if the
algorithmic contract above is preserved; reduced inner count is reported as a
scaled local reproduction.

Source: `THE POTENTIAL OF SECOND-ORDER OPTIMIZATION FOR LLMS: A STUDY WITH
FULL GAUSS-NEWTON.pdf` and the cached official repository at
`.cache/reference/full-gauss-newton`.
