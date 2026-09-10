# Fixed GN-Guided AdamW Decision Report

## Current decision

The implementation correctness gate is covered by CPU unit and tiny numerical
tests, including the fixed epoch on/off schedule. The research premise is **not
yet evaluated** because no GPU training or checkpoint-continuation experiment was
authorized on this workstation.

| Gate | Status | Required missing evidence |
|---|---|---|
| AdamW baseline reproducibility | Not evaluated | Three tuned GPU seeds and fixed validation target |
| Gate A: oracle cost-adjusted value | Not evaluated | Checkpoint oracle probes including future Adam time saved |
| Gate B: low-rank capture | Not evaluated | Rank sweep, capture, cost, and independent-batch acceptance |
| Gate C: fixed MVP | Not evaluated | Three-seed time-to-target comparison including all overhead |
| Adaptive refresh | Deferred | Gate C must pass before implementation |

No claim is made about fewer training steps, lower wall time, lower GPU-hours, or
validation quality. The next experiment should predeclare one validation target,
run the checked-in `adamw_baseline.yaml`, then probe selected checkpoints with
`gn_oracle_probe.yaml`. If the oracle fails Gate A, the correct decision is to stop
before tuning the fixed hybrid.
