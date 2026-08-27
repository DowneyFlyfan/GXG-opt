# GPT 12x512 Optimizer Tuning

## Scope

One deterministic WikiText-103 next-token epoch, using the cached 100M-token
subset. The model has 54,682,624 trainable parameters. Each run used seed
1337, micro-batch size four, two gradient-accumulation micro-batches, and the
same validation batches.

## Results

| Optimizer | Learning rate | Validation next-token accuracy | Epoch time | Decision |
|---|---:|---:|---:|---|
| AdamW | 0.0003 | 0.540630 | 723.08 s | retained baseline |
| Muon | 0.0200 | 0.285038 | 801.28 s | rejected: metric lower than AdamW |
| Muon | 0.0050 | 0.553341 | 801.87 s | selected for formal training |

## Divergence Check

The first formal-shaped AdamW run used learning rate 0.0003 and was stopped
after two epochs. Its deterministic validation metric was 0.602539 at epoch
one and 0.468681 at epoch two. This sharp decline shows that the one-epoch
pilot was insufficient evidence of convergence, so the run is not reported as
a formal baseline. The revised AdamW candidate uses learning rate 0.0001 and
will be evaluated for two pilot epochs before a new formal pair starts.

The selected Muon pilot exceeds AdamW by 10.9% in complete epoch time, which
is within the 20% execution-time constraint. The full twelve-epoch formal
run is estimated at about 2.67 hours for Muon and 2.41 hours for AdamW.

## Conclusion

The next GPT 12x512 pilot compares AdamW learning rate 0.0001 with Muon
learning rate 0.005 for two epochs. Pilot files are deliberately not retained
in the formal metrics or results directories; this record is the permanent
tuning evidence.

## Muon Follow-up

With the stable AdamW comparison in hand, Muon learning rate 0.005 reached
only 0.500240 in its first matching pilot epoch, below AdamW's 0.606117. The
candidate was stopped before its second epoch and is rejected. The next Muon
candidate uses learning rate 0.0025 for two epochs.

The initial 0.0025 trial reached 0.629993 and 0.631954, but its auxiliary
AdamW learning rate was still the old hard-coded 0.0003. This was not a fair
comparison against the selected AdamW baseline. The optimizer now exposes the
auxiliary learning rate per task; the decisive rerun uses Muon 0.0025 and
auxiliary AdamW 0.0001.

## Selected Pair

| Optimizer | Matrix learning rate | Auxiliary AdamW learning rate | Epoch 1 | Epoch 2 |
|---|---:|---:|---:|---:|
| AdamW | 0.0001 | not applicable | 0.606117 | 0.646858 |
| Muon | 0.0025 | 0.0001 | 0.676453 | 0.710655 |

Muon is higher at both matched checkpoints. The formal pair therefore uses
these settings with twelve epochs, the same seed, the same data order, and
the same effective batch size.

## Reproducibility Correction

The original data loaders relied on PyTorch's process-global random generator
for shuffled batches. A formal rerun therefore did not reproduce the pilot
trajectory. Each loader now receives its own seed-derived generator, and a
test proves that two seeded shuffled token loaders emit identical batches.
All earlier pilot and partial-formal metrics are discarded. The same selected
settings are rerun before producing any formal result.

## Reproducible Selection

| Optimizer | Epoch 1 | Epoch 2 |
|---|---:|---:|
| AdamW, learning rate 0.0001 | 0.605087 | 0.649330 |
| Muon, matrix learning rate 0.0025, auxiliary learning rate 0.0001 | 0.605614 | 0.691399 |

The two deterministic data loaders now use the same seeded batch order within
the pair. This is the sole tuning result used for the formal NLP-512 pair.

## Final Deterministic Qualification

The earlier figures in this record predate the final deterministic attention
configuration. They are retained as rejected tuning history only and are not
used to support the formal comparison. The final qualification disables the
non-deterministic Flash and memory-efficient scaled-dot-product attention
kernels, uses the mathematical attention kernel, enables deterministic
algorithms in warning mode, and preserves the independently seeded loader
generator.

| Optimizer | Matrix learning rate | Auxiliary AdamW learning rate | Epoch 1 | Epoch 2 | Two-epoch time |
|---|---:|---:|---:|---:|---:|
| AdamW | 0.0001 | not applicable | 0.631744 | 0.670967 | 4014.04 s |
| Muon | 0.0025 | 0.0001 | 0.697289 | 0.725788 | 4167.45 s |

Muon exceeds AdamW at both checkpoints. Its elapsed-time increase is 3.82%,
which satisfies the experiment's 20% limit. The formal five-epoch AdamW run
started only after this qualification; its first metric is 0.631744, exactly
matching the qualified pilot. The corresponding formal Muon run remains
queued behind AdamW and will use the same selected settings.
