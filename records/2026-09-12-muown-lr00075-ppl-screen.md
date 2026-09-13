# Muown 0.0075 Perplexity Screen on ABA A100

## Configuration

This was the second matched Muown screen after the 0.010 rate was rejected:
ABA A100 device 0, learning rate 0.0075, weight decay 0.0, auxiliary AdamW
learning rate 0.0003, micro-batch 8, gradient accumulation 6, 2,034 updates,
and 64 validation batches.

## Result

The one-epoch screen completed in 1,689.682 seconds with validation negative
log-likelihood 0.96783157 and perplexity **2.63223046**.  It remains worse
than the completed 0.005 reference at 2.61343183, and is statistically
indistinguishable at this precision from the rejected 0.010 screen
(2.63231118).  The next Muown screen narrows the bracket to 0.006; it runs on
the other A100 from the fixed-half joint-Newton screen.

## Independent local replication

The matched run `local_5070ti_muown_lr00075_m095_e1` independently repeated
this setting on the local NVIDIA GeForce RTX 5070 Ti. It used the same 2,034
updates, micro-batch 8, gradient accumulation 6, zero weight decay, and 64
validation batches. It completed in 1,953.716 seconds at negative
log-likelihood 0.96740168 and perplexity **2.63109914**. This agrees with the
A100 result within 0.00113132 perplexity and is still worse than the 0.005
reference by 0.01766731, so 0.0075 remains rejected.
