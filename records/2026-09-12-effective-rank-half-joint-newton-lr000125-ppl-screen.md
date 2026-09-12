# Static Effective-Rank-Half Joint-Newton 0.00125 Screen

## Configuration

This matched GPT2-12x512 screen used the certified joint-Newton variant with
the effective-rank floor fixed at 0.5 for all 2,034 updates.  It used ABA A100
device 1, learning rate 0.00125, weight decay 0.0, auxiliary learning rate
0.0003, micro-batch 8, gradient accumulation 6, and 64 validation batches.

## Result

Validation perplexity was **2.90923607** after 2,031.044 seconds, better than
the base fixed-half solver at 2.91308164 but still not competitive with the
2.61343183 Muown reference.  The diagnostic record confirms the static floor
was 0.5.  It also recorded zero accepted joint-Newton steps and 71,711
certified fallbacks, so the joint solver did not materially take over this
screen.  A 0.002 fixed-half joint-Newton screen is now running on the same
GPU, still with the fixed 0.5 constraint.
