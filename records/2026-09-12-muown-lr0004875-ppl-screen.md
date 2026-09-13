# Muown 0.004875 Perplexity Screen

GPT2-12x512 on ABA A100 device 0, 2,034 updates, effective batch 48, and 64
validation batches: learning rate 0.004875, weight decay 0.  Validation NLL
was 0.96338303 and perplexity was **2.62054688** after 1,800.55 seconds.

This is worse than both LR 0.0045 (2.61440876) and the completed 0.005
reference (2.61343183), so the local quadratic interpolation is rejected.
Device 0 continues with a narrow independent 0.005125 screen while device 1
continues the static-effective-rank-0.5 momentum-0.98 screen.

## Independent local replication

The matched local RTX 5070 Ti run `local_5070ti_muown_lr0004875_m095_e1`
completed 2,034 updates in 1,954.170 seconds with negative log-likelihood
0.96491698 and perplexity **2.62456975**. It confirms the A100 rejection:
the local value differs by 0.00402287 perplexity but remains worse than the
0.005 reference by 0.01113792. The 0.004875 learning rate remains rejected.
