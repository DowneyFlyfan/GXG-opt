# Muown 0.004875 Perplexity Screen

GPT2-12x512 on ABA A100 device 0, 2,034 updates, effective batch 48, and 64
validation batches: learning rate 0.004875, weight decay 0.  Validation NLL
was 0.96338303 and perplexity was **2.62054688** after 1,800.55 seconds.

This is worse than both LR 0.0045 (2.61440876) and the completed 0.005
reference (2.61343183), so the local quadratic interpolation is rejected.
Device 0 continues with a narrow independent 0.005125 screen while device 1
continues the static-effective-rank-0.5 momentum-0.98 screen.
