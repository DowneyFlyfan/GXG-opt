# Fixed Effective-Rank-Half Momentum 0.8 Screen

- GPT2-12x512 on ABA A100 device 1; 2,034 updates, batch 48, 64 validation batches.
- `EffectiveRankHalf`, rank floor fixed at 0.5, learning rate 0.00125,
  momentum 0.8, no weight decay.
- Result: validation NLL 1.18914012; perplexity **3.28425592** in 1,888.75 s.

The floor remained exactly 0.5, with 74,976 accepted and zero skipped
matrix updates.  Momentum 0.8 is worse than the matched 0.95 screen
(2.91308164), so it is rejected.  Device 1 immediately continues with the
same fixed floor/LR and momentum 0.98; Muown continues independently on
device 0.
