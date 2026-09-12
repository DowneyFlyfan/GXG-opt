# Muown 0.006 Perplexity Screen on ABA A100

## Configuration

Matched GPT2-12x512 Muown screen on ABA A100 device 0: learning rate 0.006,
weight decay 0.0, auxiliary AdamW learning rate 0.0003, micro-batch 8,
gradient accumulation 6, 2,034 updates, and 64 validation batches.

## Result

The screen completed in 1,690.476 seconds with validation negative
log-likelihood 0.96430952 and perplexity **2.62297593**.  This improves over
the 0.0075 and 0.010 screens but remains 0.00954410 above the 0.005 reference
at 2.61343183.  The next screen tests 0.0045 to determine whether the optimum
is below the existing 0.005 rate.
