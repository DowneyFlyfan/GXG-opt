# Qwen3-0.6B old-curve audit

## Evidence

Visual review of `results/nlp/qwen3_0p6b_tied_path_lr1e4_s1000_mb4a2_20m_tied_path_curvature_v1_metric_steps.png` shows:

- AdamW validation perplexity oscillates from roughly 16.56 at step 1,000 to 16.43 at 2,000, 16.54 at 3,000, and 16.49 at 3,700.
- Muon and Muown rise from roughly 16.53--16.61 through step 2,000 to about 17.20--17.21 at step 3,000, then remain near 17.15 at step 3,700.
- The tied-path curvature trace has only one visible point, at step 1,000 with perplexity about 17.82, so it does not establish a stable training trend.

## Conclusion

A pretrained-versus-scratch mismatch can invalidate a pretraining comparison, but it alone does not explain the late Muon/Muown regression. Their optimizer settings and schedule must also be tuned.

## Replacement safeguards

The replacement protocol uses Qwen3-0.6B scratch initialization, a shared 16-million-token epoch, effective batch size 64, 100-update held-out evaluations, 300-update parameter screens for AdamW/Muon/Muown, cosine decay across the 610 final updates, and only promotes the lowest held-out perplexity screen candidate for each baseline.
