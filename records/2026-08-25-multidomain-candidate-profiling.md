# Multi-Domain Muon Candidate Profiling

## Method

Every value is the median of ten synchronized complete parameter updates on an
NVIDIA GeForce RTX 5070 Ti. Each update includes forward propagation,
backpropagation, configured gradient accumulation, AdamW or reference-style
compiled bfloat16 Muon update, and the auxiliary AdamW update.

## Results

| Domain | Model | Parameters | AdamW | Muon | Muon overhead | Peak memory AdamW/Muon |
|---|---|---:|---:|---:|---:|---:|
| NLP | GPT 12x512 | 54,682,624 | 32.378 ms | 35.742 ms | 10.4% | 3185/3132 MiB |
| NLP | GPT 12x768, accumulation 4 | 110,335,488 | 110.307 ms | 129.769 ms | 17.6% | 3269/2939 MiB |
| CV | ViT Base 12x768, accumulation 4 | 85,875,556 | 127.687 ms | 145.521 ms | 14.0% | 2655/2333 MiB |
| Audio | AST Base 12x768, accumulation 5 | 85,661,184 | 111.460 ms | 129.754 ms | 16.4% | 2252/1935 MiB |

## Conclusion

All four candidates satisfy the required 10M–200M parameter range, fit the
GPU, and keep the complete-update Muon overhead at or below 20%. The old
uncompiled float32 baseline is excluded. The next gate is a real training
pilot: it must show an approximate validation-metric plateau in the 2–4 hour
budget derived from these measured update times.
