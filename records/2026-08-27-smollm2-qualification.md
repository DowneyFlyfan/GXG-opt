# SmolLM2-135M qualification

- Model: cached local `SmolLM2-135M`, 134,515,008 parameters.
- Data: cached tokenizer-specific WikiText-103 streams in `.cache/nlp`.
- ABA was not fully available: its two GPUs reported 100 percent utilization,
  so qualification ran locally on the RTX 5070 Ti.

## Capacity and timing

| Optimizer | Microbatch | Accumulation | Update time | Step time | Peak memory |
|---|---:|---:|---:|---:|---:|
| AdamW | 2 | 8 | 0.96987 s | 0.01537 s | 7,657.20 MiB |
| Muon | 2 | 8 | 1.00126 s | 0.04594 s | 7,262.37 MiB |

The measured Muon-to-AdamW update-time ratio is 1.0324. Two passes over the
100M-token stream are estimated at about 3.4 hours including validation,
within the two-to-four-hour target.

## Learning-rate pilot

An isolated 2M-token cache was derived from the existing cache, used only for
the pilot, then moved to the system trash. Both pilots used microbatch 2 and
accumulation 8.

| Optimizer | Matrix LR | Auxiliary LR | Accuracy | Wall-clock |
|---|---:|---:|---:|---:|
| AdamW | 0.00001 | not applicable | 0.475014 | 122.42 s |
| Muon | 0.00010 | 0.00001 | 0.483910 | 126.94 s |

Muon 0.0001 improved the pilot metric while its pilot time ratio was 1.0370.
The formal pair therefore uses AdamW 0.00001 and Muon 0.0001, with auxiliary
AdamW 0.00001, microbatch 2, accumulation 8, and two epochs.

## Formal result

| Optimizer | Final accuracy | Wall-clock | Peak GPU memory |
|---|---:|---:|---:|
| AdamW | 0.505775 | 3h 17m 54s | 7,657.20 MiB |
| Muon | 0.520187 | 3h 24m 13s | 7,255.11 MiB |

Muon divided by AdamW wall-clock time is 1.0319, within the required 1.20
limit, while improving the final validation next-token accuracy by 0.014412.
The paired time-annotated metric graph is
`results/nlp/nlp_smollm2_135m_metric_steps.png`.
