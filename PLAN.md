# Transformer baseline plan

The formal comparison contains four Transformer-family model pairs.  Each pair
uses independently tuned AdamW and Muon settings, cached data only, metric by
epoch records, a paired PNG, and a Markdown result record.  Each PNG states
the measured wall-clock time for both optimizers.

| Domain | Model | Parameters | AdamW | Muon | Current state |
|---|---|---:|---|---|---|
| Natural Language Processing | GPT 12x512 | 54.68M | complete | complete | retained baseline |
| Natural Language Processing | SmolLM2-135M | 134.52M | pending | pending | offline adapter complete |
| Computer Vision | DINOv3 ViT-B/16 | 85.74M with head | running | queued | qualified; AdamW formal run in progress |
| Audio | OWSM v3.1 Base | 101.18M | complete | complete | committed paired result |

## Execution gates

- Verify the parameter count is in the 10M–200M range.
- Use one fully available ABA Graphics Processing Unit when possible; otherwise
  run locally after a Graphics Processing Unit memory preflight.
- Route the first convolution, ineligible convolutions, embeddings,
  normalizations, and output heads to AdamW; apply Muon to eligible matrices.
- Profile both optimizers before a formal run and select a batch/accumulation
  setting that avoids out-of-memory failure and keeps the full experiment under
  four hours.
- Tune AdamW and Muon independently.  Draw the paired metric-versus-epoch PNG
  after both runs; include each wall-clock duration in the image.
- Write settings, qualification measurements, results, and conclusion to
  `records`, then commit and push the finished experiment.

## Immediate work

Complete the DINOv3 ViT-B/16 AdamW formal baseline from the qualified cached
model. It uses a frozen patch embedder and first eight Transformer layers;
the matched Muon baseline is queued and will produce the paired time-annotated
metric-versus-epoch graph after both result files exist.
