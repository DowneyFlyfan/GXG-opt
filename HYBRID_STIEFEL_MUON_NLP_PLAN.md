# Hybrid Muon / Stiefel-Muon NLP experiment

## Hypothesis

The first and last Transformer blocks may need unconstrained singular-value
adaptation.  Keep them on ordinary Muon and apply scaled Stiefel-Muon only to
the ten interior blocks.

## Exact routing

| Parameters | Count | Optimizer |
|---|---:|---|
| `blocks.0.*` eligible matrices | 4 | Muon |
| `blocks.1` through `blocks.10` eligible matrices | 40 | Stiefel-Muon |
| `blocks.11.*` eligible matrices | 4 | Muon |
| embeddings, tied output, normalization, biases | remaining | AdamW auxiliary |

## Protocol

1. Validate routing and manifold algebra with unit tests.
2. Reuse the safe physical batch 12 / accumulation 4 configuration, then run
   short isolated probes for the Stiefel learning rate while keeping edge-Muon
   learning rate near its separately tuned baseline value.
3. Run the strongest stable setting for at most four hours on the same
   GPT-12x512 WikiText-103 task, seed, and five-epoch target.
4. Save metric JSONL, result JSON, checkpoint in `.cache`, Metric--Steps PNG,
   Metric--Time PNG, and a detailed record; compare with retained AdamW,
   ordinary Muon, and the all-Stiefel result.
