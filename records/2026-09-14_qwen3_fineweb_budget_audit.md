# Qwen3-0.6B FineWeb-Edu data and runtime audit

## Decision being audited

The active Qwen3 optimizer study uses the cached pretrained
`Qwen/Qwen3-0.6B` causal language model and deterministic token packing from
`HuggingFaceFW/fineweb-edu`, configuration `sample-10BT`.  This is continued
pretraining, not training a 0.6-billion-parameter model from random
initialization.  Therefore a high-quality, reproducible held-out language
corpus is the relevant requirement; a from-scratch Chinchilla token budget is
not asserted by this experiment.

## Model evidence

The cached `config.json` on ABA identifies `Qwen3ForCausalLM`, 28 transformer
layers, hidden width 1,024, MLP width 3,072, 16 query heads, 8 key/value heads,
151,936 vocabulary items, bfloat16 weights, and tied input/output embeddings.
The cached model directory occupies 1.5 GiB.  The study name and upstream
checkpoint identify the model as Qwen3-0.6B.

## Dataset evidence

The immutable cache manifest at
`.cache/qwen3_0p6b_fineweb_edu/manifest.json` records:

| field | value |
| --- | ---: |
| source | `HuggingFaceFW/fineweb-edu/sample-10BT` |
| source revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| train tokens | 2,000,000,000 |
| validation tokens | 100,000,000 |
| sequence length | 2,048 |
| train documents | 1,948,209 |
| validation documents | 96,949 |
| packed cache size | 7.9 GiB |
| token representation | little-endian uint32 |

The packing code hashes document identifiers and rejects a document appearing
in both partitions.  It also verifies each packed token file's exact byte size
and SHA-256 digest before a trial begins.  The active formal traces share
manifest digest
`ced8d55d8e0849b22e5c6678194d0ca31fe42998b0760fea8c6d99ab00090649`.

## Matched formal protocol

Every baseline uses three epochs, micro-batch 8, no gradient accumulation,
sequence length 2,048, seed 1,337, 64 validation batches every 1,000 optimizer
updates, and zero data-worker randomness.  Each committed update exposes
16,384 train tokens.  One epoch therefore contains approximately
122,070 committed updates, and three epochs expose 6.0 billion tokens per
optimizer.  AdamW and Muon occupy separate A100 80GB GPUs.  Their peak observed
allocations are respectively 68,225.61 MiB and 67,445.61 MiB, so batch 8 uses
the available memory closely without an unverified larger batch.

## Observed throughput and feasibility boundary

At the latest inspected periodic points, AdamW had processed 212,992,000
tokens in 11,067.56 seconds (about 19,244 tokens/s inclusive of validation),
and Muon had processed 180,224,000 tokens in 10,528.82 seconds (about 17,118
tokens/s inclusive of validation).  Extrapolating those observed rates gives
approximately 28.9 to 32.5 hours per 2-billion-token epoch and 86.7 to 97.4
hours per three-epoch optimizer.  This is an estimate, not a completion claim;
later steps may differ.

Consequently, the selected cache is appropriate for an auditable, substantial
continued-pretraining comparison of this model, but it is not an appropriate
choice for a two-hour end-to-end optimizer screen.  The active formal runs are
not changed by this audit.  Their common data and three-epoch contract must
remain fixed; shortening one existing run would invalidate the required
matched baseline comparison.  If a fast screening tier becomes necessary, it
must be a separate, explicitly labeled cache and matched protocol rather than
an alteration of these formal baselines.
