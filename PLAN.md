# Baseline execution plan

## Retained scope

The completed GPT 12x512 Natural Language Processing baseline is retained.
The active public-model pairs are SmolLM2-135M for Natural Language Processing
and OWSM v3.1 Base for Audio. DINOv3 ViT-B remains deferred because its
publisher-gated checkpoint has not been approved.

| Domain | Model | Parameters | AdamW | Muon | Status |
|---|---|---:|---|---|---|
| NLP | GPT 12x512 legacy | 54.68M | complete | complete | retained |
| NLP | SmolLM2-135M | 134.52M | pending | pending | adapter and qualification pending |
| Audio | OWSM v3.1 Base | 101.18M | complete | running | local pair in progress |
| CV | DINOv3 ViT-B | 85.66M | deferred | deferred | manual access pending |

## Rules

- Use AdamW and Muon only.
- Keep model sizes from 10M through 200M parameters.
- Use AdamW for the first convolution, ineligible convolutions, embeddings,
  normalization parameters, and output heads.
- For every accepted pair, tune independently and require Muon complete-update
  time to remain within 20% of AdamW.
- Persist only metric-versus-epoch records and paired PNG graphs; do not write
  loss curves.
- Use one fully free GPU on ABA when it is available; otherwise run locally.

## Current evidence

OWSM passed the local timing gate at one microbatch with eight accumulated
microbatches: 0.29665 seconds per AdamW update and 0.33315 seconds per Muon
update. The final AdamW baseline reached a character error rate of 0.104833
after eight epochs in 9780.96 seconds. The Muon run uses the qualified `1e-4`
learning rate and the same data order, effective batch, and epoch count.
