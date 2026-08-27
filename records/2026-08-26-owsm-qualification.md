# OWSM v3.1 Base qualification

## Setup

- Model: `espnet/owsm_v3.1_ebf_base`, 101,182,628 parameters.
- Task: LibriSpeech automatic speech recognition from cached raw audio.
- Optimizers: AdamW and Muon with AdamW auxiliary parameters.
- Microbatch: one utterance; gradient accumulation: eight utterances.
- AdamW learning rate: `1e-5`.
- Muon learning rate: `1e-4`; auxiliary AdamW learning rate: `1e-5`.
- Weight decay: `0.01`, because this is a 101M-parameter model.

## Routing audit

- The first two-dimensional convolution is AdamW-only.
- Convolutions failing the flattened channel/kernel ratio rule are AdamW-only.
- Token embeddings, CTC projection, and decoder output projection are AdamW-only.
- Eligible Transformer and convolution projection matrices use Muon.

## Timing result

| Optimizer | Complete update | Optimizer step | Peak memory |
|---|---:|---:|---:|
| AdamW | 0.29665 s | 0.01229 s | 2432.50 MiB |
| Muon | 0.33315 s | 0.03111 s | 2265.32 MiB |

Muon divided by AdamW is 1.1230, below the project limit of 1.20.

The train split has approximately 28,500 utterances. With eight accumulated
microbatches, one epoch has about 3,567 optimizer updates. Eight epochs are
estimated at about 2.4 hours per optimizer, before final validation overhead.

## Short tuning pilot

Both optimizers used the same data order and eight accumulated microbatches.
AdamW at update 30 had mean speech-to-text loss 695.38. Muon at update 30 had
mean loss 704.98. Neither run diverged, so the listed learning rates are used
for the formal pair.

## Interrupted first attempt

The first AdamW formal attempt was interrupted by the local non-persistent
terminal after four completed epochs. It did not save model or optimizer state,
so it cannot be resumed from epoch five without misrepresenting a new model as
continuation. Its validation character-error-rate values were 0.205850,
0.150436, 0.126999, and 0.119186 for epochs one through four. They are kept as
diagnostic evidence only; the formal pair restarts from epoch one in a durable
local tmux session.
