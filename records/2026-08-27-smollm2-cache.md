# SmolLM2-135M offline data and routing audit

## Cached assets

- Model: `HuggingFaceTB/SmolLM2-135M`, loaded from
  `.cache/huggingface/models/SmolLM2-135M` without network access.
- Data: cached WikiText-103 raw text, tokenized with the model's local
  tokenizer and end-of-sequence identifier.
- Training cache: `smollm2_wikitext103_train_100000000.pt`, 100,000,000
  token identifiers.
- Validation cache: `smollm2_wikitext103_validation_10000000.pt`, 10,000,000
  token identifiers.

## Real loader verification

The cached training stream contains 97,656 next-token sequences and the
validation stream contains 263 sequences.  A real batch has input and target
shape 2 by 1,024 with 64-bit integer token identifiers.  The observed maximum
identifier was 48,912, below the 49,152-token model vocabulary.

## Muon routing audit

The token embedding and tied language-model output head are AdamW-only.
The first attention query projection and first feed-forward gate projection
are eligible for Muon.  In total, 210 matrix parameters are selected for
Muon.  This verifies that the Natural Language Processing baseline follows the
embedding/head exclusion rule before GPU qualification begins.
