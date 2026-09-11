#!/usr/bin/env bash
# Five matched GPT2-12x512 PPL runs selected by the A100 learning-rate screen.
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$project_root/scripts/run_gpt2_ppl_comparison.py" \
  --label aba_a100_ppl_final_b64 \
  --methods effective_rank_linear muon muown adamw \
  --maximum-epochs 5 \
  --micro-batch-size 64 \
  --gradient-accumulation 1 \
  --workers 4 \
  --validation-batches 64 \
  --weight-decay 0 \
  --effective-rank-learning-rate 0.01 \
  --muon-learning-rate 0.01 \
  --muown-learning-rate 0.01 \
  --adamw-learning-rate 0.001
