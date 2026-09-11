#!/usr/bin/env bash
# Short, matched learning-rate screen before the five-epoch PPL runs.
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
for setting in \
  adamw:0.0003:adamw_lr3e4 \
  adamw:0.001:adamw_lr1e3 \
  muon:0.01:muon_lr1e2 \
  muon:0.03:muon_lr3e2 \
  muown:0.01:muown_lr1e2 \
  muown:0.03:muown_lr3e2 \
  effective_rank_linear:0.01:rank_lr1e2 \
  effective_rank_linear:0.03:rank_lr3e2
do
  IFS=: read -r method learning_rate label <<<"$setting"
  python3 "$project_root/scripts/run_gpt2_ppl_comparison.py" \
    --label "aba_a100_ppl_screen_${label}" \
    --methods "$method" \
    --maximum-updates 128 \
    --maximum-epochs 5 \
    --micro-batch-size 64 \
    --gradient-accumulation 1 \
    --workers 4 \
    --validation-batches 64 \
    --weight-decay 0 \
    --adamw-learning-rate "$learning_rate" \
    --muon-learning-rate "$learning_rate" \
    --muown-learning-rate "$learning_rate" \
    --effective-rank-learning-rate "$learning_rate"
done
