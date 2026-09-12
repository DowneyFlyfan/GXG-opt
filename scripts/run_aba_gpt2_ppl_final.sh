#!/usr/bin/env bash
# Five formal-rate GPT2-12x512 PPL runs, with global batch 48.
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${ABA_PYTHON_BIN:-/home/yufan/New_Optimizer/.venv/bin/python}"
PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}" exec "$python_bin" "$project_root/scripts/run_gpt2_ppl_comparison.py" \
  --label aba_a100_ppl_formal_b8_a6_joint_newton \
  --methods effective_rank_linear_joint_newton muon muown adamw \
  --maximum-epochs 5 \
  --micro-batch-size 8 \
  --gradient-accumulation 6 \
  --workers 4 \
  --validation-batches 64
