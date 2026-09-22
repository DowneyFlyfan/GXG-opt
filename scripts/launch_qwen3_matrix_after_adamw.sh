#!/usr/bin/env bash
# Start matrix-baseline tuning only after all three AdamW screens completed.
set -euo pipefail

project_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
gpu0_uuid="GPU-d5c984dc-a988-2e9a-219f-df1577167aaf"
adam_results=(
  "${project_root}/results/nlp/qwen3_0p6b__scratch_16mep_adamw_tune16m_v1_s300_mb8a8_lr0.001__adamw.ppl.json"
  "${project_root}/results/nlp/qwen3_0p6b__scratch_16mep_adamw_tune16m_v1_s300_mb8a8_lr0.003__adamw.ppl.json"
  "${project_root}/results/nlp/qwen3_0p6b__scratch_16mep_adamw_tune16m_v1_s300_mb8a8_lr0.006__adamw.ppl.json"
)

completed_adamw_screens() {
  local result
  for result in "${adam_results[@]}"; do
    [[ -f "${result}" ]] || return 1
    "${project_root}/.venv/bin/python" - "${result}" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    payload = json.load(handle)
if not (
    payload.get("initialization") == "scratch"
    and payload.get("optimizer") == "adamw"
    and payload.get("completed_updates") == 300
):
    raise SystemExit(1)
PY
  done
}

until completed_adamw_screens; do
  sleep 30
done

until ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fq "${gpu0_uuid}"; do
  sleep 30
done

exec env CUDA_VISIBLE_DEVICES=0 "${project_root}/.venv/bin/python" -u \
  "${project_root}/scripts/run_qwen3_scratch_screen.py" \
  --data-directory "${project_root}/.cache/Fineweb_Edu_2B" \
  --optimizers muon muown \
  --updates 300 \
  --micro-batch-size 8 \
  --gradient-accumulation 8 \
  --train-tokens-per-epoch 16000000 \
  --schedule-updates 610 \
  --checkpoint-interval-updates 300 \
  --run-label-prefix scratch_16mep \
  --candidates "${project_root}/records/2026-09-15_qwen3_matrix_16m_epoch_candidates.json" \
  --campaign matrix_tune16m_v3
