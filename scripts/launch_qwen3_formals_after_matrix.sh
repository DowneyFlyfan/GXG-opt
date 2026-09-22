#!/usr/bin/env bash
set -euo pipefail

root=${1:?expected project root}
gpu0_uuid=GPU-d5c984dc-a988-2e9a-219f-df1577167aaf
result_dir="$root/results/nlp"
log_dir="$root/.cache/qwen3_0p6b/logs"
mkdir -p "$log_dir"

matrix_results=(
  "muon:scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.003_aux0.001"
  "muon:scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.003_aux0.003"
  "muown:scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.01_gain0.0003_aux0.001"
  "muown:scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.01_gain0.0003_aux0.003"
  "muown:scratch_16mep_matrix_tune16m_v3_s300_mb8a8_lr0.01_gain0.0003_aux0.01"
)

while true; do
  ready=1
  for entry in "${matrix_results[@]}"; do
    optimizer=${entry%%:*}
    label=${entry#*:}
    result="$result_dir/qwen3_0p6b__${label}__${optimizer}.ppl.json"
    if [[ ! -f "$result" ]] || ! python3 - "$result" "$optimizer" <<'PY'
import json
import sys
path, optimizer = sys.argv[1:]
payload = json.load(open(path))
ok = (
    payload.get("optimizer") == optimizer
    and payload.get("initialization") == "scratch"
    and payload.get("completed_updates") == 300
    and isinstance(payload.get("final_perplexity"), (int, float))
)
raise SystemExit(0 if ok else 1)
PY
    then
      ready=0
      break
    fi
  done
  [[ "$ready" == 1 ]] && break
  sleep 30
done

while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fqx "$gpu0_uuid"; do
  sleep 30
done

exec env CUDA_VISIBLE_DEVICES=0 "$root/.venv/bin/python" -u \
  "$root/scripts/run_qwen3_formals_after_tuning.py" "$root" \
  >> "$log_dir/formals_after_matrix_handoff.log" 2>&1
