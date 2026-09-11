#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
base_python="${GPT2_PYTHON:-/home/justin/miniconda3/bin/python}"
experiment_config="configs/experiments/gpt2_v2_5090.yaml"
experiment_output="results/gpt2_v2_5090_seed0_5epochs"
setup_only=0
dry_run=0
extra_args=()
while (($#)); do
    case "$1" in
        --output) experiment_output="$2"; shift 2 ;;
        --config) experiment_config="$2"; shift 2 ;;
        --steps) extra_args+=(--steps "$2"); shift 2 ;;
        --setup-only) setup_only=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if ((dry_run)); then
    exec "$base_python" scripts/run_gpt2_v2_comparison.py --config "$experiment_config" --output "$experiment_output" "${extra_args[@]}" --dry-run
fi
"$base_python" - <<'PY'
import torch, transformers
if torch.__version__ != '2.11.0+cu130' or transformers.__version__ != '5.7.0':
    raise SystemExit('Expected the pinned torch 2.11.0+cu130 / transformers 5.7.0 environment.')
if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'NVIDIA GeForce RTX 5090':
    raise SystemExit('The local RTX 5090 is required.')
PY
experiment_python="$project_root/.venv-gpt2-v2/bin/python"
if [[ ! -x "$experiment_python" ]]; then
    "$base_python" -m venv --system-site-packages "$project_root/.venv-gpt2-v2"
fi
if ! "$experiment_python" -c 'import datasets; assert datasets.__version__ == "5.0.1"' 2>/dev/null; then
    "$experiment_python" -m pip install 'datasets==5.0.1'
fi
if ! "$experiment_python" -c 'import matplotlib' 2>/dev/null; then
    "$experiment_python" -m pip install 'matplotlib==3.10.8'
fi
"$experiment_python" - "$experiment_config" <<'PY'
import json, sys
from pathlib import Path
import yaml
from scripts.prepare_gpt2_wikitext import prepare
config = yaml.safe_load(Path(sys.argv[1]).read_text())
dataset = Path(config['dataset']['local_path'])
root = dataset.parent
manifest = root / 'manifest.json'
if not manifest.exists() or not dataset.exists() or not (Path(config['model']['local_path']) / 'config.json').exists():
    if dataset.exists():
        raise SystemExit(f'Incomplete assets at {root}; preserve or move them before preparing again.')
    prepare(root, model_name=config['model']['name'], dataset_name=config['dataset']['name'],
            dataset_config=config['dataset']['config'], sequence_length=config['model']['sequence_length'],
            processes=4, model_init='random')
else:
    saved = json.loads(manifest.read_text())
    if saved.get('model_init') != 'random' or saved['sequence_length'] != config['model']['sequence_length']:
        raise SystemExit('Existing asset manifest does not match the experiment.')
PY
if ((setup_only)); then
    exit 0
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
exec "$experiment_python" -u scripts/run_gpt2_v2_comparison.py --config "$experiment_config" --output "$experiment_output" "${extra_args[@]}"
