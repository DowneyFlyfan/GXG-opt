"""Sequential, resumable scratch-only rate screen; all trials share one protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-directory', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--optimizers', nargs='+', choices=('adamw', 'muon', 'muown'), default=['adamw', 'muon', 'muown'])
    parser.add_argument('--micro-batch-size', type=int, default=12)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--activation-checkpointing', action='store_true')
    parser.add_argument('--train-tokens-per-epoch', type=int)
    parser.add_argument('--schedule-updates', type=int, default=2000)
    parser.add_argument(
        '--checkpoint-interval-updates',
        type=int,
        help='persist a screen checkpoint every N updates; defaults to evaluation cadence',
    )
    parser.add_argument('--run-label-prefix', default='scratch_2b')
    parser.add_argument('--candidates', type=Path, help='JSON list of optimizer, label, and rate arguments')
    parser.add_argument('--campaign', default='screen')
    args = parser.parse_args()
    if 96 % args.micro_batch_size:
        parser.error('micro-batch must divide the fixed 96 validation blocks')
    root = Path(__file__).resolve().parents[1]
    logs = root / '.cache/qwen3_0p6b/logs'
    logs.mkdir(parents=True, exist_ok=True)
    candidates = []
    for rate in (0.0003, 0.001, 0.003):
        candidates.append(('adamw', f'lr{rate:g}', ['--learning-rate', str(rate)]))
    for optimizer in ('muon', 'muown'):
        for rate in (0.003, 0.01, 0.03):
            rates = (['--learning-rate', str(rate)] if optimizer == 'muon' else
                     ['--direction-lr', str(rate), '--gain-lr', '0.0003'])
            candidates.append((optimizer, f'lr{rate:g}_aux0.0003', rates + ['--auxiliary-lr', '0.0003']))
    if args.candidates is not None:
        candidates = json.loads(args.candidates.read_text())
    summary_path = root / (f'records/2026-09-14_qwen3_scratch_{args.campaign}_' + '_'.join(args.optimizers) + '.json')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = []
    for optimizer, label, rates in candidates:
        if optimizer not in args.optimizers:
            continue
        run_label = f'{args.run_label_prefix}_{args.campaign}_s{args.updates}_mb{args.micro_batch_size}a{args.gradient_accumulation}_{label}'
        stem = f'qwen3_0p6b__{run_label}__{optimizer}'
        result_path = root / 'results/nlp' / f'{stem}.ppl.json'
        checkpoint = root / '.cache/qwen3_0p6b/checkpoints' / f'{stem}.checkpoint.pt'
        metric = root / 'metrics/nlp' / f'{stem}.ppl.jsonl'
        command = [sys.executable, '-u', str(root / 'src/run_qwen3_ppl.py'), 'run',
                   '--initialization', 'scratch', '--optimizer', optimizer, '--run-label', run_label,
                   '--data-directory', str(args.data_directory.resolve()),
                   '--micro-batch-size', str(args.micro_batch_size), '--gradient-accumulation', str(args.gradient_accumulation),
                   '--maximum-epochs', '5',
                   '--maximum-updates', str(args.updates), '--validation-batches', str(96 // args.micro_batch_size),
                   '--evaluation-interval-updates', '100', '--warmup-updates', '100',
                   '--schedule-updates', str(args.schedule_updates), '--gradient-clip', '1', '--weight-decay', '0.1',
                   '--seed', '1337', *rates]
        if args.checkpoint_interval_updates is not None:
            command.extend(['--checkpoint-interval-updates', str(args.checkpoint_interval_updates)])
        if args.train_tokens_per_epoch is not None:
            command.extend(['--train-tokens-per-epoch', str(args.train_tokens_per_epoch)])
        if args.activation_checkpointing:
            command.append('--activation-checkpointing')
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('initialization') != 'scratch' or result.get('completed_updates') != args.updates:
                raise RuntimeError(f'Incompatible existing result: {result_path}')
            code = 0
        else:
            if checkpoint.exists():
                command.append('--resume')
            elif metric.exists():
                raise RuntimeError(f'Partial trial without recoverable checkpoint: {metric}')
            print(json.dumps({'starting': run_label, 'optimizer': optimizer, 'command': command}), flush=True)
            with (logs / f'{stem}.log').open('a') as log:
                code = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT).returncode
            result = json.loads(result_path.read_text()) if code == 0 and result_path.exists() else None
        entry = {'optimizer': optimizer, 'run_label': run_label, 'command': command,
                 'exit_code': code, 'result': result}
        summary.append(entry)
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(entry), flush=True)
        # Screens retain their complete metrics/configurations; only final models need weights.
        # Delete only this script's successfully completed screen checkpoint to bound disk use.
        if code == 0 and result is not None and checkpoint.exists():
            checkpoint.unlink()
    if any(entry['exit_code'] != 0 for entry in summary):
        raise SystemExit('Some candidates failed; inspect retained logs before selection.')


if __name__ == '__main__':
    main()
