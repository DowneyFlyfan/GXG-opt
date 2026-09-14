"""Command-line entry point for the Qwen3 optimizer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_data import (
    DEFAULT_SEQUENCE_LENGTH,
    FINEWEB_EDU_CONFIG,
    FINEWEB_EDU_DATASET,
    prepare_qwen_fineweb_cache,
    stream_fineweb_edu_tokens,
)
from qwen3_model import qwen_checkpoint_path
from qwen3_ppl_experiment import QwenTrialConfig, render_qwen_comparison, run_qwen_trial


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-0.6B matched optimizer benchmark")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="stream and pack the FineWeb-Edu cache")
    prepare.add_argument("--train-tokens", type=int, default=2_000_000_000)
    prepare.add_argument("--validation-tokens", type=int, default=100_000_000)
    prepare.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    prepare.add_argument("--seed", type=int, default=1337)
    run = commands.add_parser("run", help="run one baseline screen or formal trial")
    run.add_argument("--optimizer", choices=("adamw", "muon", "muown"), required=True)
    run.add_argument("--run-label", required=True)
    run.add_argument("--learning-rate", type=float)
    run.add_argument("--direction-lr", type=float)
    run.add_argument("--gain-lr", type=float)
    run.add_argument("--auxiliary-lr", type=float, default=3.0e-4)
    run.add_argument("--weight-decay", type=float, default=0.1)
    run.add_argument("--micro-batch-size", type=int, default=1)
    run.add_argument("--gradient-accumulation", type=int, default=1)
    run.add_argument("--maximum-epochs", type=int, default=3)
    run.add_argument("--maximum-updates", type=int)
    run.add_argument("--validation-batches", type=int, default=1)
    run.add_argument("--workers", type=int, default=0)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--device", default="cuda")
    render = commands.add_parser("render", help="render the three baseline comparison curves")
    render.add_argument("--run-label", required=True)
    return parser.parse_args(arguments)


def _prepare(root: Path, arguments: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    checkpoint = qwen_checkpoint_path(root)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    cache = prepare_qwen_fineweb_cache(
        root,
        train_tokens=arguments.train_tokens,
        validation_tokens=arguments.validation_tokens,
        sequence_length=arguments.sequence_length,
        eos_token_id=int(tokenizer.eos_token_id),
        source=stream_fineweb_edu_tokens(
            root,
            tokenizer_path=checkpoint,
            selection_seed=arguments.seed,
        ),
        source_name=f"{FINEWEB_EDU_DATASET}/{FINEWEB_EDU_CONFIG}",
        source_revision="streaming-default",
        selection_seed=arguments.seed,
    )
    return {"manifest": str(cache.manifest_path), "written_tokens": cache.manifest["written_tokens"]}


def main(arguments: list[str] | None = None) -> None:
    parsed = parse_args(arguments)
    root = parsed.root.resolve()
    if parsed.command == "prepare":
        result = _prepare(root, parsed)
    elif parsed.command == "run":
        result = run_qwen_trial(
            QwenTrialConfig(
                root=root,
                optimizer=parsed.optimizer,
                run_label=parsed.run_label,
                learning_rate=parsed.learning_rate,
                direction_lr=parsed.direction_lr,
                gain_lr=parsed.gain_lr,
                auxiliary_lr=parsed.auxiliary_lr,
                weight_decay=parsed.weight_decay,
                micro_batch_size=parsed.micro_batch_size,
                gradient_accumulation=parsed.gradient_accumulation,
                maximum_epochs=parsed.maximum_epochs,
                maximum_updates=parsed.maximum_updates,
                validation_batches=parsed.validation_batches,
                workers=parsed.workers,
                seed=parsed.seed,
                device=parsed.device,
            )
        )
    else:
        result = {"outputs": [str(path) for path in render_qwen_comparison(root, run_label=parsed.run_label)]}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
