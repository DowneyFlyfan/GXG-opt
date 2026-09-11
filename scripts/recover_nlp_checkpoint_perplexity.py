#!/usr/bin/env python3
"""Recover endpoint PPL from saved GPT2-12x512 checkpoint weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifacts import plot
from nlp_checkpoint_perplexity import evaluate_checkpoint_perplexity


def _default_checkpoints(root: Path) -> list[Path]:
    paths = list((root / ".cache" / "nlp" / "checkpoints").glob("*.checkpoint.pt"))
    paths.extend((root / ".cache").glob("nlp_gpt_12x512__*.checkpoint.pt"))
    return sorted(paths)


def _output_path(root: Path, checkpoint: Path) -> Path:
    stem = checkpoint.name.removesuffix(".checkpoint.pt")
    return root / "results" / "nlp" / "checkpoint_perplexity" / f"{stem}.ppl.json"


def _write_endpoint_plot(root: Path, outputs: list[Path]) -> Path:
    records = [json.loads(output.read_text()) for output in outputs]
    labels = [Path(record["checkpoint"]).name.removesuffix(".checkpoint.pt") for record in records]
    figure, axis = plot.subplots(figsize=(12, max(5, 0.38 * len(records))))
    axis.barh(labels, [record["perplexity"] for record in records])
    axis.set(xlabel="Validation perplexity (lower is better)", title="Recovered GPT2-12x512 checkpoint endpoints")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    output = root / "results" / "nlp" / "gpt2_checkpoint_recovery_perplexity.png"
    figure.savefig(output, dpi=160)
    plot.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", type=Path, action="append")
    parser.add_argument("--validation-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    checkpoints = arguments.checkpoint or _default_checkpoints(root)
    if not checkpoints:
        raise RuntimeError("no checkpoint candidates were found")
    outputs = [
        evaluate_checkpoint_perplexity(
            root,
            checkpoint.resolve(),
            _output_path(root, checkpoint),
            validation_batches=arguments.validation_batches,
            batch_size=arguments.batch_size,
            workers=arguments.workers,
        )
        for checkpoint in checkpoints
    ]
    print(_write_endpoint_plot(root, outputs))


if __name__ == "__main__":
    main()
