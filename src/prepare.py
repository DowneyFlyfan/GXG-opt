from __future__ import annotations

import argparse
from pathlib import Path

from config import FORMAL_TASKS
from training import _loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare formal benchmark data")
    parser.add_argument("--domain", choices=("nlp", "cv", "audio"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    task = next(task for task in FORMAL_TASKS if task.domain == args.domain)
    train, validation = _loaders(task, root, args.workers)
    print(f"domain={args.domain} train_batches={len(train)} validation_batches={len(validation)}", flush=True)


if __name__ == "__main__":
    main()
