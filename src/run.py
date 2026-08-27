from __future__ import annotations

import argparse
from pathlib import Path

from config import FORMAL_TASKS
from training import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal AdamW and Muon baselines")
    parser.add_argument("--task", choices=[task.identifier for task in FORMAL_TASKS])
    parser.add_argument("--optimizer", choices=("adamw", "muon"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    tasks = [task for task in FORMAL_TASKS if args.task is None or task.identifier == args.task]
    optimizers = (args.optimizer,) if args.optimizer else ("adamw", "muon")
    for task in tasks:
        for optimizer in optimizers:
            run_trial(task, optimizer, root, args.workers)


if __name__ == "__main__":
    main()
