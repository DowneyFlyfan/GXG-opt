#!/usr/bin/env python3
"""Run the six optimizer 2.0 proposals with pinned AdamW and Muon baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gpt2_v2_experiment import (  # noqa: E402
    comparison_plan, resolved_config, run, validate_config,
)
from optimizer_v2 import METHODS  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/gpt2_v2_5090.yaml"


def load_config(path: Path) -> dict:
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("results/gpt2_v2_5090_seed0_5epochs"))
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = resolved_config(load_config(args.config), args.methods, args.seeds, args.steps)
    if args.dry_run:
        print(json.dumps(comparison_plan(config), indent=2))
        return
    raise SystemExit(run(config, args.output))


if __name__ == "__main__":
    main()
