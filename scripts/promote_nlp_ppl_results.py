#!/usr/bin/env python3
"""Promote final measured GPT2 PPL figures and archive legacy accuracy PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from nlp_ppl_result_promotion import promote_ppl_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="completed four-way PPL comparison label")
    parser.add_argument("--execute", action="store_true", help="perform the checked archival move")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = promote_ppl_results(root, label=arguments.label, execute=arguments.execute)
    action = "promoted" if arguments.execute else "previewed"
    print(f"{action}: {len(result['archived_accuracy_figures'])} legacy figures")


if __name__ == "__main__":
    main()
