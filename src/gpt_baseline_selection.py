"""The selected GPT-12x512 baselines for result figures."""

from __future__ import annotations

from pathlib import Path


TASK_IDENTIFIER = "nlp_gpt_12x512"
SELECTED_GPT_BASELINES = {
    "adamw": ("AdamW (1.5e-4)", "literature_adamw00015_full_b12_a4"),
    "muon": ("Muon (2.5e-3)", "literature_mu0025_adamw0005_b12_a4"),
    "muown": ("Muown (5e-3, wd=0)", "lr0005_b8_a6_final"),
}


def selected_baseline_paths(root: Path, optimizer: str) -> tuple[Path, Path]:
    """Return tuned-winner files, with canonical files as a test fallback."""
    _, run_label = SELECTED_GPT_BASELINES[optimizer]
    stem = f"{TASK_IDENTIFIER}__{run_label}__{optimizer}"
    selected = (
        root / "metrics" / "nlp" / f"{stem}.jsonl",
        root / "results" / "nlp" / f"{stem}.json",
    )
    if all(path.exists() for path in selected):
        return selected
    canonical_stem = f"{TASK_IDENTIFIER}__{optimizer}"
    return (
        root / "metrics" / "nlp" / f"{canonical_stem}.jsonl",
        root / "results" / "nlp" / f"{canonical_stem}.json",
    )


def selected_baseline_label(optimizer: str) -> str:
    """Return the display label with the tuned learning rate."""
    return SELECTED_GPT_BASELINES[optimizer][0]
