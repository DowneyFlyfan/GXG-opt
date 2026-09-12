"""Promote measured GPT2 perplexity figures and archive non-convertible accuracy PNGs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def promote_ppl_results(root: Path, *, label: str, execute: bool) -> dict[str, object]:
    """Archive legacy accuracy plots only after final measured PPL figures exist.

    Accuracy curves cannot be relabelled as perplexity because argmax accuracy
    does not determine token negative log likelihood.  This operation makes
    ``results/nlp`` an honest PPL-only result surface while retaining the
    historical files, unchanged, below ``.cache`` for provenance.
    """
    result_root = root / "results" / "nlp"
    measured = [
        result_root / f"gpt2_ppl_{label}_steps.png",
        result_root / f"gpt2_ppl_{label}_time.png",
        result_root / "gpt2_checkpoint_recovery_perplexity.png",
    ]
    missing = [path for path in measured if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing completed PPL figures: " + ", ".join(str(path) for path in missing)
        )
    protected = set(measured)
    legacy = sorted(path for path in result_root.glob("*.png") if path not in protected)
    archive_root = root / ".cache" / "nlp" / "archived_accuracy_figures"
    manifest = result_root / "gpt2_ppl_result_manifest.json"
    result: dict[str, object] = {
        "measured_ppl_figures": measured,
        "archived_accuracy_figures": legacy,
        "manifest": manifest,
    }
    if not execute:
        return result
    archive_root.mkdir(parents=True, exist_ok=True)
    for source in legacy:
        destination = archive_root / source.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite archived figure: {destination}")
        shutil.move(source, destination)
    manifest.write_text(
        json.dumps(
            {
                "measured_ppl_figures": [str(path.relative_to(root)) for path in measured],
                "archived_accuracy_figures": [str((archive_root / path.name).relative_to(root)) for path in legacy],
                "reason": "accuracy traces lack pointwise NLL and cannot be converted to perplexity",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return result
