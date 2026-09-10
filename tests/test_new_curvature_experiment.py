import json
from pathlib import Path


def _write_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_new_curvature_plot_contains_selected_baselines_and_candidate(tmp_path: Path):
    from new_curvature_experiment import write_new_curvature_comparison_plots

    root = Path(tmp_path)
    rows = [
        {"epoch": 1, "metric": 0.2},
        {"epoch": 2, "metric": 0.3},
    ]
    _write_records(root / "metrics/nlp/nlp_gpt_12x512__adamw.jsonl", rows)
    _write_records(root / "metrics/nlp/nlp_gpt_12x512__muon.jsonl", rows)
    _write_records(root / "metrics/nlp/nlp_gpt_12x512__muown.jsonl", rows)
    (root / "results/nlp").mkdir(parents=True)
    (root / "results/nlp/nlp_gpt_12x512__adamw.json").write_text('{"seconds": 20, "epochs": 2}')
    (root / "results/nlp/nlp_gpt_12x512__muon.json").write_text('{"seconds": 25, "epochs": 2}')
    (root / "results/nlp/nlp_gpt_12x512__muown.json").write_text('{"seconds": 25, "epochs": 2}')
    _write_records(
        root / "metrics/nlp/nlp_gpt_12x512__racs_probe.jsonl",
        [{"step": 1, "metric": 0.25, "elapsed_seconds": 2.0}],
    )

    outputs = write_new_curvature_comparison_plots(
        root, candidate="racs", label="racs_probe", display_name="RACS"
    )

    assert outputs is not None
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in outputs)
