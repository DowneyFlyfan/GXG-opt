import json

from gn_guided_adam.reporting import (
    plot_metric_steps_comparison,
    plot_metric_time_comparison,
    plot_metric_tokens_comparison,
)


def test_reporting_writes_png_comparisons_for_time_steps_and_tokens(tmp_path):
    adam = tmp_path / "adam.jsonl"
    guided = tmp_path / "guided.jsonl"
    rows = [
        {"step": 1, "tokens": 10, "wall_time_seconds": 1.0, "validation_metric": 0.2},
        {"step": 2, "tokens": 20, "wall_time_seconds": 2.0, "validation_metric": 0.4},
    ]
    adam.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    guided.write_text("".join(json.dumps({**row, "validation_metric": row["validation_metric"] + 0.1}) + "\n" for row in rows), encoding="utf-8")

    outputs = (
        plot_metric_time_comparison(adam, guided, tmp_path / "time.png"),
        plot_metric_steps_comparison(adam, guided, tmp_path / "steps.png"),
        plot_metric_tokens_comparison(adam, guided, tmp_path / "tokens.png"),
    )

    assert all(path.is_file() and path.read_bytes().startswith(b"\x89PNG") for path in outputs)
