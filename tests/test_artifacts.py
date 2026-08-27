from pathlib import Path

from artifacts import write_metric, write_metric_plot


def test_metric_plot_writes_png_with_epoch_metric_data(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    write_metric(metrics, {"epoch": 1, "metric": 0.3})
    write_metric(metrics, {"epoch": 2, "metric": 0.4})
    comparison = tmp_path / "comparison.jsonl"
    write_metric(comparison, {"epoch": 1, "metric": 0.35})
    write_metric(comparison, {"epoch": 2, "metric": 0.45})

    output = write_metric_plot(metrics, comparison, tmp_path / "curve.png", "validation accuracy")

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
