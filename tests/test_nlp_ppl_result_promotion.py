from pathlib import Path

import pytest


def test_ppl_result_promotion_requires_final_measured_plots_and_archives_accuracy_images(tmp_path):
    from nlp_ppl_result_promotion import promote_ppl_results

    results = tmp_path / "results" / "nlp"
    results.mkdir(parents=True)
    label = "four_way_final"
    with pytest.raises(FileNotFoundError, match="missing completed PPL figures"):
        promote_ppl_results(tmp_path, label=label, execute=False)

    steps = results / f"gpt2_ppl_{label}_steps.png"
    time = results / f"gpt2_ppl_{label}_time.png"
    recovery = results / "gpt2_checkpoint_recovery_perplexity.png"
    historical = results / "historical_accuracy_metric_steps.png"
    for path in (steps, time, recovery, historical):
        path.write_bytes(b"png")

    preview = promote_ppl_results(tmp_path, label=label, execute=False)
    assert preview["archived_accuracy_figures"] == [historical]
    assert historical.exists()

    result = promote_ppl_results(tmp_path, label=label, execute=True)
    archived = tmp_path / ".cache" / "nlp" / "archived_accuracy_figures" / historical.name
    assert result["manifest"].is_file()
    assert archived.read_bytes() == b"png"
    assert not historical.exists()
    assert steps.exists() and time.exists() and recovery.exists()
