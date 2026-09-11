from __future__ import annotations

import json


def test_recorded_nll_traces_are_rendered_as_perplexity(tmp_path):
    from gpt2_v2_experiment import render_recorded_perplexity

    run = tmp_path / "runs" / "muon" / "seed_0"
    run.mkdir(parents=True)
    run.joinpath("metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "elapsed_wall_seconds": float(step),
                    "validation_nll": nll,
                }
            )
            for step, nll in ((0, 3.0), (10, 2.0))
        )
        + "\n"
    )
    run.joinpath("run_summary.json").write_text(json.dumps({"label": "Muon"}))

    outputs = render_recorded_perplexity(tmp_path)

    assert [path.name for path in outputs] == [
        "gpt2_wikitext103_validation_perplexity_steps.png",
        "gpt2_wikitext103_validation_perplexity_time.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
