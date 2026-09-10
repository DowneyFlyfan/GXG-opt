import json
from pathlib import Path

import torch


def test_stiefel_artifacts_stay_in_nlp_metrics_results_and_cache(tmp_path: Path):
    from stiefel_muon_experiment import stiefel_muon_paths

    paths = stiefel_muon_paths(tmp_path, "probe")

    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__stiefel_muon_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__stiefel_muon_probe.json"
    assert paths.checkpoint == tmp_path / ".cache/nlp/checkpoints/nlp_gpt_12x512__stiefel_muon_probe.checkpoint.pt"


def test_stiefel_task_can_change_microbatch_without_changing_effective_batch():
    from stiefel_muon_experiment import stiefel_task

    task = stiefel_task(micro_batch_size=8, gradient_accumulation=1)

    assert task.micro_batch_size == 8
    assert task.gradient_accumulation == 1
    assert task.model == "gpt_12x512"


def test_historical_baseline_steps_respect_gradient_accumulation(tmp_path: Path):
    from stiefel_muon_experiment import _historical_baseline_records

    metric = tmp_path / "metrics/nlp/nlp_gpt_12x512__muon.jsonl"
    result = tmp_path / "results/nlp/nlp_gpt_12x512__muon.json"
    metric.parent.mkdir(parents=True)
    result.parent.mkdir(parents=True)
    metric.write_text(json.dumps({"epoch": 1, "metric": 0.5}) + "\n")
    result.write_text(
        json.dumps({"gradient_accumulation": 2, "epochs": 5, "seconds": 100.0})
    )

    records = _historical_baseline_records(tmp_path, "muon")

    assert records == [
        {"epoch": 1, "metric": 0.5, "step": 6104, "elapsed_seconds": 20.0}
    ]


def test_completed_epoch_requires_metric_after_an_unlogged_last_update():
    from stiefel_muon_experiment import _needs_epoch_end_evaluation

    assert _needs_epoch_end_evaluation(steps=10, last_evaluated_steps=8)
    assert not _needs_epoch_end_evaluation(steps=10, last_evaluated_steps=10)


def test_checkpoint_extension_preserves_current_rate_and_extends_cosine_tail(tmp_path: Path):
    from stiefel_muon_experiment import _load_checkpoint, _save_checkpoint

    original_parameter = torch.nn.Parameter(torch.ones(1))
    original_optimizer = torch.optim.SGD([original_parameter], lr=0.003)
    original_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        original_optimizer, T_max=6
    )
    for _ in range(4):
        original_parameter.grad = torch.zeros_like(original_parameter)
        original_optimizer.step()
        original_scheduler.step()
    rate_before_save = original_optimizer.param_groups[0]["lr"]
    checkpoint = tmp_path / "resume.pt"
    _save_checkpoint(
        checkpoint,
        torch.nn.Linear(1, 1),
        {"optimizer": original_optimizer},
        {"optimizer": original_scheduler},
        epoch=5,
        batch=7,
        steps=8_710,
        elapsed_seconds=10_425.7,
    )

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    model = torch.nn.Linear(1, 1)
    _load_checkpoint(
        checkpoint,
        model,
        {"optimizer": optimizer},
        {"optimizer": scheduler},
        scheduler_t_max=8,
    )

    assert optimizer.param_groups[0]["lr"] == rate_before_save
    assert scheduler.T_max == 8
    assert scheduler.last_epoch == 4
