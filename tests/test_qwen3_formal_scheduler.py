import importlib.util
import json
from pathlib import Path


def _scheduler_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen3_formals_after_tuning.py"
    spec = importlib.util.spec_from_file_location("qwen3_formal_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_load_winners_selects_lowest_completed_perplexity_within_optimizer(tmp_path):
    scheduler = _scheduler_module()
    for optimizer, labels in scheduler.SCREEN_LABELS.items():
        for index, label in enumerate(labels):
            path = scheduler.result_path(tmp_path, optimizer, label)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "optimizer": optimizer,
                "initialization": "scratch",
                "completed_updates": 300,
                "final_perplexity": 10.0 - index,
                "run_label": label,
                "data_manifest_sha256": "matched",
                "config": {},
            }))

    winners = scheduler.load_winners(tmp_path)

    assert {optimizer: winner["run_label"] for optimizer, winner in winners.items()} == {
        optimizer: labels[-1] for optimizer, labels in scheduler.SCREEN_LABELS.items()
    }


def test_formal_command_locks_the_five_epoch_update_budget(tmp_path):
    scheduler = _scheduler_module()

    command = scheduler.formal_command(tmp_path, "adamw", {"config": {"learning_rate": 0.001}})

    assert command[command.index("--maximum-epochs") + 1] == "5"
    assert command[command.index("--maximum-updates") + 1] == "610"
    assert command[command.index("--micro-batch-size") + 1] == "8"
    assert command[command.index("--gradient-accumulation") + 1] == "8"


def test_formal_result_accepts_zero_based_epoch_field_when_five_epoch_budget_completed():
    scheduler = _scheduler_module()

    scheduler.validate_formal_result("adamw", {
        "optimizer": "adamw",
        "initialization": "scratch",
        "completed_epochs": 4,
        "completed_updates": 610,
        "config": {"maximum_epochs": 5, "maximum_updates": 610},
    })
