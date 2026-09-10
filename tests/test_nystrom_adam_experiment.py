from pathlib import Path

import torch


def test_nystrom_adam_paths_follow_the_nlp_result_contract(tmp_path: Path):
    from nystrom_adam_experiment import nystrom_adam_paths

    paths = nystrom_adam_paths(tmp_path, "probe")

    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__nystrom_adam_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__nystrom_adam_probe.json"


def test_adamw_handoff_preserves_weights_but_does_not_invent_moments():
    from nystrom_adam_experiment import fresh_adamw_after_nystrom

    model = torch.nn.Linear(3, 2)
    expected = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    optimizer = fresh_adamw_after_nystrom(model, learning_rate=1.5e-4, weight_decay=0.01)

    assert optimizer.param_groups[0]["lr"] == 1.5e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert not optimizer.state
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
