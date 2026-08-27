import json
from dataclasses import replace
from pathlib import Path

import pytest
from torch import nn

from gn_guided_adam.blocks import BlockRegistry
from gn_guided_adam.config import (
    AdaptiveConfig,
    FixedEpochDutyCycleConfig,
    GNConfig,
    GuidedAdamConfig,
)


def test_all_checked_in_configs_load_and_adaptive_remains_disabled():
    root = Path(__file__).resolve().parents[1] / "src/gn_guided_adam/configs"

    configs = {path.name: GuidedAdamConfig.from_yaml(path) for path in root.glob("*.yaml")}

    assert set(configs) == {
        "adamw_baseline.yaml",
        "gn_oracle_probe.yaml",
        "gn_guided_fixed.yaml",
        "gn_guided_adaptive.yaml",
    }
    assert not configs["adamw_baseline.yaml"].gn.enabled
    assert configs["gn_guided_fixed.yaml"].gn.enabled
    assert configs["gn_guided_fixed.yaml"].fixed_epoch_duty_cycle.enabled
    assert configs["gn_guided_fixed.yaml"].fixed_epoch_duty_cycle.on_epochs == 1
    assert configs["gn_guided_fixed.yaml"].fixed_epoch_duty_cycle.off_epochs == 1
    assert all(not config.adaptive.enabled for config in configs.values())


def test_json_round_trip_is_strict(tmp_path):
    config = GuidedAdamConfig()
    path = tmp_path / "guided.json"
    path.write_text(json.dumps(config.to_dict(wrapped=True)), encoding="utf-8")

    assert GuidedAdamConfig.from_json(path) == config
    with pytest.raises(ValueError, match="Unknown optimizer keys"):
        GuidedAdamConfig.from_dict({"future_controller": {}})


def test_adaptive_refresh_is_explicitly_deferred():
    with pytest.raises(ValueError, match="deferred"):
        GuidedAdamConfig(adaptive=AdaptiveConfig(enabled=True))


@pytest.mark.parametrize(
    "config,match",
    [
        (lambda: GuidedAdamConfig(gn=replace(GNConfig(), rank=0)), "rank"),
        (lambda: GuidedAdamConfig(gn=replace(GNConfig(), momentum_subspace_decay=1.1)), "subspace"),
        (lambda: GuidedAdamConfig(gn=replace(GNConfig(), min_damping=1.0)), "min <= initial"),
        (lambda: GuidedAdamConfig(gn=replace(GNConfig(), curvature_batches=2)), "exactly one"),
        (
            lambda: GuidedAdamConfig(
                fixed_epoch_duty_cycle=FixedEpochDutyCycleConfig(on_epochs=0)
            ),
            "on_epochs",
        ),
        (lambda: GuidedAdamConfig.from_dict({"adamw": {"lr": "fast"}}), "numeric"),
    ],
)
def test_invalid_configurations_fail_before_optimizer_creation(config, match):
    with pytest.raises(ValueError, match=match):
        config()


class RegistryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.projection = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)

    def forward(self, value):
        return self.head(self.norm(self.projection(self.embedding(value))))


def test_registry_uses_one_weight_tensor_blocks_and_initial_exclusions():
    registry = BlockRegistry(RegistryModel(), replace(GNConfig(), min_block_numel=1))

    enabled = registry.enabled
    assert [spec.name for spec in enabled] == ["projection.weight"]
    assert enabled[0].numel == 16
    assert enabled[0].module_path == "projection"
    assert len(enabled[0].parameters) == 1
    reasons = {spec.name: spec.disabled_reason for spec in registry.specs}
    assert reasons["embedding.weight"] == "embedding_or_normalization"
    assert reasons["projection.bias"] == "vector_or_bias"
    assert reasons["norm.weight"] == "vector_or_bias"
    assert reasons["head.weight"] == "output_projection"


def test_registry_disables_tied_parameters_and_honors_fixed_patterns():
    model = nn.Module()
    model.first = nn.Linear(4, 4, bias=False)
    model.second = nn.Linear(4, 4, bias=False)
    model.second.weight = model.first.weight

    tied = BlockRegistry(model, replace(GNConfig(), min_block_numel=1))
    assert not tied.specs[0].enabled
    assert tied.specs[0].disabled_reason == "tied_or_shared"

    untied = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    selected = BlockRegistry(
        untied,
        replace(GNConfig(), min_block_numel=1, guided_block_patterns=(r"^1\.",)),
    )
    assert [spec.name for spec in selected.enabled] == ["1.weight"]
