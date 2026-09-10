import json
from dataclasses import replace
from pathlib import Path

import pytest
from torch import nn

from gxg_opt_adam.config import BridgeConfig, DutyCycleConfig, FinalMileConfig, GNConfig, GXGConfig
from gxg_opt_adam.layer_partition import LayerPartitioner, muon_eligible_parameter_names


def test_example_yaml_is_strict_and_round_trips():
    path = Path(__file__).resolve().parents[1] / "src/gxg_opt_adam/configs/gxg_optimizer.example.yaml"

    config = GXGConfig.from_yaml(path)

    assert GXGConfig.from_dict(config.to_dict(wrapped=True)) == config
    assert config.name == "gxg_optimizer"
    assert config.duty_cycle == DutyCycleConfig(gn_epochs=1, adam_epochs=3)


def test_json_configuration_loader_round_trips(tmp_path):
    config = GXGConfig()
    path = tmp_path / "gxg.json"
    path.write_text(json.dumps(config.to_dict(wrapped=True)), encoding="utf-8")

    assert GXGConfig.from_json(path) == config


@pytest.mark.parametrize(
    "config,error",
    [
        (lambda: GXGConfig(final_mile=FinalMileConfig(enabled=True)), "metric_threshold"),
        (lambda: GXGConfig(final_mile=FinalMileConfig(eval_split="test")), "test split"),
        (lambda: GXGConfig(bridge=BridgeConfig(rho_start=1.1)), "rho"),
        (lambda: GXGConfig(gn=replace(GNConfig(), line_search_alphas=(1.0, 0.5))), "no-op"),
        (lambda: GXGConfig(gn=replace(GNConfig(), initial_damping=0.0)), "damping"),
        (
            lambda: GXGConfig(gn=replace(GNConfig(), inner_optimizer_matrix="cg")),
            "both matrix and vector",
        ),
        (lambda: GXGConfig(duty_cycle=DutyCycleConfig(gn_epochs=0)), "gn_epochs"),
    ],
)
def test_invalid_configuration_is_rejected(config, error):
    with pytest.raises(ValueError, match=error):
        config()


def test_unknown_configuration_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown optimizer keys"):
        GXGConfig.from_dict({"name": "gxg_optimizer", "adaptive_magic": True})


def test_malformed_scalar_types_fail_as_configuration_errors():
    with pytest.raises(ValueError, match="adam.lr must be numeric"):
        GXGConfig.from_dict({"adam": {"lr": "fast"}})


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)

    def forward(self, tokens):
        value = self.embedding(tokens)
        for block in self.blocks:
            value = block(value)
        return self.head(self.norm(value))


def test_automatic_partition_is_complete_and_nonoverlapping():
    model = TinyTransformer()

    groups = LayerPartitioner(model).automatic_transformer()

    names = [name for group in groups for name in group.parameter_names]
    assert set(names) == {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert len(names) == len(set(names))
    block_one = next(group for group in groups if group.name == "block:blocks.1")
    assert "norm.weight" in block_one.parameter_names
    assert "norm.bias" in block_one.parameter_names


def test_tied_weights_cannot_be_assigned_to_different_groups():
    model = TinyTransformer()
    model.head.weight = model.embedding.weight

    with pytest.raises(ValueError, match="Tied parameter aliases"):
        LayerPartitioner(model).automatic_transformer()


def test_explicit_partition_rejects_missing_and_overlapping_parameters():
    model = nn.Linear(2, 1)
    partitioner = LayerPartitioner(model)

    with pytest.raises(ValueError, match="missing"):
        partitioner.from_explicit({"only_weight": ("weight",)})
    with pytest.raises(ValueError, match="more than one"):
        partitioner.from_explicit({"first": ("weight", "bias"), "second": ("weight",)})


def test_regex_partition_rules_are_complete_and_detect_overlap():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    partitioner = LayerPartitioner(model)

    groups = partitioner.from_regex_rules({"first": (r"^0\.",), "second": (r"^1\.",)})
    assert {group.name for group in groups} == {"first", "second"}

    with pytest.raises(ValueError, match="overlapping"):
        partitioner.from_regex_rules({"all": (r"weight",), "first": (r"^0\.",), "bias": (r"bias",)})


def test_muon_cnn_rules_use_grouped_matrix_shape_and_skip_first_cnn():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.Conv2d(32, 32, 3),
        nn.Conv2d(32, 160, 1),
        nn.Conv2d(32, 32, 3, groups=32),
    )

    names = muon_eligible_parameter_names(model)

    assert "0.weight" not in names
    assert "1.weight" in names
    assert "2.weight" not in names
    assert "3.weight" not in names
