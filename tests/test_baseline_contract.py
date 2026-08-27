import math
from pathlib import Path

import pytest
import torch
from torch import nn

import training
from config import FORMAL_TASKS
from models import create_audio_model, create_cv_model, create_nlp_model, parameter_count
from optimizers import Muon, build_optimizers, muon_parameter_names, qualify_ratio
from training import _evaluate, _loaders, _loss, _model, configure_reproducibility


def test_formal_matrix_has_three_active_tasks_and_six_baselines():
    assert len(FORMAL_TASKS) == 3
    assert sum(2 for _ in FORMAL_TASKS) == 6
    assert {task.domain for task in FORMAL_TASKS} == {"nlp", "cv", "audio"}
    assert {task.model for task in FORMAL_TASKS} == {"smollm2_135m", "dinov3_vitb16", "owsm_v3.1_base"}


def test_declared_models_are_within_parameter_range():
    models = (
        create_nlp_model("gpt_12x512"),
        create_nlp_model("gpt_12x768"),
        create_cv_model("vit_base_12x768"),
        create_audio_model("ast_base_12x768"),
    )
    for model in models:
        assert 10_000_000 <= parameter_count(model) <= 200_000_000


def test_smollm2_loads_offline_from_the_project_cache():
    model = create_nlp_model("smollm2_135m")

    assert 130_000_000 <= parameter_count(model) <= 140_000_000


def test_dinov3_vitb16_loads_offline_with_a_cifar100_classifier():
    model = create_cv_model("dinov3_vitb16")

    assert 85_000_000 <= parameter_count(model) <= 86_000_000
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 100)


def test_dinov3_routes_patch_projection_and_classifier_to_adamw():
    names = muon_parameter_names(create_cv_model("dinov3_vitb16"))

    assert "backbone.embeddings.patch_embeddings.projection.weight" not in names
    assert "classifier.weight" not in names
    assert "backbone.layer.0.attention.q_proj.weight" in names


def test_dinov3_freezes_the_patch_embedder_and_first_eight_transformer_blocks():
    model = create_cv_model("dinov3_vitb16")

    assert not any(parameter.requires_grad for parameter in model.backbone.embeddings.parameters())
    assert not any(parameter.requires_grad for layer in model.backbone.layer[:8] for parameter in layer.parameters())
    assert all(parameter.requires_grad for layer in model.backbone.layer[8:] for parameter in layer.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_dinov3_formal_task_uses_the_qualified_batch_and_learning_rates():
    task = next(item for item in FORMAL_TASKS if item.identifier == "cv_dinov3_vitb16")

    assert (task.estimated_epochs, task.micro_batch_size, task.gradient_accumulation) == (75, 8, 8)
    assert (task.adamw_lr, task.muon_lr, task.muon_aux_lr) == (1e-4, 3e-4, 1e-4)


def test_smollm2_task_uses_the_tokenizer_specific_loader(monkeypatch):
    task = next(task for task in FORMAL_TASKS if task.model == "smollm2_135m")
    expected = object()
    observed = {}

    def loader(root, batch_size, workers, seed):
        observed.update(root=root, batch_size=batch_size, workers=workers, seed=seed)
        return expected

    monkeypatch.setattr(training, "smollm2_wikitext_loaders", loader)

    assert training._loaders(task, Path("/tmp/project"), workers=3, seed=19) is expected
    assert observed == {"root": Path("/tmp/project"), "batch_size": 2, "workers": 3, "seed": 19}


def test_smollm2_logits_are_extracted_from_the_causal_language_model_output():
    model = create_nlp_model("smollm2_135m")

    logits = training._model_logits(model, torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert logits.shape == (1, 3, 49_152)


def test_cnn_selection_excludes_first_and_degenerate_convolutions():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.Conv2d(32, 32, 3),
        nn.Conv2d(32, 160, 1),
        nn.Conv2d(32, 8, 3),
    )

    names = muon_parameter_names(model)

    assert "0.weight" not in names
    assert "1.weight" in names
    assert "2.weight" not in names
    assert "3.weight" not in names


def test_owsm_parameter_routing_excludes_embeddings_heads_and_ineligible_convolutions():
    root = Path(__file__).resolve().parents[1]
    model = create_audio_model("owsm_v3.1_base", root)

    names = muon_parameter_names(model)

    assert "encoder.embed.conv.0.weight" not in names
    assert "encoder.encoders.0.cgmlp.csgu.conv.weight" not in names
    assert "decoder.embed.0.weight" not in names
    assert "decoder.output_layer.weight" not in names
    assert "ctc.ctc_lo.weight" not in names
    assert "encoder.embed.out.0.weight" in names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for OWSM adapter verification")
def test_owsm_loss_accepts_the_adapter_batch():
    root = Path(__file__).resolve().parents[1]
    task = next(task for task in FORMAL_TASKS if task.model == "owsm_v3.1_base")
    train, _ = _loaders(task, root, workers=0)
    model = _model(task).cuda()

    loss = _loss(task, model, next(iter(train)), torch.device("cuda"))

    assert loss.requires_grad
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for OWSM adapter verification")
def test_owsm_evaluation_reports_a_finite_character_error_rate():
    root = Path(__file__).resolve().parents[1]
    task = next(task for task in FORMAL_TASKS if task.model == "owsm_v3.1_base")
    _, validation = _loaders(task, root, workers=0)
    model = _model(task).cuda()

    metric = _evaluate(task, model, validation, torch.device("cuda"), maximum_batches=1)

    assert metric >= 0.0
    assert math.isfinite(metric)


def test_owsm_profile_batches_use_the_cached_speech_model_format():
    import profiling

    root = Path(__file__).resolve().parents[1]
    task = next(task for task in FORMAL_TASKS if task.model == "owsm_v3.1_base")

    batch = next(iter(profiling._profile_batches(task, root, workers=0)))

    assert "speech" in batch
    assert "text_ctc" in batch


def test_muon_scale_matches_reference_formula():
    assert Muon.scaled_lr(0.02, rows=768, columns=3072) == pytest.approx(0.02 * 0.2 * math.sqrt(3072))


def test_muon_auxiliary_adamw_uses_the_tuned_learning_rate():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(16, 16, bias=False)
            self.embedding = nn.Embedding(32, 16)

    optimizers = build_optimizers(Model(), "muon", lr=0.0025, weight_decay=0.01, auxiliary_lr=1e-4)

    assert optimizers["adamw_aux"].param_groups[0]["lr"] == pytest.approx(1e-4)


def test_training_configuration_enables_deterministic_algorithms():
    configure_reproducibility(17)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_qualification_enforces_twenty_percent_limit():
    assert qualify_ratio(0.1, 0.12).qualified
    assert not qualify_ratio(0.1, 0.1201).qualified


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compiled Muon verification")
def test_compiled_zero_power_is_bfloat16_and_finite():
    update = torch.randn(64, 32, device="cuda")

    output = Muon.orthogonalize(update, steps=5)

    assert output.shape == update.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
