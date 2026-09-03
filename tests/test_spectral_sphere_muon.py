import torch


def _spectral_norm(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(matrix.float())[0]


def test_article_direction_is_tangent_to_spectral_sphere():
    from spectral_sphere_muon import spectral_sphere_direction, spectral_tangent

    torch.manual_seed(31)
    weight = torch.randn(12, 8)
    weight /= _spectral_norm(weight)
    gradient = torch.randn_like(weight)
    theta, _, _ = spectral_tangent(weight, power_iterations=32)

    direction, _ = spectral_sphere_direction(
        gradient, theta, lambda_steps=10, ns_steps=10
    )

    assert abs(torch.sum(theta * direction).item()) < 3.0e-3


def test_optimizer_preserves_initial_spectral_norm_after_update():
    from spectral_sphere_muon import SpectralSphereMuon

    torch.manual_seed(37)
    parameter = torch.nn.Parameter(torch.randn(12, 8))
    initial_norm = _spectral_norm(parameter)
    optimizer = SpectralSphereMuon(
        [parameter], lr=0.001, power_iterations=32, lambda_steps=10, ns_steps=10
    )
    parameter.grad = torch.randn_like(parameter)

    optimizer.step()

    torch.testing.assert_close(_spectral_norm(parameter), initial_norm, rtol=3e-3, atol=3e-3)


def test_build_optimizers_routes_baseline_muon_matrices_to_spectral_sphere_muon():
    from models import DecoderTransformer
    from optimizers import build_optimizers, muon_parameter_names

    model = DecoderTransformer(width=16, heads=4, layers=2, vocabulary_size=64)
    optimizers = build_optimizers(
        model,
        "spectral_sphere_muon",
        lr=0.001,
        weight_decay=0.01,
        auxiliary_lr=0.0003,
    )
    selected = muon_parameter_names(model)
    selected_ids = {
        id(parameter)
        for parameter in optimizers["spectral_sphere_muon"].param_groups[0]["params"]
    }

    assert selected_ids == {
        id(parameter) for name, parameter in model.named_parameters() if name in selected
    }
    assert optimizers["adamw_aux"].param_groups[0]["lr"] == 0.0003


def test_spectral_sphere_artifacts_have_a_distinct_optimizer_prefix(tmp_path):
    from stiefel_muon_experiment import spectral_sphere_muon_paths

    paths = spectral_sphere_muon_paths(tmp_path, "probe")

    assert paths.metric == (
        tmp_path / "metrics/nlp/nlp_gpt_12x512__spectral_sphere_muon_probe.jsonl"
    )
    assert paths.result == (
        tmp_path / "results/nlp/nlp_gpt_12x512__spectral_sphere_muon_probe.json"
    )
