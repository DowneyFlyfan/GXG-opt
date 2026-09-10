import torch


def _condition(matrix: torch.Tensor) -> float:
    values = torch.linalg.svdvals(matrix)
    return float(values[0] / values[-1])


def test_srip_direction_has_unit_spectral_norm_and_active_upper_feasibility():
    from srip_band import srip_band_direction

    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    gradient = torch.tensor([[0.2, -0.1, 0.3], [0.1, 0.3, -0.2], [0.3, 0.1, 0.1], [0.2, -0.1, 0.2]])

    phi, upper_violation, lower_violation = srip_band_direction(
        weight, gradient, rho=0.9, dual_steps=8
    )

    torch.testing.assert_close(torch.linalg.svdvals(phi)[0], torch.tensor(1.0), atol=2e-4, rtol=2e-4)
    assert upper_violation <= 2e-4
    assert lower_violation <= 2e-4


def test_srip_cayley_update_remains_in_the_normalized_spectral_band():
    from srip_band import srip_band_update, srip_condition_limit

    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    gradient = torch.randn_like(weight)

    updated, diagnostics = srip_band_update(weight, gradient, rho=0.9, step_size=1e-3)

    values = torch.linalg.svdvals(updated)
    assert torch.isfinite(updated).all()
    assert values[0] <= 1.0 + 3e-4
    assert _condition(updated) <= srip_condition_limit(0.9) * (1.0 + 3e-4)
    assert diagnostics.upper_boundary_violation <= 2e-4


def test_srip_interior_update_remains_feasible_without_forced_rescaling():
    from srip_band import srip_band_update, srip_condition_limit

    weight = torch.tensor([[0.9997, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    updated, _ = srip_band_update(weight, torch.randn_like(weight), rho=0.9, step_size=1e-3)

    values = torch.linalg.svdvals(updated)
    assert values[0] <= 1.0 + 3e-4
    assert _condition(updated) <= srip_condition_limit(0.9) * (1.0 + 3e-4)


def test_repeated_srip_updates_never_accept_an_out_of_band_state():
    from srip_band import srip_band_update, srip_condition_limit

    weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.7, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    for _ in range(10):
        weight, _ = srip_band_update(weight, torch.randn_like(weight), rho=0.9, step_size=1e-3)
        values = torch.linalg.svdvals(weight)
        assert values[0] <= 1.0 + 1e-4
        assert _condition(weight) <= srip_condition_limit(0.9) * (1.0 + 1e-4)


def test_srip_degenerate_boundary_rejects_instead_of_spectral_clipping():
    from srip_band import srip_band_update

    weight = torch.diag(torch.tensor([1.0, 1.0, 0.5]))
    updated, diagnostics = srip_band_update(weight, torch.randn_like(weight), rho=0.9, step_size=1e-3)

    torch.testing.assert_close(updated, weight)
    assert diagnostics.rejected_degenerate_update


def test_srip_builder_routes_ineligible_matrix_to_adamw():
    from optimizers import build_optimizers
    from srip_band import SRIPBand

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, bias=False)
            self.projection = torch.nn.Parameter(torch.diag(torch.tensor([1.0, 0.1, 0.01])))
            with torch.no_grad():
                self.linear.weight.copy_(torch.diag(torch.tensor([1.0, 0.7, 0.5])))

    model = Model()
    optimizers = build_optimizers(model, "srip_band", lr=1e-3, weight_decay=0.0, srip_rho=0.9)

    assert isinstance(optimizers["srip_band"], SRIPBand)
    assert optimizers["srip_band"].param_groups[0]["params"] == [model.linear.weight]
    assert model.projection in optimizers["adamw_aux"].param_groups[0]["params"]


def test_srip_artifact_paths_preserve_the_nlp_contract(tmp_path):
    from srip_band_experiment import srip_band_paths

    paths = srip_band_paths(tmp_path, "probe")

    assert paths.metric == tmp_path / "metrics/nlp/nlp_gpt_12x512__srip_band_probe.jsonl"
    assert paths.result == tmp_path / "results/nlp/nlp_gpt_12x512__srip_band_probe.json"
