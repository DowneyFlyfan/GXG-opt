import itertools
import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from optimizer_v2.attention import (
    attention_rows, correction_adjoint, correction_jacobian, edge_factors,
    edge_probability, factor_combination, factor_projection, filter_routing_increment,
    finite_qk_defect, omega_action, qk_correction, sample_unordered_edges, weighted_factor_gram,
)
from optimizer_v2.layernorm import (
    choose_shared_damping, response_operator, solve_equivalent_damped_block, solve_response_pair,
)
from optimizer_v2.linalg import low_rank_prox, orthonormalize
from optimizer_v2.probes import sample_output_covariance_probe, split_path_forward
from optimizer_v2.temporal import (
    cohort_step, df2t_step, fit_block_maps, guarded_filter_step, near_identity_map,
    notch_coefficients, predictive_maps, prime_steady_state, remap_gradient, spectral_detector,
)


@pytest.fixture(autouse=True)
def deterministic_math():
    torch.manual_seed(13)
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def rand(*shape):
    return torch.randn(*shape, dtype=torch.float64)


def test_qk_finite_identity_biases_jacobian_and_adjoint():
    x, wq, wk = rand(7, 5), rand(5, 3), rand(5, 3)
    q, k = x @ wq + rand(3), x @ wk + rand(3)
    dq, dk = x @ rand(5, 3) + rand(3), x @ rand(5, 3) + rand(3)
    linear, defect = finite_qk_defect(q, k, dq, dk)
    torch.testing.assert_close(((q + dq) @ (k + dk).T - q @ k.T) / math.sqrt(3), linear + defect, atol=1e-10, rtol=1e-10)
    rows = torch.tensor([1, 3, 6])
    rq, rk = rand(5, 3), rand(5, 3)
    weighted = rand(3, 7)
    tangent = correction_jacobian(x, q + dq, k + dk, rq, rk, rows)
    aq, ak = correction_adjoint(x, q + dq, k + dk, weighted, rows)
    torch.testing.assert_close((tangent * weighted).sum(), (rq * aq).sum() + (rk * ak).sum())
    def replay(scale):
        return ((q + dq + scale * x @ rq)[rows] @ (k + dk + scale * x @ rk).T) / math.sqrt(3)
    torch.testing.assert_close((replay(1e-5) - replay(-1e-5)) / 2e-5, tangent, atol=1e-8, rtol=1e-8)
    quadratic = (x[rows] @ rq) @ (x @ rk).T / math.sqrt(3)
    torch.testing.assert_close(replay(1) - replay(0), tangent + quadratic)
    assert float(quadratic.norm()) > 1


def test_attention_mask_shift_metric_and_basis_reorthogonalization():
    q, k = rand(5, 2), rand(5, 2)
    rows = torch.tensor([1, 4])
    _, p, valid = attention_rows(q, k, rows)
    assert torch.count_nonzero(p[~valid]) == 0
    torch.testing.assert_close(omega_action(p, torch.ones_like(p) * 3), torch.zeros_like(p), atol=1e-14, rtol=0)
    vectors = [rand(20), rand(20)]
    basis = orthonormalize([*vectors, vectors[0] + 2 * vectors[1], torch.zeros(20)])
    assert len(basis) == 2
    matrix = torch.stack(basis)
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(2, dtype=torch.float64), atol=1e-12, rtol=0)


def head_probe():
    x = rand(7, 4)
    return {"x": x, "q": x @ rand(4, 2), "k": x @ rand(4, 2),
            "v": x @ rand(4, 2), "o": rand(2, 4), "error": rand(7, 4) * 0.01,
            "rows": torch.tensor([1, 3, 5, 6])}


def test_qk_ridge_stationarity_cap_and_zero_defect():
    fit, check = head_probe(), head_probe()
    delta = {"wq": rand(4, 2) * 0.1, "wk": rand(4, 2) * 0.1, "wv": rand(4, 2) * 0.1,
             "wo": rand(2, 4) * 0.1, "bq": rand(2) * 0.1, "bk": rand(2) * 0.1, "bv": rand(2) * 0.1}
    rq, rk, diagnostic = qk_correction(fit, check, delta)
    assert diagnostic["basis_rank"] == 4
    assert diagnostic["solve_residual"] < 1e-10
    assert float(torch.sqrt(rq.square().sum() + rk.square().sum())) <= 0.25 * float(torch.sqrt(delta["wq"].square().sum() + delta["wk"].square().sum())) + 1e-12
    if diagnostic["selected_gamma"]:
        assert min(diagnostic["checking_surrogates"]) < diagnostic["checking_surrogates"][0]
    delta["wk"].zero_()
    delta["bk"].zero_()
    rq, rk, diagnostic = qk_correction(fit, check, delta)
    assert torch.count_nonzero(rq) == torch.count_nonzero(rk) == 0


def test_qk_acceptance_rejects_cancelling_a_helpful_interaction():
    probe = head_probe()
    probe["q"].zero_()
    probe["k"].zero_()
    delta = {"wq": rand(4, 2) * 0.2, "wk": rand(4, 2) * 0.2,
             "wv": torch.zeros(4, 2, dtype=torch.float64), "wo": torch.zeros(2, 4, dtype=torch.float64),
             "bq": torch.zeros(2, dtype=torch.float64), "bk": torch.zeros(2, dtype=torch.float64),
             "bv": torch.zeros(2, dtype=torch.float64)}
    _, original, _ = attention_rows(probe["q"], probe["k"], probe["rows"])
    _, baseline, _ = attention_rows(probe["x"] @ delta["wq"], probe["x"] @ delta["wk"], probe["rows"])
    improvement = (baseline - original) @ probe["v"] @ probe["o"]
    probe["error"][probe["rows"]] = -improvement
    rq, rk, diagnostic = qk_correction(probe, probe, delta, nu_rel=0)
    assert diagnostic["checking_surrogates"][0] < 0
    assert diagnostic["selected_gamma"] == 0
    assert torch.count_nonzero(rq) == torch.count_nonzero(rk) == 0


def test_resistance_probabilities_exact_expectation_and_sampler():
    p = torch.tensor([0.02, 0.17, 0.81], dtype=torch.float64)
    laplacian = torch.diag(p) - p[:, None] * p[None, :]
    explicit = torch.zeros_like(laplacian)
    expectation = torch.zeros_like(laplacian)
    probabilities = []
    pairs = list(itertools.combinations(range(3), 2))
    for j, k in pairs:
        edge = torch.eye(3, dtype=torch.float64)[j] - torch.eye(3, dtype=torch.float64)[k]
        weight = p[j] * p[k]
        probability = edge_probability(p, j, k)
        probabilities.append(float(probability))
        explicit += weight * torch.outer(edge, edge)
        expectation += probability * weight / probability * torch.outer(edge, edge)
        torch.testing.assert_close(edge @ torch.linalg.pinv(laplacian) @ edge, 1 / p[j] + 1 / p[k])
    torch.testing.assert_close(explicit, laplacian)
    torch.testing.assert_close(expectation, laplacian)
    assert sum(probabilities) == pytest.approx(1)
    samples = sample_unordered_edges(p, 6000, torch.Generator().manual_seed(92))
    for pair, probability in zip(pairs, probabilities):
        assert samples.count(pair) / len(samples) == pytest.approx(probability, abs=0.025)
    assert len(samples) > len(set(samples))
    assert sample_unordered_edges(torch.ones(1), 4, torch.Generator()) == []


def test_routing_rank_one_gradients_gram_and_dense_inverse():
    x, wq, wk = rand(6, 4), rand(4, 2), rand(4, 2)
    wq.requires_grad_()
    wk.requires_grad_()
    q, k = x @ wq + rand(2), x @ wk + rand(2)
    scalar = q[3] @ (k[0] - k[2]) / math.sqrt(2)
    gq, gk = torch.autograd.grad(scalar, (wq, wk))
    torch.testing.assert_close(gq, torch.outer(x[3], k[0] - k[2]) / math.sqrt(2))
    torch.testing.assert_close(gk, torch.outer(x[0] - x[2], q[3]) / math.sqrt(2))
    factors, _ = edge_factors(x, q.detach(), k.detach(), torch.tensor([1, 3, 5]), generator=torch.Generator().manual_seed(3))
    a, b, c, d, weight = factors
    columns = torch.stack([torch.cat([torch.outer(a[i], b[i]).flatten(), torch.outer(c[i], d[i]).flatten()]) * weight[i] for i in range(len(a))], 1)
    gram = weighted_factor_gram(factors)
    torch.testing.assert_close(gram, columns.T @ columns)
    dq, dk = rand(4, 2), rand(4, 2)
    initial = torch.cat([dq.flatten(), dk.flatten()])
    torch.testing.assert_close(factor_projection(factors, dq, dk), columns.T @ initial)
    fq, fk, diagnostic = filter_routing_increment(factors, dq, dk)
    actual = torch.cat([fq.flatten(), fk.flatten()])
    dense = torch.linalg.solve(torch.eye(16, dtype=torch.float64) + diagnostic["kappa"] * columns @ columns.T, initial)
    torch.testing.assert_close(actual, dense, atol=1e-10, rtol=1e-10)
    assert actual.norm() <= initial.norm()
    assert diagnostic["routing_quadratic_after"] <= diagnostic["routing_quadratic_before"]
    metric = columns @ columns.T
    separate = metric.clone()
    separate[:8, 8:] = separate[8:, :8] = 0
    assert float((metric - separate).norm()) > 0


def test_categorical_probe_covariance_and_independent_positions():
    p = torch.tensor([[0.2, 0.8], [0.6, 0.4]], dtype=torch.float64)
    mean = torch.zeros(4, dtype=torch.float64)
    covariance = torch.zeros(4, 4, dtype=torch.float64)
    for first, second in itertools.product(range(2), repeat=2):
        probability = p[0, first] * p[1, second]
        r = (-p.clone()).scatter_add_(1, torch.tensor([[first], [second]]), torch.ones(2, 1, dtype=torch.float64)) / math.sqrt(2)
        mean += probability * r.flatten()
        covariance += probability * torch.outer(r.flatten(), r.flatten())
    expected = torch.block_diag(*[(torch.diag(row) - row[:, None] * row[None, :]) / 2 for row in p])
    torch.testing.assert_close(mean, torch.zeros_like(mean), atol=1e-15, rtol=0)
    torch.testing.assert_close(covariance, expected)
    generator = torch.Generator().manual_seed(17)
    samples = torch.stack([sample_output_covariance_probe(p, generator).flatten() for _ in range(5000)])
    torch.testing.assert_close(samples.T @ samples / len(samples), expected, atol=0.01, rtol=0.05)


def test_tied_alias_split_gradient_jacobians_and_cross_metric():
    model = GPT2LMHeadModel(GPT2Config(vocab_size=5, n_embd=4, n_layer=1, n_head=1, n_positions=8,
                                      attn_implementation="eager", use_cache=False)).double().eval()
    ids = torch.tensor([[0, 1, 2]])
    embedding = model.transformer.wte.weight
    left, right = embedding.detach().clone().requires_grad_(), embedding.detach().clone().requires_grad_()
    split = split_path_forward(model, ids, left, right)
    tied = model(input_ids=ids).logits
    torch.testing.assert_close(split, tied, atol=0, rtol=0)
    probe = rand(*split.shape)
    u, v = torch.autograd.grad((split * probe).sum(), (left, right))
    gradient, = torch.autograd.grad((tied * probe).sum(), embedding)
    torch.testing.assert_close(gradient, u + v, atol=1e-12, rtol=1e-12)
    jin, jout = torch.autograd.functional.jacobian(lambda a, b: split_path_forward(model, ids, a, b)[:, :-1].flatten(), (left, right))
    jin, jout = jin.reshape(10, -1), jout.reshape(10, -1)
    p = split[:, :-1].detach().softmax(-1).reshape(2, 5)
    curvature = torch.block_diag(*[(torch.diag(row) - row[:, None] * row[None, :]) / 2 for row in p])
    joint = (jin + jout).T @ curvature @ (jin + jout)
    separate = jin.T @ curvature @ jin + jout.T @ curvature @ jout
    mixed = jin.T @ curvature @ jout + jout.T @ curvature @ jin
    torch.testing.assert_close(joint - separate, mixed)
    assert float(mixed.norm()) > 1e-8
    assert torch.linalg.eigvalsh(joint).min() > -1e-10
    assert torch.linalg.eigvalsh(mixed).min() < -1e-8
    assert model.lm_head.weight is model.transformer.wte.weight


def test_low_rank_embedding_woodbury_stationarity():
    columns = [rand(7, 3) for _ in range(2)]
    increment = rand(7, 3)
    filtered, diagnostic = low_rank_prox(increment, columns)
    z = torch.stack([column.flatten() for column in columns], 1)
    system = torch.eye(21, dtype=torch.float64) + diagnostic["kappa"] * z @ z.T
    torch.testing.assert_close(system @ filtered.flatten(), increment.flatten(), atol=1e-10, rtol=1e-10)
    assert filtered.norm() <= increment.norm()
    zero, _ = low_rank_prox(increment, [torch.zeros_like(increment)] * 2)
    torch.testing.assert_close(zero, increment, atol=0, rtol=0)


@pytest.mark.parametrize("omega", [math.pi / 4, math.pi / 2, math.pi])
def test_notch_transfer_function_constant_priming_and_difference_equation(omega):
    coefficients = notch_coefficients(omega)
    b0, b1, b2, a1, a2 = coefficients
    def transfer(frequency):
        z = complex(math.cos(frequency), -math.sin(frequency))
        return (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
    assert transfer(0) == pytest.approx(1)
    assert abs(transfer(omega)) < 1e-12
    direction = rand(3, 4)
    first, second = prime_steady_state(direction, coefficients)
    for _ in range(30):
        output, first, second = df2t_step(direction, first, second, coefficients)
        torch.testing.assert_close(output, direction)
    signal = [math.sin(0.3 * t) for t in range(1000)]
    first = second = torch.tensor(0.0, dtype=torch.float64)
    previous_x = [0.0, 0.0]
    previous_y = [0.0, 0.0]
    outputs = []
    for x in signal:
        expected = b0 * x + b1 * previous_x[0] + b2 * previous_x[1] - a1 * previous_y[0] - a2 * previous_y[1]
        actual, first, second = df2t_step(torch.tensor(x, dtype=torch.float64), first, second, coefficients)
        assert float(actual) == pytest.approx(expected, abs=1e-12)
        previous_x = [x, previous_x[0]]
        previous_y = [expected, previous_y[0]]
        outputs.append(float(actual))
    times = torch.arange(300, 1000, dtype=torch.float64)
    design = torch.stack([torch.sin(0.3 * times), torch.cos(0.3 * times)], 1)
    fitted = torch.linalg.lstsq(design, torch.tensor(outputs[300:], dtype=torch.float64)).solution
    assert float(fitted.norm()) == pytest.approx(abs(transfer(0.3)), rel=1e-10)


def test_notch_detection_dwell_and_guard_reset():
    state = {}
    direction = torch.ones(2, 3)
    for step in range(1, 161):
        sketch = torch.ones(8) * math.sin(2 * math.pi * 12 * step / 64)
        output, state, diagnostic = guarded_filter_step(direction, direction, state, step, sketch)
        torch.testing.assert_close(output, direction)
        if step < 128:
            assert not state.get("active", False)
    assert state["active"] and state["peak"] == 12
    assert state["last_change"] == 128
    state["first"] = torch.full_like(direction, float("inf"))
    output, state, diagnostic = guarded_filter_step(direction, direction, state, 161, torch.zeros(8))
    assert diagnostic["guard_rejected"] and not state["active"]
    assert "first" not in state and state["cooldown_until"] == 289
    torch.testing.assert_close(output, direction, atol=0, rtol=0)
    assert not spectral_detector(torch.zeros(64, 8))["eligible"]


@pytest.mark.parametrize("mode", ["active", "cooldown", "switched"])
def test_notch_checkpoint_active_cooldown_and_coefficient_switch(tmp_path, mode):
    direction = rand(3, 4)
    coefficients = notch_coefficients(math.pi / 2)
    first, second = prime_steady_state(direction, coefficients)
    state = {"active": mode != "cooldown", "peak": 16, "last_change": 128,
             "coefficients": coefficients, "first": first, "second": second,
             "history": rand(40, 8), "cooldown_until": 300 if mode == "cooldown" else 0}
    if mode == "switched":
        state["coefficients"] = notch_coefficients(math.pi / 3)
        state["first"], state["second"] = prime_steady_state(direction, state["coefficients"])
    path = tmp_path / "filter.pt"
    torch.save(state, path)
    loaded = torch.load(path, weights_only=True)
    output, expected, diagnostic = guarded_filter_step(direction, direction, state, 129, torch.zeros(8, dtype=torch.float64))
    resumed, actual, resumed_diagnostic = guarded_filter_step(direction, direction, loaded, 129, torch.zeros(8, dtype=torch.float64))
    torch.testing.assert_close(output, resumed, atol=0, rtol=0)
    for key in expected:
        if isinstance(expected[key], torch.Tensor):
            torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)
        else:
            assert expected[key] == actual[key]
    assert diagnostic == resumed_diagnostic


def test_feature_factorization_ridge_and_bounded_block_maps():
    x, error, left, right = rand(11, 5), rand(11, 3), rand(5, 5), rand(3, 3)
    torch.testing.assert_close((x @ left).T @ (error @ right), left.T @ (x.T @ error) @ right)
    target = x @ (torch.eye(5, dtype=torch.float64) + 0.5 * rand(5, 5))
    raw, bounded, diagnostic = near_identity_map(x, target)
    identity = torch.eye(5, dtype=torch.float64)
    torch.testing.assert_close((x.T @ x + diagnostic["lambda"] * identity) @ raw,
                              x.T @ target + diagnostic["lambda"] * identity)
    assert torch.linalg.svdvals(bounded - identity).max() <= 0.1 + 1e-12
    assert (raw - bounded).norm() > 0
    maps = fit_block_maps(x, target, block_size=2)
    assert [len(matrix) for matrix in maps] == [2, 2, 1]
    zero_raw, zero_bounded, _ = near_identity_map(torch.zeros_like(x), target)
    torch.testing.assert_close(zero_raw, identity)
    torch.testing.assert_close(zero_bounded, identity)


def test_feature_check_rejects_overfit_maps_and_correct_age_cohort():
    x, error = rand(20, 4), rand(20, 3)
    maps, diagnostic = predictive_maps((x, error), (x * 1.03, error * 1.03),
                                       (x, error), (x * 0.97, error * 0.97), block_size=2)
    assert not diagnostic["accepted"]
    previous = rand(4, 3)
    torch.testing.assert_close(remap_gradient(previous, *maps), previous)
    left = [torch.eye(4, dtype=torch.float64) + 0.1 * rand(4, 4)]
    right = [torch.eye(3, dtype=torch.float64) + 0.1 * rand(3, 3)]
    beta = 0.8
    historical, fresh = previous.clone(), torch.zeros_like(previous)
    gradients = [rand(4, 3) for _ in range(3)]
    for index, gradient in enumerate(gradients):
        momentum, historical, fresh = cohort_step(historical, fresh, gradient, beta, (left, right) if index == 2 else None)
    expected = left[0].T @ (beta ** 3 * previous) @ right[0]
    expected += sum((1 - beta) * beta ** (2 - i) * g for i, g in enumerate(gradients))
    torch.testing.assert_close(momentum, expected)
    naive = left[0].T @ (beta ** 3 * previous + fresh) @ right[0]
    assert (momentum - naive).norm() > 1e-3
    one, _, _ = cohort_step(previous, torch.zeros_like(previous), gradients[0], beta, (left, right))
    torch.testing.assert_close(one, beta * left[0].T @ previous @ right[0] + (1 - beta) * gradients[0])
    zero, _, _ = cohort_step(previous, fresh, gradients[0], 0, (left, right))
    torch.testing.assert_close(zero, gradients[0])


@pytest.mark.parametrize("tau", [0, 1e-9, 0.1, 5, float("inf")])
def test_layernorm_response_equivalent_block_limits_and_coupling(tau):
    random = rand(6, 6)
    matrix = random @ random.T + torch.eye(6, dtype=torch.float64)
    gradient = rand(6)
    x, y = solve_response_pair(matrix, gradient, 2, tau)
    oracle = solve_equivalent_damped_block(matrix, gradient, 2, tau)
    torch.testing.assert_close(torch.cat([x, y]), oracle, atol=1e-10, rtol=1e-9)
    c = matrix[2:, 2:]
    response = response_operator(c, tau)
    assert torch.linalg.eigvalsh(response).min() >= -1e-15
    assert torch.linalg.eigvalsh(matrix[:2, :2] - matrix[:2, 2:] @ response @ matrix[2:, :2]).min() > 0
    if tau == 0:
        assert torch.count_nonzero(y) == 0
    elif tau == float("inf"):
        torch.testing.assert_close(torch.cat([x, y]), torch.linalg.solve(matrix, -gradient))
    elif tau < 1e-6:
        torch.testing.assert_close(response / tau, torch.eye(4, dtype=torch.float64), atol=1e-7, rtol=1e-7)
    else:
        assert torch.linalg.eigvalsh(torch.linalg.inv(response) - c).min() > -1e-10


def test_layernorm_damping_retries_preserve_shared_solve_and_budgets():
    random = rand(5, 5)
    metric = random @ random.T
    gradient = rand(5)
    pairs, diagnostic = choose_shared_damping(metric, gradient, 2, 0.01, 0.01)
    assert pairs and diagnostic["damping_retries"] > 0
    matrix = metric + diagnostic["lambda"] * torch.eye(5, dtype=torch.float64)
    for horizon, (x, y) in zip(diagnostic["horizons"], pairs):
        assert x.norm() <= 0.01 and y.norm() <= 0.01
        oracle = solve_equivalent_damped_block(matrix, gradient, 2, horizon / diagnostic["c_max"])
        torch.testing.assert_close(torch.cat([x, y]), oracle)
