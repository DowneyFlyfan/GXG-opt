from __future__ import annotations

import math

import torch

from .linalg import finite, orthonormalize, proximal_coefficients, psd, solve_spd


def attention_rows(q, k, rows):
    logits = q[rows] @ k.T / math.sqrt(q.shape[-1])
    valid = torch.arange(len(k), device=k.device)[None, :] <= rows[:, None]
    return logits, torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=-1), valid


def omega_action(probabilities, values):
    return probabilities * (values - (probabilities * values).sum(-1, keepdim=True)) / len(values)


def finite_qk_defect(q, k, dq, dk):
    scale = math.sqrt(q.shape[-1])
    return (dq @ k.T + q @ dk.T) / scale, dq @ dk.T / scale


def correction_jacobian(x, qb, kb, rq, rk, rows):
    return ((x[rows] @ rq) @ kb.T + qb[rows] @ (x @ rk).T) / math.sqrt(qb.shape[-1])


def correction_adjoint(x, qb, kb, weighted, rows):
    scale = math.sqrt(qb.shape[-1])
    return x[rows].T @ (weighted @ kb) / scale, x.T @ (weighted.T @ qb[rows]) / scale


def qk_correction(fit, check, delta, *, rho=0.25, lambda_rel=0.01, nu_rel=1.0, replay_counter=None):
    x, q, k, rows = (fit[key] for key in ("x", "q", "k", "rows"))
    dq = x @ delta["wq"] + delta["bq"]
    dk = x @ delta["wk"] + delta["bk"]
    qb, kb = q + dq, k + dk
    linear, defect = finite_qk_defect(q[rows], k, dq[rows], dk)
    _, probabilities, _ = attention_rows(qb, kb, rows)
    adjq, adjk = correction_adjoint(x, qb, kb, omega_action(probabilities, defect), rows)
    zeros = torch.zeros_like(delta["wq"])
    pairs = [(delta["wq"], zeros), (zeros, delta["wk"]), (-adjq, zeros), (zeros, -adjk)]
    vectors = orthonormalize([torch.cat([a.flatten(), b.flatten()]) for a, b in pairs])
    diagnostic = {"basis_rank": len(vectors), "selected_gamma": 0.0}
    if not vectors or rho == 0:
        return zeros, zeros.clone(), diagnostic
    size = zeros.numel()
    basis = [(vector[:size].reshape_as(zeros).to(x.dtype),
              vector[size:].reshape_as(zeros).to(x.dtype)) for vector in vectors]
    images = [correction_jacobian(x, qb, kb, a, b, rows) for a, b in basis]
    gram = psd(torch.stack([
        torch.stack([(left * omega_action(probabilities, right)).sum() for right in images])
        for left in images
    ]))
    rhs = torch.stack([(image * omega_action(probabilities, defect)).sum() for image in images])
    trace = float(gram.trace())
    if trace <= 0:
        return zeros, zeros.clone(), diagnostic
    damping = lambda_rel * trace / len(basis)
    coefficients = solve_spd(gram + damping * torch.eye(len(basis), device=x.device), -rhs)
    rq, rk = zeros.clone(), zeros.clone()
    for coefficient, (a, b) in zip(coefficients, basis):
        rq.add_(a, alpha=float(coefficient))
        rk.add_(b, alpha=float(coefficient))
    norm = torch.sqrt(rq.square().sum() + rk.square().sum())
    budget = rho * torch.sqrt(delta["wq"].square().sum() + delta["wk"].square().sum())
    cap = min(1.0, float(budget / norm)) if float(norm) else 1.0
    rq, rk = rq * cap, rk * cap

    def replay(probe, gamma):
        if replay_counter is not None:
            replay_counter()
        px, pr = probe["x"], probe["rows"]
        query = probe["q"] + px @ (delta["wq"] + gamma * rq) + delta["bq"]
        key = probe["k"] + px @ (delta["wk"] + gamma * rk) + delta["bk"]
        value = probe["v"] + px @ delta["wv"] + delta["bv"]
        _, probability, _ = attention_rows(query, key, pr)
        return probability @ value @ (probe["o"] + delta["wo"])

    def original(probe):
        if replay_counter is not None:
            replay_counter()
        _, probability, _ = attention_rows(probe["q"], probe["k"], probe["rows"])
        return probability @ probe["v"] @ probe["o"]

    fit_y = original(fit)
    fit_difference = replay(fit, 0) - fit_y
    epsilon = 1e-8 * max(float(fit_y.norm()), 1.0)
    nu = nu_rel * float(fit["error"][rows].norm()) / max(float(fit_difference.norm()), epsilon)
    check_y = original(check)
    scores = []
    for gamma in (0.0, 0.25, 0.5, 1.0):
        difference = replay(check, gamma) - check_y
        score = (check["error"][check["rows"]] * difference).sum() + 0.5 * nu * difference.square().sum()
        finite(score)
        scores.append(float(score))
    best = min(range(4), key=lambda index: (scores[index], index))
    accepted = scores[0] - scores[best] > 1e-10 + 1e-4 * abs(scores[0])
    gamma = (0.0, 0.25, 0.5, 1.0)[best] if accepted else 0.0
    diagnostic.update({
        "selected_gamma": gamma, "checking_surrogates": scores, "nu": nu,
        "lambda": damping, "uncapped_norm": float(norm), "cap_scale": cap,
        "defect_metric": float((defect * omega_action(probabilities, defect)).sum()),
        "linear_metric": float((linear * omega_action(probabilities, linear)).sum()),
        "solve_residual": float(((gram + damping * torch.eye(len(basis), device=x.device)) @ coefficients + rhs).norm()),
        "gram_eigenvalues": torch.linalg.eigvalsh(gram).tolist(),
    })
    return gamma * rq, gamma * rk, diagnostic


def edge_probability(p, j, k, mixture=0.05):
    n = len(p)
    return (1 - mixture) * (p[j] + p[k]) / (n - 1) + mixture * 2 / (n * (n - 1))


def sample_unordered_edges(p, count, generator, mixture=0.05):
    p = p.detach().cpu()
    n = len(p)
    if n < 2:
        return []
    pairs = []
    for _ in range(count):
        if float(torch.rand((), generator=generator)) < mixture:
            j = int(torch.randint(n, (), generator=generator))
        else:
            j = int(torch.multinomial(p, 1, generator=generator))
        k = int(torch.randint(n - 1, (), generator=generator))
        k += k >= j
        pairs.append((min(j, k), max(j, k)))
    return pairs


def edge_factors(x, q, k, rows, *, edges_per_row=4, mixture=0.05, generator):
    _, probabilities, _ = attention_rows(q, k, rows)
    factors = []
    sample_probabilities = []
    for row_index, row in enumerate(rows.tolist()):
        if row < 1:
            continue
        p = probabilities[row_index, :row + 1]
        for j, other in sample_unordered_edges(p, edges_per_row, generator, mixture):
            probability = edge_probability(p, j, other, mixture)
            weight = torch.sqrt(p[j] * p[other] / (len(rows) * edges_per_row * probability * q.shape[-1]))
            factors.append((x[row], k[j] - k[other], x[j] - x[other], q[row], weight))
            sample_probabilities.append(float(probability))
    if not factors:
        raise ValueError("routing probe needs at least one query with two valid keys")
    return tuple(torch.stack([factor[index] for factor in factors]) for index in range(5)), sample_probabilities


def weighted_factor_gram(factors):
    x, kd, xd, q, weight = factors
    return ((x @ x.T) * (kd @ kd.T) + (xd @ xd.T) * (q @ q.T)) * weight[:, None] * weight[None, :]


def factor_projection(factors, dq, dk):
    x, kd, xd, q, weight = factors
    return (((x @ dq) * kd).sum(-1) + ((xd @ dk) * q).sum(-1)) * weight


def factor_combination(factors, coefficients):
    x, kd, xd, q, weight = factors
    weights = weight * coefficients.to(weight.dtype)
    return (x * weights[:, None]).T @ kd, (xd * weights[:, None]).T @ q


def filter_routing_increment(factors, dq, dk, rho=1.0):
    gram = weighted_factor_gram(factors)
    projected = factor_projection(factors, dq, dk)
    coefficients, kappa = proximal_coefficients(gram, projected, rho)
    cq, ck = factor_combination(factors, coefficients)
    fq, fk = dq - cq, dk - ck
    after = factor_projection(factors, fq, fk)
    return fq, fk, {
        "kappa": kappa, "gram_eigenvalues": torch.linalg.eigvalsh(psd(gram)).tolist(),
        "routing_quadratic_before": float(projected.square().sum()),
        "routing_quadratic_after": float(after.square().sum()),
        "update_norm_before": float(torch.sqrt(dq.square().sum() + dk.square().sum())),
        "update_norm_after": float(torch.sqrt(fq.square().sum() + fk.square().sum())),
    }
