from __future__ import annotations

import torch
from torch.func import jvp

from .linalg import finite, orthonormalize, psd, solve_spd
from .probes import diagnostic_mode, functional_logits, lm_loss


def response_operator(curvature, tau):
    eigenvalues, vectors = torch.linalg.eigh(curvature.double())
    if float(eigenvalues.min()) <= 0:
        raise ValueError("normalization curvature must be positive definite")
    if tau == float("inf"):
        response = 1 / eigenvalues
    else:
        response = -torch.expm1(-tau * eigenvalues) / eigenvalues
    return (vectors * response) @ vectors.T


def solve_response_pair(matrix, gradient, x_size, tau):
    a, b, c = matrix[:x_size, :x_size], matrix[:x_size, x_size:], matrix[x_size:, x_size:]
    gx, gy = gradient[:x_size], gradient[x_size:]
    response = response_operator(c, tau)
    x = solve_spd(a - b @ response @ b.T, -(gx - b @ response @ gy))
    y = -response @ (gy + b.T @ x)
    return x, y


def solve_equivalent_damped_block(matrix, gradient, x_size, tau):
    if tau == 0:
        return torch.cat([solve_spd(matrix[:x_size, :x_size], -gradient[:x_size]),
                          torch.zeros_like(gradient[x_size:])])
    result = matrix.clone()
    response = response_operator(matrix[x_size:, x_size:], tau)
    result[x_size:, x_size:] = solve_spd(response, torch.eye(len(response), device=response.device, dtype=response.dtype))
    return solve_spd(result, -gradient)


def choose_shared_damping(metric, gradient, x_size, budget_x, budget_y, horizons=(0, 0.25, 1, 4)):
    metric = psd(metric)
    damping = max(0.1 * float(metric.trace()) / len(metric), 1e-8)
    for retry in range(7):
        matrix = metric + damping * torch.eye(len(metric), device=metric.device)
        largest = float(torch.linalg.eigvalsh(matrix[x_size:, x_size:])[-1])
        candidates = [solve_response_pair(matrix, gradient, x_size, horizon / largest) for horizon in horizons]
        if all(float(x.norm()) <= budget_x and float(y.norm()) <= budget_y for x, y in candidates):
            return candidates, {"lambda": damping, "damping_retries": retry,
                                "c_max": largest, "horizons": list(horizons)}
        damping *= 10
    return [], {"lambda": damping / 10, "damping_retries": 6, "budget_rejected": True}


def make_local_basis(model, proposals, block):
    parameters = dict(model.named_parameters())
    root = f"transformer.h.{block}"
    x_names = [f"{root}.mlp.c_fc.weight", f"{root}.mlp.c_proj.weight"]
    y_names = [f"{root}.ln_2.weight", f"{root}.ln_2.bias"]
    basis = []
    for name in x_names:
        increment = proposals[name].learning
        norm = float(increment.norm())
        if norm:
            basis.append({name: increment / norm})
    x_size = len(basis)
    for name in y_names:
        parameter = parameters[name]
        for column in orthonormalize([torch.ones_like(parameter), parameter.grad.detach()]):
            basis.append({name: column.to(parameter.dtype)})
    budget_x = 0.25 * sum(float(proposals[name].learning.square().sum()) for name in x_names) ** 0.5
    budget_y = 0.25 * sum(float(proposals[name].learning.square().sum()) for name in y_names) ** 0.5
    return basis, x_size, budget_x, budget_y


def reduced_ggn(model, provisional, basis, ids, counter=None):
    """Full logit directional derivatives share one probability metric, including cross blocks."""
    names = sorted({name for column in basis for name in column})
    values = tuple(provisional[name] for name in names)
    def forward(*local):
        parameters = dict(provisional)
        parameters.update(zip(names, local))
        return functional_logits(model, parameters, ids)[:, :-1]
    with diagnostic_mode(model):
        if counter is not None:
            counter(forward=1)
        primal = forward(*values).detach()
        finite(primal)
        probability = primal.softmax(-1)
        columns = []
        for column in basis:
            tangent = tuple(column.get(name, torch.zeros_like(value)) for name, value in zip(names, values))
            if counter is not None:
                counter(forward=1, jvp=1)
            _, derivative = jvp(forward, values, tangent)
            columns.append(derivative.detach())
        targets = ids[:, 1:]
        residual = probability.clone()
        residual.scatter_add_(-1, targets[..., None], -torch.ones_like(targets[..., None], dtype=residual.dtype))
        count = targets.numel()
        gradient = torch.stack([(column * residual).sum() / count for column in columns]).double()
        actions = [probability * (column - (probability * column).sum(-1, keepdim=True)) / count for column in columns]
        metric = torch.stack([torch.stack([(left * right).sum() for right in actions]) for left in columns])
    return gradient, psd(metric)


def layernorm_correction(model, proposals, fit_ids, check_ids, block, counter=None):
    basis, x_size, budget_x, budget_y = make_local_basis(model, proposals, block)
    if x_size == 0 or x_size == len(basis) or budget_x == 0 or budget_y == 0:
        return {}, {"empty_basis_or_budget": True}, 0, 0
    provisional = {name: proposals[name].value if name in proposals else value.detach()
                   for name, value in model.named_parameters()}
    gradient, metric = reduced_ggn(model, provisional, basis, fit_ids,
                                   (lambda **calls: counter(fit_ids, **calls)) if counter is not None else None)
    candidates, diagnostic = choose_shared_damping(metric, gradient, x_size, budget_x, budget_y)
    diagnostic.update(gram_eigenvalues=torch.linalg.eigvalsh(metric).tolist(),
                      cross_block_norm=float(metric[:x_size, x_size:].norm()))
    if not candidates:
        return {}, diagnostic, len(basis), 0
    corrections, losses = [], []
    with diagnostic_mode(model), torch.no_grad():
        if counter is not None:
            counter(check_ids, forward=1)
        baseline_loss = lm_loss(functional_logits(model, provisional, check_ids), check_ids)
        finite(baseline_loss)
        baseline_loss = float(baseline_loss)
        for x, y in candidates:
            correction = {}
            for coefficient, column in zip(torch.cat([x, y]), basis):
                for name, direction in column.items():
                    correction[name] = correction.get(name, torch.zeros_like(direction)) + float(coefficient) * direction
            parameters = dict(provisional)
            parameters.update({name: provisional[name] + delta for name, delta in correction.items()})
            if counter is not None:
                counter(check_ids, forward=1)
            loss = lm_loss(functional_logits(model, parameters, check_ids), check_ids)
            finite(loss)
            losses.append(float(loss))
            corrections.append(correction)
    best = min(range(len(losses)), key=lambda index: (losses[index], index))
    accepted = baseline_loss - losses[best] > 1e-7 + 1e-6 * abs(baseline_loss)
    diagnostic.update(checking_baseline_loss=baseline_loss, checking_losses=losses,
                      selected_horizon=diagnostic["horizons"][best] if accepted else None,
                      accepted=accepted)
    return corrections[best] if accepted else {}, diagnostic, len(basis), len(losses) + 1
