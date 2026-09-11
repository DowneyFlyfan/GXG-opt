from __future__ import annotations

import math

import torch

from .linalg import finite, solve_spd


def notch_coefficients(omega, radius=0.8):
    cosine = math.cos(omega)
    c = (1 - 2 * radius * cosine + radius ** 2) / (2 - 2 * cosine)
    return c, -2 * c * cosine, c, -2 * radius * cosine, radius ** 2


def prime_steady_state(direction, coefficients):
    b0, _, b2, _, a2 = coefficients
    return (1 - b0) * direction, (b2 - a2) * direction


def df2t_step(direction, first, second, coefficients):
    b0, b1, b2, a1, a2 = coefficients
    output = b0 * direction + first
    return output, b1 * direction - a1 * output + second, b2 * direction - a2 * output


def spectral_detector(history):
    length = len(history)
    window = torch.hann_window(length, periodic=False, dtype=history.dtype, device=history.device)
    spectrum = torch.fft.rfft((history - history.mean(0)) * window[:, None], dim=0)
    power = spectrum.abs().square().sum(-1)
    peak = int(torch.argmax(power[8:33])) + 8
    total = float(power[1:33].sum())
    band = float(power[max(1, peak - 1):min(33, peak + 2)].sum())
    # torch.median returns the lower middle value, the specified convention here.
    ratio = float(power[peak] / (power[8:33].median() + 1e-16))
    fraction = band / total if total > 0 else 0.0
    return {"peak": peak, "band_fraction": fraction, "median_ratio": ratio,
            "eligible": total > 1e-16 and fraction >= 0.45 and ratio >= 8}


def fixed_sketch(direction, seed, signs=None):
    count = direction.numel()
    if signs is None:
        generator = torch.Generator().manual_seed(seed)
        signs = (torch.randint(2, (8, count), dtype=torch.int8, generator=generator) * 2 - 1).to(direction.device)
    norm = direction.norm()
    projection = signs.to(direction.dtype) @ direction.flatten()
    return projection / (math.sqrt(count) * norm.clamp_min(1e-12)), signs


def guarded_filter_step(direction, weight, state, step, sketch, *, radius=0.8):
    state = dict(state)
    history = state.get("history", sketch.new_empty((0, 8)))
    state["history"] = torch.cat([history, sketch[None]])[-64:]
    diagnostic = {}
    change = False
    if step % 32 == 0 and len(state["history"]) == 64:
        diagnostic = spectral_detector(state["history"])
        eligible = diagnostic["eligible"]
        peak = diagnostic["peak"]
        previous = state.get("previous_peak", -100)
        state["eligible_count"] = state.get("eligible_count", 0) + 1 if eligible and abs(peak - previous) <= 1 else int(eligible)
        state["ineligible_count"] = 0 if eligible else state.get("ineligible_count", 0) + 1
        state["previous_peak"] = peak if eligible else -100
        persistent = state["eligible_count"] >= 2
        if state.get("active", False):
            if step - state["last_change"] >= 64:
                if state["ineligible_count"] >= 2:
                    state["active"] = False
                elif persistent and peak != state["peak"]:
                    change = True
        elif persistent and step >= 128 and step >= state.get("cooldown_until", 0):
            change = True
        if change:
            state.update(active=True, peak=peak, last_change=step,
                         coefficients=notch_coefficients(2 * math.pi * peak / 64, radius))
    if not state.get("active", False):
        state.pop("first", None)
        state.pop("second", None)
        return direction, state, {**diagnostic, "active": False, "guard_rejected": False}
    if change:
        state["first"], state["second"] = prime_steady_state(direction, state["coefficients"])
    output, first, second = df2t_step(direction, state["first"], state["second"], state["coefficients"])
    guard = 2 * float(direction.norm()) + 1e-12 * max(1, float(weight.norm()))
    rejected = not all(bool(torch.isfinite(value).all()) for value in (output, first, second)) or float(output.norm()) > guard
    if rejected:
        state.update(active=False, cooldown_until=step + 128)
        state.pop("first", None)
        state.pop("second", None)
        output = direction
    else:
        state.update(first=first, second=second)
    return output, state, {**diagnostic, "active": state["active"], "guard_rejected": rejected,
                           "peak": state.get("peak"), "norm_ratio": float(output.norm() / direction.norm().clamp_min(1e-12))}


def near_identity_map(source, target, ridge_rel=0.1, delta=0.1):
    covariance = source.T @ source
    trace = float(covariance.trace())
    identity = torch.eye(source.shape[-1], dtype=source.dtype, device=source.device)
    if trace == 0:
        return identity, identity, {"low_confidence": True, "lambda": 0.0}
    damping = ridge_rel * trace / source.shape[-1]
    raw = solve_spd(covariance + damping * identity, source.T @ target + damping * identity)
    difference = raw - identity
    u, singular, vh = torch.linalg.svd(difference, full_matrices=False)
    bounded = identity + (u * singular.clamp(max=delta)) @ vh
    return raw.to(source.dtype), bounded.to(source.dtype), {
        "low_confidence": False, "lambda": damping, "raw_departure": float(singular.max()),
    }


def fit_block_maps(source, target, block_size=32):
    maps = []
    for start in range(0, source.shape[-1], block_size):
        end = min(start + block_size, source.shape[-1])
        _, bounded, _ = near_identity_map(source[:, start:end], target[:, start:end])
        maps.append(bounded)
    return maps


def map_factors(factors, maps):
    pieces = []
    start = 0
    for matrix in maps:
        end = start + len(matrix)
        pieces.append(factors[:, start:end] @ matrix)
        start = end
    return torch.cat(pieces, dim=-1)


def remap_gradient(gradient, left_maps, right_maps):
    left = []
    start = 0
    for matrix in left_maps:
        end = start + len(matrix)
        left.append(matrix.T @ gradient[start:end])
        start = end
    intermediate = torch.cat(left, dim=0)
    right = []
    start = 0
    for matrix in right_maps:
        end = start + len(matrix)
        right.append(intermediate[:, start:end] @ matrix)
        start = end
    return torch.cat(right, dim=1)


def predictive_maps(old_fit, new_fit, old_check, new_check, block_size=32):
    left = fit_block_maps(old_fit[0], new_fit[0], block_size)
    right = fit_block_maps(old_fit[1], new_fit[1], block_size)
    old_gradient = old_check[0].T @ old_check[1]
    current = new_check[0].T @ new_check[1]
    predicted = remap_gradient(old_gradient, left, right)
    finite(current, predicted)
    raw_error = (current - old_gradient).square().sum()
    map_error = (current - predicted).square().sum()
    finite(raw_error, map_error)
    raw_error, map_error = float(raw_error), float(map_error)
    accepted = map_error <= 0.95 * raw_error and raw_error > 1e-12 * max(float(current.square().sum()), 1e-12)
    identity_left = [torch.eye(len(a), device=a.device, dtype=a.dtype) for a in left]
    identity_right = [torch.eye(b.shape[0], device=b.device, dtype=b.dtype) for b in right]
    activation = remap_gradient(old_gradient, left, identity_right)
    error_only = remap_gradient(old_gradient, identity_left, right)
    scalar = (old_gradient * current).sum() / old_gradient.square().sum().clamp_min(1e-30)
    diagnostic = {
        "accepted": accepted, "raw_error": raw_error, "map_error": map_error,
        "activation_only_error": float((current - activation).square().sum()),
        "error_only_error": float((current - error_only).square().sum()),
        "scalar_error": float((current - scalar * old_gradient).square().sum()),
        "decay_error": float((current - 0.95 * old_gradient).square().sum()),
    }
    for split, old, new in (("fit", old_fit, new_fit), ("check", old_check, new_check)):
        for factor, index, matrices in (("x", 0, left), ("error", 1, right)):
            residual = float((new[index] - map_factors(old[index], matrices)).norm())
            diagnostic[f"{split}_{factor}_residual"] = residual
            diagnostic[f"{split}_{factor}_relative_error"] = residual / max(float(new[index].norm()), 1e-30)
    return (left, right) if accepted else (identity_left, identity_right), diagnostic


def cohort_step(historical, fresh, gradient, beta, maps=None):
    historical = historical * beta
    fresh = fresh * beta + (1 - beta) * gradient
    if maps is not None:
        historical = remap_gradient(historical, *maps)
    return historical + fresh, historical, fresh
