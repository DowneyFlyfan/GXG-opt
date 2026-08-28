#!/usr/bin/env python3
"""Offline signed relative-residual spectrum diagnostic.

The optional input is a torch checkpoint containing ``layers[layer_id]`` entries
with dense ``activation``, ``output``, and ``ggn`` matrices plus ``damping``.
Without it, the script runs a deterministic mathematical reference diagnostic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator


def dense_action(operator, function):
    rows, columns = operator.matrix_shape
    identity = torch.eye(
        rows * columns,
        dtype=operator.activation.dtype,
        device=operator.activation.device,
    )
    return torch.stack(
        [
            function(identity[:, index].reshape(rows, columns)).reshape(-1)
            for index in range(rows * columns)
        ],
        dim=1,
    )


def synthetic_payload(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    rows = columns = 8
    left = torch.randn(columns, columns, generator=generator, dtype=torch.float64)
    right = torch.randn(rows, rows, generator=generator, dtype=torch.float64)
    activation = left.T @ left / columns + 0.2 * torch.eye(columns, dtype=torch.float64)
    output = right.T @ right / rows + 0.2 * torch.eye(rows, dtype=torch.float64)
    damping = 0.05
    kron = KroneckerSpectralOperator(activation, output, damping)
    square_root = dense_action(kron, kron.apply_sqrt)
    q, _ = torch.linalg.qr(
        torch.randn(rows * columns, 6, generator=generator, dtype=torch.float64)
    )
    mismatch = torch.tensor([1.4, -0.65, 0.35, -0.18, 0.08, -0.03], dtype=torch.float64)
    hessian = (
        square_root
        @ (
            torch.eye(rows * columns, dtype=torch.float64)
            + q @ torch.diag(mismatch) @ q.T
        )
        @ square_root
    )
    return {
        "layers": {
            "synthetic.linear": {
                "activation": activation,
                "output": output,
                "ggn": hessian
                - damping * torch.eye(rows * columns, dtype=torch.float64),
                "damping": damping,
            }
        },
        "curvature_mode": "exact_ggn",
        "source": "deterministic_synthetic_reference",
    }


def analyze_layer(layer: dict, ranks: list[int], seed: int) -> dict:
    activation = layer["activation"].double().cpu()
    output = layer["output"].double().cpu()
    ggn = layer["ggn"].double().cpu()
    damping = float(layer["damping"])
    operator = KroneckerSpectralOperator(
        activation, output, damping, compute_dtype=torch.float64
    )
    inverse_sqrt = dense_action(operator, operator.apply_inverse_sqrt)
    identity = torch.eye(ggn.shape[0], dtype=torch.float64)
    residual = inverse_sqrt @ (ggn + damping * identity) @ inverse_sqrt - identity
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (residual + residual.T))
    order = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    energy = eigenvalues.square()
    cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1.0e-30)
    generator = torch.Generator().manual_seed(seed)
    gradient = torch.randn(ggn.shape[0], generator=generator, dtype=torch.float64)
    exact = torch.linalg.solve(ggn + damping * identity, -gradient)
    baseline = torch.linalg.solve(dense_action(operator, operator.matvec), -gradient)
    rank_metrics = []
    whitened = inverse_sqrt @ gradient
    for rank in ranks:
        selected_values = eigenvalues[:rank]
        selected_vectors = eigenvectors[:, :rank]
        corrected_whitened = whitened - selected_vectors @ (
            selected_values / (1 + selected_values) * (selected_vectors.T @ whitened)
        )
        corrected = -(inverse_sqrt @ corrected_whitened)
        cosine = float(
            torch.dot(corrected, exact).item()
            / (corrected.norm() * exact.norm()).clamp_min(1.0e-30).item()
        )
        relative_error = float(
            (corrected - exact).norm().item() / exact.norm().clamp_min(1.0e-30).item()
        )
        rank_metrics.append(
            {
                "rank": rank,
                "oracle_direction_cosine": cosine,
                "relative_direction_error": relative_error,
            }
        )
    baseline_cosine = float(
        torch.dot(baseline, exact).item()
        / (baseline.norm() * exact.norm()).clamp_min(1.0e-30).item()
    )
    return {
        "eigenvalues": eigenvalues.tolist(),
        "absolute_eigenvalues": eigenvalues.abs().tolist(),
        "cumulative_energy": cumulative.tolist(),
        "baseline_oracle_direction_cosine": baseline_cosine,
        "rank_metrics": rank_metrics,
        "minimum_one_plus_eigenvalue": float((1 + eigenvalues).min().item()),
    }


def plot_report(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plot

    figure, axes = plot.subplots(1, 3, figsize=(14, 4))
    for layer_id, layer in report["layers"].items():
        indexes = list(range(1, len(layer["absolute_eigenvalues"]) + 1))
        axes[0].semilogy(
            indexes, layer["absolute_eigenvalues"], marker=".", label=layer_id
        )
        axes[1].plot(indexes, layer["cumulative_energy"], label=layer_id)
        axes[2].plot(
            [item["rank"] for item in layer["rank_metrics"]],
            [item["oracle_direction_cosine"] for item in layer["rank_metrics"]],
            marker="o",
            label=layer_id,
        )
    axes[0].set(
        xlabel="Signed-eigenpair rank", ylabel="Absolute relative-residual eigenvalue"
    )
    axes[1].set(
        xlabel="Rank", ylabel="Cumulative squared correction energy", ylim=(0, 1.02)
    )
    axes[2].set(
        xlabel="Correction rank",
        ylabel="Cosine to exact GN direction",
        ylim=(-1.02, 1.02),
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "kronecker_ggn"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranks", type=int, nargs="+", default=[0, 1, 2, 4, 8])
    args = parser.parse_args()
    payload = (
        torch.load(args.input, map_location="cpu", weights_only=True)
        if args.input
        else synthetic_payload(args.seed)
    )
    if payload.get("curvature_mode") not in {"exact_ggn", "mc_ggn", "empirical_fisher"}:
        raise ValueError("Input must explicitly identify curvature_mode")
    report = {
        "curvature_mode": payload["curvature_mode"],
        "source": payload.get("source", str(args.input)),
        "seed": args.seed,
        "layers": {
            layer_id: analyze_layer(layer, args.ranks, args.seed + index)
            for index, (layer_id, layer) in enumerate(payload["layers"].items())
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "residual_spectrum.json"
    png_path = args.output_dir / "residual_spectrum.png"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_report(report, png_path)
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
