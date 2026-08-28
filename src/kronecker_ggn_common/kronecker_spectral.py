from __future__ import annotations

import torch
from torch import Tensor

from .damping import joint_damped_eigenvalues


class KroneckerSpectralOperator:
    """Matrix-shaped actions for A ⊗ B + damping I without forming a Kronecker matrix."""

    def __init__(
        self,
        activation: Tensor,
        output: Tensor,
        damping: float,
        *,
        eigenvalue_floor: float = 1.0e-8,
        joint_eigenvalue_floor: float = 1.0e-12,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        self.damping = float(damping)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.joint_eigenvalue_floor = float(joint_eigenvalue_floor)
        self.compute_dtype = compute_dtype or (
            torch.float64 if activation.dtype == torch.float64 else torch.float32
        )
        self.update_factors(activation, output)

    def update_factors(self, activation: Tensor, output: Tensor) -> None:
        activation = activation.detach().to(dtype=self.compute_dtype)
        output = output.detach().to(device=activation.device, dtype=self.compute_dtype)
        if activation.ndim != 2 or activation.shape[0] != activation.shape[1]:
            raise ValueError("activation factor must be square")
        if output.ndim != 2 or output.shape[0] != output.shape[1]:
            raise ValueError("output factor must be square")
        if not torch.isfinite(activation).all() or not torch.isfinite(output).all():
            raise ValueError("spectral factors must be finite")
        activation = 0.5 * (activation + activation.T)
        output = 0.5 * (output + output.T)
        activation_values, self.activation_vectors = torch.linalg.eigh(activation)
        output_values, self.output_vectors = torch.linalg.eigh(output)
        self.activation_values = activation_values.clamp_min(self.eigenvalue_floor)
        self.output_values = output_values.clamp_min(self.eigenvalue_floor)
        self.joint_values = joint_damped_eigenvalues(
            self.activation_values,
            self.output_values,
            self.damping,
            self.joint_eigenvalue_floor,
        )
        self.activation = activation
        self.output = output

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return self.output.shape[0], self.activation.shape[0]

    def _check(self, value: Tensor) -> Tensor:
        if tuple(value.shape) != self.matrix_shape:
            raise ValueError(
                f"Expected matrix shape {self.matrix_shape}, got {tuple(value.shape)}"
            )
        return value.to(device=self.activation.device, dtype=self.compute_dtype)

    def _spectral_action(self, value: Tensor, power: float) -> Tensor:
        original = value
        matrix = self._check(value)
        transformed = self.output_vectors.T @ matrix @ self.activation_vectors
        result = (
            self.output_vectors
            @ (transformed * self.joint_values.pow(power))
            @ self.activation_vectors.T
        )
        return result.to(device=original.device, dtype=original.dtype)

    def matvec(self, value: Tensor) -> Tensor:
        matrix = self._check(value)
        result = self.output @ matrix @ self.activation.T + self.damping * matrix
        return result.to(device=value.device, dtype=value.dtype)

    def apply_inverse(self, value: Tensor) -> Tensor:
        return self._spectral_action(value, -1.0)

    def apply_inverse_sqrt(self, value: Tensor) -> Tensor:
        return self._spectral_action(value, -0.5)

    def apply_sqrt(self, value: Tensor) -> Tensor:
        return self._spectral_action(value, 0.5)

    def condition_number(self) -> float:
        return float((self.joint_values.max() / self.joint_values.min()).item())

    def state_dict(self) -> dict[str, Tensor | float]:
        return {
            "activation": self.activation,
            "output": self.output,
            "activation_values": self.activation_values,
            "activation_vectors": self.activation_vectors,
            "output_values": self.output_values,
            "output_vectors": self.output_vectors,
            "joint_values": self.joint_values,
            "damping": self.damping,
            "eigenvalue_floor": self.eigenvalue_floor,
            "joint_eigenvalue_floor": self.joint_eigenvalue_floor,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Tensor | float]
    ) -> KroneckerSpectralOperator:
        return cls(
            state["activation"],
            state["output"],
            float(state["damping"]),
            eigenvalue_floor=float(state["eigenvalue_floor"]),
            joint_eigenvalue_floor=float(state["joint_eigenvalue_floor"]),
            compute_dtype=state["activation"].dtype,
        )
