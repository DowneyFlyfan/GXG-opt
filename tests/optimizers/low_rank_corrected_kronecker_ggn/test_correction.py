import pytest
import torch

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator
from low_rank_corrected_kronecker_ggn.correction import corrected_direction


def operator():
    activation = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    output = torch.tensor([[1.5, 0.1], [0.1, 0.8]], dtype=torch.float64)
    return KroneckerSpectralOperator(activation, output, 0.3)


def test_corrected_inverse_recovers_dense_rank_two_system():
    kron = operator()
    basis = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    eigenvalues = torch.tensor([2.0, -0.5], dtype=torch.float64)
    gradient = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64)
    actual = corrected_direction(
        kron,
        gradient,
        basis,
        eigenvalues,
        eigenvalue_margin=0.01,
        absolute_eigenvalue_cap=100.0,
    ).direction
    identity = torch.eye(4, dtype=torch.float64)
    square_root = torch.stack(
        [
            kron.apply_sqrt(identity[:, index].reshape(2, 2)).reshape(-1)
            for index in range(4)
        ],
        dim=1,
    )
    flat_basis = basis.reshape(2, -1).T
    dense = (
        square_root
        @ (identity + flat_basis @ torch.diag(eigenvalues) @ flat_basis.T)
        @ square_root
    )

    expected = torch.linalg.solve(dense, -gradient.reshape(-1)).reshape(2, 2)

    assert torch.allclose(actual, expected, atol=1.0e-11)


def test_signed_eigenvalues_change_whitened_components_in_the_expected_direction():
    identity = torch.eye(2, dtype=torch.float64)
    kron = KroneckerSpectralOperator(
        identity, identity, 1.0e-12, joint_eigenvalue_floor=1.0e-15
    )
    basis = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    gradient = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    baseline = kron.apply_inverse(gradient).norm()
    positive = corrected_direction(
        kron,
        gradient,
        basis,
        torch.tensor([2.0]),
        eigenvalue_margin=0.1,
        absolute_eigenvalue_cap=None,
    )
    negative = corrected_direction(
        kron,
        gradient,
        basis,
        torch.tensor([-0.5]),
        eigenvalue_margin=0.1,
        absolute_eigenvalue_cap=None,
    )
    unsafe = corrected_direction(
        kron,
        gradient,
        basis,
        torch.tensor([-0.999]),
        eigenvalue_margin=0.1,
        absolute_eigenvalue_cap=None,
    )

    assert positive.direction.norm() < baseline
    assert negative.direction.norm() > baseline
    assert unsafe.clipped_eigenvalues.item() == pytest.approx(-0.9)
    assert unsafe.clipped_count == 1


def test_basis_sign_and_repeated_eigenspace_rotation_are_invariant():
    kron = operator()
    basis = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    gradient = torch.tensor([[0.2, -0.1], [0.5, 0.3]], dtype=torch.float64)
    eigenvalues = torch.tensor([0.7, 0.7], dtype=torch.float64)
    reference = corrected_direction(
        kron,
        gradient,
        basis,
        eigenvalues,
        eigenvalue_margin=0.1,
        absolute_eigenvalue_cap=None,
    ).direction
    signed = corrected_direction(
        kron,
        gradient,
        -basis,
        eigenvalues,
        eigenvalue_margin=0.1,
        absolute_eigenvalue_cap=None,
    ).direction
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
    rotated = (rotation @ basis.reshape(2, -1)).reshape_as(basis)
    rotated_direction = corrected_direction(
        kron,
        gradient,
        rotated,
        eigenvalues,
        eigenvalue_margin=0.1,
        absolute_eigenvalue_cap=None,
    ).direction

    assert torch.allclose(reference, signed, atol=1.0e-12)
    assert torch.allclose(reference, rotated_direction, atol=1.0e-12)
