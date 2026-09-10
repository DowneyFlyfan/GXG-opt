import torch

from kronecker_ggn_common.kronecker_spectral import KroneckerSpectralOperator
from low_rank_corrected_kronecker_ggn.eigensolver import signed_lanczos
from low_rank_corrected_kronecker_ggn.residual_operator import RelativeResidualOperator
from low_rank_corrected_kronecker_ggn.storage import plan_dense_correction


def test_signed_lanczos_returns_largest_magnitude_eigenpairs_deterministically():
    generator = torch.Generator().manual_seed(7)
    q, _ = torch.linalg.qr(torch.randn(6, 6, generator=generator, dtype=torch.float64))
    eigenvalues = torch.tensor([-3.0, 2.0, 1.0, 0.5, 0.1, 0.0], dtype=torch.float64)
    matrix = q @ torch.diag(eigenvalues) @ q.T

    def solve():
        return signed_lanczos(
            lambda value: (matrix @ value.reshape(-1)).reshape(2, 3),
            (2, 3),
            3,
            steps=6,
            oversampling=3,
            generator=torch.Generator().manual_seed(11),
            dtype=torch.float64,
            tolerance=1.0e-10,
        )

    first, second = solve(), solve()

    assert torch.allclose(
        first.eigenvalues,
        torch.tensor([-3.0, 2.0, 1.0], dtype=torch.float64),
        atol=1.0e-10,
    )
    assert first.residuals.max().item() < 1.0e-10
    assert torch.equal(first.eigenvalues, second.eigenvalues)
    assert torch.equal(first.basis, second.basis)


def test_relative_residual_matches_explicit_whitened_matrix():
    activation = torch.tensor([[2.0, 0.2], [0.2, 1.0]], dtype=torch.float64)
    output = torch.tensor([[1.5, 0.1], [0.1, 0.8]], dtype=torch.float64)
    kron = KroneckerSpectralOperator(activation, output, 0.3)
    mismatch = torch.tensor(
        [
            [0.4, 0.1, 0.0, 0.0],
            [0.1, -0.2, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    identity = torch.eye(4, dtype=torch.float64)
    square_root = torch.stack(
        [
            kron.apply_sqrt(identity[:, index].reshape(2, 2)).reshape(-1)
            for index in range(4)
        ],
        dim=1,
    )
    ggn = square_root @ (identity + mismatch) @ square_root - 0.3 * identity

    class DenseGGN:
        def matvec(self, _layer_id, value):
            return (ggn @ value.reshape(-1)).reshape(2, 2)

    residual = RelativeResidualOperator("layer", kron, DenseGGN())
    vector = torch.tensor([[0.2, -0.1], [0.4, 0.3]], dtype=torch.float64)

    assert torch.allclose(
        residual.matvec(vector).reshape(-1), mismatch @ vector.reshape(-1), atol=1.0e-11
    )


def test_memory_planner_reduces_rank_or_refuses_before_allocation():
    reduced = plan_dense_correction(
        (10, 10), 8, 8, torch.float32, remaining_budget_bytes=7_000
    )
    refused = plan_dense_correction(
        (100, 100), 4, 8, torch.float32, remaining_budget_bytes=1_024
    )

    assert 0 < reduced.allocated_rank < reduced.requested_rank
    assert reduced.state_bytes + reduced.workspace_bytes <= 7_000
    assert refused.allocated_rank == 0
    assert "budget" in refused.reason
