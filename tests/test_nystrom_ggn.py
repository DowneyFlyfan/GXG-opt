import pytest
import torch


def test_nystrom_inverse_matches_damped_sampled_column_matrix():
    from nystrom_ggn import build_nystrom_state

    hessian = torch.diag(torch.tensor([2.0, 3.0, 5.0]))
    state = build_nystrom_state(
        lambda vector: hessian @ vector,
        dimension=3,
        indices=(0, 1),
        rank=2,
        damping=0.5,
    )

    direction = state.inverse_action(torch.ones(3))

    assert torch.allclose(direction, torch.tensor([1.0 / 2.5, 1.0 / 3.5, 2.0]))


def test_nystrom_rejects_nonpositive_sampled_intersection():
    from nystrom_ggn import build_nystrom_state

    with pytest.raises(ValueError, match="positive"):
        build_nystrom_state(
            lambda vector: torch.diag(torch.tensor([-1.0, 2.0])) @ vector,
            dimension=2,
            indices=(0,),
            rank=1,
            damping=0.1,
        )


def test_nystrom_retains_strictly_positive_small_curvature_intersection():
    from nystrom_ggn import build_nystrom_state

    curvature = torch.diag(torch.tensor([1.0e-8, 2.0e-8]))

    state = build_nystrom_state(
        lambda vector: curvature @ vector,
        dimension=2,
        indices=(0, 1),
        rank=2,
        damping=0.1,
    )

    assert state.rank == 2
