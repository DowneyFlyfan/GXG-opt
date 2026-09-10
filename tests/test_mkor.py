import torch
from torch import nn


def test_mkor_rank_one_update_keeps_identity_factors_positive_definite():
    from mkor import MKORHookState

    state = MKORHookState(2, 3, device=torch.device("cpu"))
    state.update(
        output_gradient=torch.tensor([1.0, -2.0]),
        activation=torch.tensor([3.0, 4.0, -5.0]),
        decay=0.9,
        stabilizer_threshold=1.0e6,
        stabilizer_mix=0.5,
    )

    assert torch.linalg.eigvalsh(state.left_inverse).min() > 0
    assert torch.linalg.eigvalsh(state.right_inverse).min() > 0


def test_mkor_repeated_rank_one_updates_remain_finite_and_bounded():
    from mkor import MKORHookState

    state = MKORHookState(4, 4, device=torch.device("cpu"))
    for _ in range(100):
        state.update(
            output_gradient=torch.full((4,), 100.0),
            activation=torch.full((4,), 100.0),
            decay=0.95,
            stabilizer_threshold=1.0e4,
            stabilizer_mix=0.1,
        )

    assert torch.isfinite(state.left_inverse).all()
    assert torch.isfinite(state.right_inverse).all()
    assert state.right_inverse.abs().max() <= 2.0


def test_eligible_linear_modules_excludes_vocabulary_sized_head():
    from mkor import eligible_linear_modules

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Linear(3, 2)
            self.head = nn.Linear(2, 11, bias=False)

    names = eligible_linear_modules(TinyModel())

    assert names == {"block"}


def test_mkor_step_norm_preserves_the_matrix_gradient():
    from mkor import MKOR

    parameter = nn.Parameter(torch.zeros(2, 3))
    optimizer = MKOR([parameter], lr=0.1, momentum=0.0, weight_decay=0.0)
    parameter.grad = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    optimizer.step()

    assert parameter.detach().norm() == torch.tensor(0.1).mul(parameter.grad.norm()).item()
