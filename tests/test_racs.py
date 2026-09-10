import torch


def test_racs_scales_rows_and_columns_and_limits_update_norm():
    from racs import RACS

    parameter = torch.nn.Parameter(torch.zeros(2, 3))
    optimizer = RACS([parameter], lr=1.0, beta=0.9, scale=1.0, limiter=0.5)
    parameter.grad = torch.tensor([[1.0, 4.0, 9.0], [2.0, 7.0, 17.0]])
    optimizer.step()
    assert not torch.allclose(
        optimizer.state[parameter]["q"][0], optimizer.state[parameter]["q"][1]
    )
    parameter.grad = torch.full((2, 3), 100.0)
    optimizer.step()

    assert optimizer.state[parameter]["limited_norm"] <= 0.5 + 1.0e-6
