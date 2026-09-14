from __future__ import annotations

import torch


def test_muon_proposal_commit_matches_direct_custom_muon_step():
    from optimizers import Muon
    from qwen3_proposals import QwenMuonProposalAdapter

    torch.manual_seed(7)
    reference = torch.nn.Parameter(torch.randn(5, 3))
    candidate = torch.nn.Parameter(reference.detach().clone())
    gradient = torch.randn_like(reference)
    reference.grad = gradient.clone()
    candidate.grad = gradient.clone()

    direct = Muon([reference], lr=0.02, weight_decay=0.1)
    adapter = QwenMuonProposalAdapter({"matrix": candidate}, {"matrix"})
    direct.step()
    proposal = adapter.propose({"matrix": 0.02}, {"matrix": 0.1})
    adapter.commit(proposal)

    # The compiled bfloat16 Newton--Schulz path is numerically, not bitwise,
    # reproducible across separately compiled invocations.
    assert torch.allclose(candidate, reference, atol=2e-5, rtol=2e-5)
    assert torch.allclose(adapter.state["matrix"]["momentum_buffer"], direct.state[reference]["momentum_buffer"])


def test_disabled_filter_can_commit_the_unmodified_learning_increment():
    from qwen3_proposals import QwenMuonProposalAdapter

    parameter = torch.nn.Parameter(torch.ones(4, 2))
    parameter.grad = torch.full_like(parameter, 0.25)
    adapter = QwenMuonProposalAdapter({"matrix": parameter}, {"matrix"})

    proposal = adapter.propose({"matrix": 0.01}, {"matrix": 0.0})
    adapter.commit(proposal, {"matrix": torch.zeros_like(parameter)})

    assert torch.allclose(parameter, proposal["matrix"].value)


def test_qwen_notch_bypasses_the_first_unqualified_proposal_exactly():
    from qwen3_proposals import QwenMuonProposalAdapter, qwen_notch_corrections

    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    parameter.grad = torch.full_like(parameter, 0.2)
    adapter = QwenMuonProposalAdapter({"matrix": parameter}, {"matrix"})
    proposal = adapter.propose({"matrix": 0.01}, {"matrix": 0.0})

    corrections, state, diagnostics = qwen_notch_corrections(
        proposal, {"matrix": parameter}, {}, step=1, seed=11, learning_rates={"matrix": 0.01}
    )

    assert torch.equal(corrections["matrix"], torch.zeros_like(parameter))
    assert diagnostics["matrix"]["active"] is False
    assert state["filters"]["matrix"]["active"] is False
