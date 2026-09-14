from __future__ import annotations

import torch


class _Identity(torch.nn.Module):
    def forward(self, values):
        return values


class _RoutingAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"num_attention_heads": 4, "num_key_value_heads": 2})()
        self.head_dim = 2
        self.q_proj = torch.nn.Linear(4, 8, bias=False)
        self.k_proj = torch.nn.Linear(4, 4, bias=False)
        self.q_norm = _Identity()
        self.k_norm = _Identity()

    def forward(self, hidden_states, position_embeddings):
        del position_embeddings
        return hidden_states


class _RoutingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(3))
        self.model.layers[1].self_attn = _RoutingAttention()


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


def test_qwen_notch_optimizer_matches_muon_before_detection_is_eligible():
    from optimizers import Muon
    from qwen3_proposals import QwenProposalNotchOptimizer

    torch.manual_seed(16)
    reference = torch.nn.Parameter(torch.randn(5, 3))
    candidate = torch.nn.Parameter(reference.detach().clone())
    gradient = torch.randn_like(reference)
    reference.grad = gradient.clone()
    candidate.grad = gradient.clone()
    direct = Muon([reference], lr=0.02, weight_decay=0.1)
    proposal = QwenProposalNotchOptimizer(
        {"matrix": candidate}, {"matrix"}, learning_rate=0.02, weight_decay=0.1, seed=17
    )

    direct.step()
    proposal.step()

    assert torch.allclose(candidate, reference, atol=2e-5, rtol=2e-5)
    assert proposal.steps == 1
    assert proposal.last_diagnostics["matrix"]["active"] is False


def test_disabled_routing_controller_directly_matches_muon():
    from optimizers import Muon
    from qwen3_proposals import QwenRoutingResistanceOptimizer

    torch.manual_seed(21)
    model = _RoutingModel()
    attention = model.model.layers[1].self_attn
    names = {
        "model.layers.1.self_attn.q_proj.weight": attention.q_proj.weight,
        "model.layers.1.self_attn.k_proj.weight": attention.k_proj.weight,
    }
    references = [torch.nn.Parameter(parameter.detach().clone()) for parameter in names.values()]
    for parameter, reference in zip(names.values(), references):
        gradient = torch.randn_like(parameter)
        parameter.grad = gradient.clone()
        reference.grad = gradient.clone()
    direct = Muon(references, lr=0.02, weight_decay=0.1)
    proposal = QwenRoutingResistanceOptimizer(
        model, names, set(names), learning_rate=0.02, weight_decay=0.1, rho=0.0, interval=1, seed=22
    )
    hidden = torch.randn(1, 3, 4)
    position_embeddings = (torch.ones(1, 3, 2), torch.zeros(1, 3, 2))

    proposal.prepare_forward()
    attention(hidden, position_embeddings)
    direct.step()
    proposal.step()

    assert all(torch.allclose(parameter, reference, atol=2e-5, rtol=2e-5) for parameter, reference in zip(names.values(), references))
    assert proposal.steps == 1
    assert proposal.last_diagnostics["disabled"] is True


def test_routing_controller_filters_one_captured_query_key_head_and_releases_hook():
    from qwen3_proposals import QwenRoutingResistanceOptimizer

    torch.manual_seed(23)
    model = _RoutingModel()
    attention = model.model.layers[1].self_attn
    names = {
        "model.layers.1.self_attn.q_proj.weight": attention.q_proj.weight,
        "model.layers.1.self_attn.k_proj.weight": attention.k_proj.weight,
    }
    for parameter in names.values():
        parameter.grad = torch.randn_like(parameter)
    proposal = QwenRoutingResistanceOptimizer(
        model, names, set(names), learning_rate=0.02, weight_decay=0.0, rho=1.0, interval=1, seed=24
    )
    hidden = torch.randn(1, 4, 4)
    position_embeddings = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))

    proposal.prepare_forward()
    attention(hidden, position_embeddings)
    proposal.step()

    assert proposal.steps == 1
    assert proposal.last_diagnostics["sampled_edges"] == 12
    assert proposal.last_diagnostics["layer"] == 1
    assert proposal._capture is None
