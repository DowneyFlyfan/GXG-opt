from __future__ import annotations

import torch


class _TinyTiedQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(11, 4)
        self.lm_head = torch.nn.Linear(4, 11, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        del use_cache
        return type("Output", (), {"logits": self.lm_head(self.model.embed_tokens(input_ids))})()


def test_tied_adamw_proposal_commit_matches_two_direct_adamw_steps():
    from qwen3_tied import QwenTiedAdamWProposalAdapter

    torch.manual_seed(25)
    reference = torch.nn.Parameter(torch.randn(11, 4))
    candidate = torch.nn.Parameter(reference.detach().clone())
    direct = torch.optim.AdamW([reference], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    adapter = QwenTiedAdamWProposalAdapter(candidate, learning_rate=3e-4, weight_decay=0.1)
    for _ in range(2):
        gradient = torch.randn_like(reference)
        reference.grad = gradient.clone()
        candidate.grad = gradient.clone()
        direct.step()
        proposal = adapter.propose()
        adapter.commit(proposal)

    assert torch.allclose(candidate, reference, atol=2e-7, rtol=2e-7)


def test_disabled_tied_path_controller_matches_direct_adamw_with_one_physical_embedding():
    from qwen3_tied import QwenTiedPathOptimizer

    torch.manual_seed(26)
    model = _TinyTiedQwen()
    candidate = model.model.embed_tokens.weight
    reference = torch.nn.Parameter(candidate.detach().clone())
    direct = torch.optim.AdamW([reference], lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    gradient = torch.randn_like(candidate)
    candidate.grad = gradient.clone()
    reference.grad = gradient.clone()
    optimizer = QwenTiedPathOptimizer(
        model, candidate, learning_rate=3e-4, weight_decay=0.1, rho=0.0, seed=27
    )

    optimizer.prepare_batch(torch.tensor([[1, 2, 3]]))
    direct.step()
    optimizer.step()

    assert torch.allclose(candidate, reference, atol=2e-7, rtol=2e-7)
    assert optimizer.steps == 1
    assert optimizer.last_diagnostics["disabled"] is True


def test_tied_path_controller_refreshes_a_paired_joint_sketch_before_one_commit():
    from qwen3_tied import QwenTiedPathOptimizer

    torch.manual_seed(28)
    model = _TinyTiedQwen()
    parameter = model.model.embed_tokens.weight
    parameter.grad = torch.randn_like(parameter)
    optimizer = QwenTiedPathOptimizer(
        model, parameter, learning_rate=3e-4, weight_decay=0.0, rho=1.0, probes=2, interval=1, seed=29
    )

    optimizer.prepare_batch(torch.tensor([[1, 2, 3]]))
    optimizer.step()

    assert optimizer.steps == 1
    assert optimizer.last_diagnostics["refresh"] is True
    assert optimizer.last_diagnostics["metric_age"] == 0
    assert len(optimizer.columns) == 2


def test_split_tied_forward_matches_the_physical_tied_forward():
    from qwen3_tied import split_qwen_tied_forward

    torch.manual_seed(5)
    model = _TinyTiedQwen()
    ids = torch.tensor([[1, 2, 3]])
    embedding = model.model.embed_tokens.weight.detach()

    split = split_qwen_tied_forward(model, ids, embedding, embedding)
    direct = model(input_ids=ids).logits

    assert torch.allclose(split, direct)


def test_split_tied_paths_produce_two_gradients_that_sum_to_tied_gradient():
    from qwen3_tied import split_qwen_tied_forward

    torch.manual_seed(6)
    model = _TinyTiedQwen()
    ids = torch.tensor([[1, 2, 3]])
    input_leaf = model.model.embed_tokens.weight.detach().clone().requires_grad_()
    output_leaf = model.lm_head.weight.detach().clone().requires_grad_()
    probe = torch.randn(1, 3, 11)

    logits = split_qwen_tied_forward(model, ids, input_leaf, output_leaf)
    left, right = torch.autograd.grad((logits * probe).sum(), (input_leaf, output_leaf))
    model.zero_grad(set_to_none=True)
    (model(input_ids=ids).logits * probe).sum().backward()

    assert torch.allclose(left + right, model.model.embed_tokens.weight.grad)


def test_paired_tied_sketch_uses_one_column_per_shared_categorical_probe():
    from qwen3_tied import qwen_paired_embedding_sketch

    torch.manual_seed(8)
    model = _TinyTiedQwen()
    ids = torch.tensor([[1, 2, 3]])

    columns, diagnostics = qwen_paired_embedding_sketch(
        model, ids, count=2, generator=torch.Generator().manual_seed(9)
    )

    assert len(columns) == len(diagnostics) == 2
    assert all(column.shape == model.model.embed_tokens.weight.shape for column in columns)
    assert all(item["input_norm"] > 0 and item["output_norm"] > 0 for item in diagnostics)


def test_tied_proximal_filter_has_an_exact_zero_strength_bypass():
    from qwen3_tied import qwen_tied_proximal_correction

    learning = torch.randn(11, 4)
    columns = [torch.randn_like(learning), torch.randn_like(learning)]
    correction, diagnostics = qwen_tied_proximal_correction(learning, columns, rho=0.0)

    assert torch.equal(correction, torch.zeros_like(learning))
    assert diagnostics["kappa"] == 0.0
