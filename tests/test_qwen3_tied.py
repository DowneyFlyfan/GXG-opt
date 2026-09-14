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
