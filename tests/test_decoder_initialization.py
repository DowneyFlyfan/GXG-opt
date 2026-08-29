import torch

from models import DecoderTransformer


def test_standard_decoder_initialization_sets_small_tied_embedding_scale():
    from decoder_initialization import initialize_decoder_transformer

    torch.manual_seed(7)
    model = DecoderTransformer(width=16, heads=4, layers=1, vocabulary_size=64)
    initialize_decoder_transformer(model)

    assert model.embedding.weight.std().item() < 0.03
    assert model.pos_embedding.std().item() < 0.02
    assert torch.allclose(model.norm.weight, torch.ones_like(model.norm.weight))

