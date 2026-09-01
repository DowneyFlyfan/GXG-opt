from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn
from torchvision import models as vision_models


VOCABULARY_SIZE = 32_000
CONTEXT_LENGTH = 1_024


def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Use forward-AD-compatible math attention only for exact GGN products."""
    if torch.autograd.forward_ad.unpack_dual(query).tangent is None:
        return functional.scaled_dot_product_attention(query, key, value, is_causal=causal)
    scores = (query @ key.transpose(-2, -1)) * query.size(-1) ** -0.5
    if causal:
        sequence = query.size(-2)
        mask = torch.ones(
            sequence, sequence, dtype=torch.bool, device=query.device
        ).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(dim=-1) @ value


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward_in = nn.Linear(width, 4 * width, bias=False)
        self.feedforward_out = nn.Linear(4 * width, width, bias=False)
        self.heads = heads

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = hidden.shape
        qkv = self.qkv(self.norm1(hidden)).view(batch, sequence, 3, self.heads, width // self.heads)
        query, key, value = qkv.unbind(dim=2)
        attention = _attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            causal=self.causal,
        )
        hidden = hidden + self.projection(attention.transpose(1, 2).reshape(batch, sequence, width))
        feedforward = self.feedforward_out(functional.gelu(self.feedforward_in(self.norm2(hidden)), approximate="tanh"))
        return hidden + feedforward


class DecoderTransformer(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, vocabulary_size: int = VOCABULARY_SIZE) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, width)
        self.pos_embedding = nn.Parameter(torch.zeros(1, CONTEXT_LENGTH, width))
        self.blocks = nn.Sequential(*(TransformerBlock(width, heads, causal=True) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids) + self.pos_embedding[:, : token_ids.size(1)]
        return functional.linear(self.norm(self.blocks(hidden)), self.embedding.weight)


class AudioSpectrogramTransformer(nn.Module):
    def __init__(self, vocabulary_size: int = 128) -> None:
        super().__init__()
        width, heads, layers, patches = 768, 12, 12, 512
        self.patch_projection = nn.Conv2d(1, width, kernel_size=16, stride=16, bias=False)
        self.pos_embedding = nn.Parameter(torch.zeros(1, patches, width))
        self.blocks = nn.Sequential(*(TransformerBlock(width, heads, causal=False) for _ in range(layers)))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocabulary_size, bias=False)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        hidden = self.patch_projection(spectrogram).flatten(2).transpose(1, 2)
        hidden = hidden + self.pos_embedding[:, : hidden.size(1)]
        return self.head(self.norm(self.blocks(hidden)))


class DINOv3CIFAR100Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModel

        checkpoint = Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "models" / "DINOv3-ViT-B"
        self.backbone = AutoModel.from_pretrained(checkpoint, local_files_only=True)
        self.backbone.embeddings.requires_grad_(False)
        for layer in self.backbone.layer[:8]:
            layer.requires_grad_(False)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 100)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(pixel_values=images).pooler_output)


def create_nlp_model(name: str) -> nn.Module:
    constructors = {
        "gpt_12x512": lambda: DecoderTransformer(width=512, heads=8, layers=12),
        "gpt_12x768": lambda: DecoderTransformer(width=768, heads=12, layers=12),
    }
    if name == "smollm2_135m":
        from transformers import AutoModelForCausalLM

        checkpoint = Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "models" / "SmolLM2-135M"
        return AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True)
    return constructors[name]()


def create_cv_model(name: str) -> nn.Module:
    if name == "dinov3_vitb16":
        return DINOv3CIFAR100Classifier()
    if name != "vit_base_12x768":
        raise ValueError(f"Unsupported CV model: {name}")
    return vision_models.VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        num_classes=100,
    )


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def create_audio_model(name: str, root: Path | None = None) -> nn.Module:
    if name == "ast_base_12x768":
        return AudioSpectrogramTransformer()
    if name != "owsm_v3.1_base":
        raise ValueError(f"Unsupported audio model: {name}")

    project_root = root or Path(__file__).resolve().parents[1]
    checkpoint_root = project_root / ".cache" / "huggingface" / "models" / "owsm_v3.1_ebf_base"
    configuration = checkpoint_root / (
        "exp/s2t_train_s2t_ebf_conv2d_size384_e6_d6_piecewise_lr1e-3_"
        "warmup60k_flashattn_lessreg_raw_bpe50000/config.yaml"
    )
    weights = checkpoint_root / (
        "exp/s2t_train_s2t_ebf_conv2d_size384_e6_d6_piecewise_lr1e-3_"
        "warmup60k_flashattn_lessreg_raw_bpe50000/valid.total_count.ave_5best.pth"
    )
    if not configuration.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Missing local OWSM checkpoint below {checkpoint_root}")

    os.environ.setdefault("NLTK_DATA", str(project_root / ".cache" / "nltk"))
    from espnet2.tasks.s2t import S2TTask

    with _working_directory(checkpoint_root):
        model, _ = S2TTask.build_model_from_file(configuration, weights, "cpu")
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
