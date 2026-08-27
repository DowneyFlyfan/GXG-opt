from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from models import CONTEXT_LENGTH


WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
CHARACTERS = " abcdefghijklmnopqrstuvwxyz'"
CHARACTER_IDS = {character: index + 1 for index, character in enumerate(CHARACTERS)}


@lru_cache(maxsize=1)
def _smollm2_tokenizer(project_root: Path):
    from transformers import AutoTokenizer

    checkpoint = project_root / ".cache" / "huggingface" / "models" / "SmolLM2-135M"
    return AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)


def smollm2_tokenize(project_root: Path, texts: list[str]) -> torch.Tensor:
    tokenizer = _smollm2_tokenizer(project_root.resolve())
    encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    end_of_text = tokenizer.eos_token_id
    token_stream = [token for sequence in encoded for token in (*sequence, end_of_text)]
    return torch.tensor(token_stream, dtype=torch.long)


@lru_cache(maxsize=1)
def _owsm_tokenizer_and_converter(project_root: Path):
    import yaml
    from espnet2.text.build_tokenizer import build_tokenizer
    from espnet2.text.token_id_converter import TokenIDConverter

    checkpoint_root = project_root / ".cache" / "huggingface" / "models" / "owsm_v3.1_ebf_base"
    configuration = checkpoint_root / (
        "exp/s2t_train_s2t_ebf_conv2d_size384_e6_d6_piecewise_lr1e-3_"
        "warmup60k_flashattn_lessreg_raw_bpe50000/config.yaml"
    )
    bpe_model = checkpoint_root / "data/token_list/bpe_unigram50000/bpe.model"
    if not configuration.is_file() or not bpe_model.is_file():
        raise FileNotFoundError(f"Missing local OWSM tokenizer below {checkpoint_root}")
    token_list = yaml.safe_load(configuration.read_text())["token_list"]
    return build_tokenizer("bpe", bpe_model), TokenIDConverter(token_list)


def owsm_transcript_labels(project_root: Path, transcript: str) -> dict[str, torch.Tensor]:
    tokenizer, converter = _owsm_tokenizer_and_converter(project_root.resolve())

    def tokenize(text: str) -> torch.Tensor:
        return torch.tensor(converter.tokens2ids(tokenizer.text2tokens(text)), dtype=torch.long)

    return {
        "text": tokenize(f"<eng><asr><notimestamps> {transcript}"),
        "text_prev": tokenize("<eng><asr>"),
        "text_ctc": tokenize(transcript),
    }


def owsm_decode_ctc_ids(project_root: Path, token_ids: torch.Tensor) -> str:
    tokenizer, converter = _owsm_tokenizer_and_converter(project_root.resolve())
    return tokenizer.tokens2text(converter.ids2tokens(token_ids.detach().cpu().tolist()))


class TokenStreamDataset(Dataset):
    def __init__(self, token_stream: torch.Tensor, context_length: int = CONTEXT_LENGTH) -> None:
        self.token_stream = token_stream
        self.context_length = context_length

    def __len__(self) -> int:
        return (self.token_stream.numel() - 1) // self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.context_length
        segment = self.token_stream[start : start + self.context_length + 1]
        return segment[:-1].long(), segment[1:].long()


def _byte_tokens(texts, maximum_tokens: int) -> torch.Tensor:
    stream = bytearray()
    for text in texts:
        stream.extend(text.encode("utf-8", errors="ignore"))
        if len(stream) >= maximum_tokens:
            break
    del stream[maximum_tokens:]
    return torch.frombuffer(stream, dtype=torch.uint8).clone().to(torch.int16) + 1


def loader_options(batch_size: int, workers: int, seed: int) -> dict:
    return {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "generator": torch.Generator().manual_seed(seed),
    }


def wikitext_loaders(root: Path, batch_size: int, workers: int, seed: int = 1337) -> tuple[DataLoader, DataLoader]:
    cache = root / ".cache" / "nlp"
    cache.mkdir(parents=True, exist_ok=True)
    train_cache, validation_cache = cache / "wikitext103_train_100m.pt", cache / "wikitext103_validation_10m.pt"
    if not train_cache.exists() or not validation_cache.exists():
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=str(cache / "hf"))
        torch.save(_byte_tokens(dataset["train"]["text"], 100_000_000), train_cache)
        torch.save(_byte_tokens(dataset["validation"]["text"], 10_000_000), validation_cache)
    train, validation = TokenStreamDataset(torch.load(train_cache, weights_only=True)), TokenStreamDataset(torch.load(validation_cache, weights_only=True))
    return (
        DataLoader(train, shuffle=True, **loader_options(batch_size, workers, seed)),
        DataLoader(validation, shuffle=False, **loader_options(batch_size, workers, seed + 1)),
    )


def _smollm2_token_stream(project_root: Path, texts, maximum_tokens: int) -> torch.Tensor:
    tokenizer = _smollm2_tokenizer(project_root.resolve())
    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        tokens.append(tokenizer.eos_token_id)
        if len(tokens) >= maximum_tokens:
            break
    return torch.tensor(tokens[:maximum_tokens], dtype=torch.long)


def smollm2_wikitext_loaders(
    root: Path,
    batch_size: int,
    workers: int,
    seed: int = 1337,
    train_tokens: int = 100_000_000,
    validation_tokens: int = 10_000_000,
) -> tuple[DataLoader, DataLoader]:
    cache = root / ".cache" / "nlp"
    cache.mkdir(parents=True, exist_ok=True)
    train_cache = cache / f"smollm2_wikitext103_train_{train_tokens}.pt"
    validation_cache = cache / f"smollm2_wikitext103_validation_{validation_tokens}.pt"
    if not train_cache.exists() or not validation_cache.exists():
        dataset = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            cache_dir=str(cache / "hf"),
        )
        if not train_cache.exists():
            torch.save(_smollm2_token_stream(root, dataset["train"]["text"], train_tokens), train_cache)
        if not validation_cache.exists():
            torch.save(_smollm2_token_stream(root, dataset["validation"]["text"], validation_tokens), validation_cache)
    train = TokenStreamDataset(torch.load(train_cache, weights_only=True))
    validation = TokenStreamDataset(torch.load(validation_cache, weights_only=True))
    return (
        DataLoader(train, shuffle=True, **loader_options(batch_size, workers, seed)),
        DataLoader(validation, shuffle=False, **loader_options(batch_size, workers, seed + 1)),
    )


def cifar100_loaders(root: Path, batch_size: int, workers: int, seed: int = 1337) -> tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
    )
    validation_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    cache = root / ".cache" / "cv"
    train = datasets.CIFAR100(cache, train=True, download=True, transform=train_transform)
    validation = datasets.CIFAR100(cache, train=False, download=True, transform=validation_transform)
    return (
        DataLoader(train, shuffle=True, **loader_options(batch_size, workers, seed)),
        DataLoader(validation, shuffle=False, **loader_options(batch_size, workers, seed + 1)),
    )


def _transcript_tokens(text: str) -> torch.Tensor:
    return torch.tensor([CHARACTER_IDS[character] for character in text.lower() if character in CHARACTER_IDS] or [1])


class LibriSpeechFeatures(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        cache = root / ".cache" / "audio"
        cache.mkdir(parents=True, exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(cache, url=split, download=True)
        self.resample = torchaudio.transforms.Resample(16_000, 16_000)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=16_000, n_mels=128, n_fft=400, hop_length=160)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate, transcript, *_ = self.dataset[index]
        waveform = self.resample(waveform) if sample_rate != 16_000 else waveform
        waveform = waveform.mean(dim=0, keepdim=True)
        target_samples = 160_000
        waveform = torch.nn.functional.pad(waveform[:, :target_samples], (0, max(target_samples - waveform.size(1), 0)))
        feature = torch.log1p(self.mel(waveform))[:, :, :1024]
        feature = torch.nn.functional.pad(feature, (0, max(1024 - feature.size(-1), 0)))
        return feature, _transcript_tokens(transcript)


class OWSMLibriSpeech(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        cache = root / ".cache" / "audio"
        cache.mkdir(parents=True, exist_ok=True)
        self.dataset = torchaudio.datasets.LIBRISPEECH(cache, url=split, download=False)
        self.project_root = root

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        waveform, sample_rate, transcript, *_ = self.dataset[index]
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        labels = owsm_transcript_labels(self.project_root, transcript)
        return {"speech": waveform.mean(dim=0), **labels, "transcript": transcript}


def audio_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, targets = zip(*batch, strict=True)
    lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
    return torch.stack(features), torch.cat(targets), lengths


def owsm_audio_collate(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    def pad(name: str, padding_value: float | int) -> torch.Tensor:
        values = [item[name] for item in batch]
        assert all(isinstance(value, torch.Tensor) for value in values)
        return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=padding_value)

    speech = pad("speech", 0.0)
    text, text_prev, text_ctc = (pad(name, -1) for name in ("text", "text_prev", "text_ctc"))
    return {
        "speech": speech,
        "speech_lengths": torch.tensor([item["speech"].numel() for item in batch], dtype=torch.long),
        "text": text,
        "text_lengths": torch.tensor([item["text"].numel() for item in batch], dtype=torch.long),
        "text_prev": text_prev,
        "text_prev_lengths": torch.tensor([item["text_prev"].numel() for item in batch], dtype=torch.long),
        "text_ctc": text_ctc,
        "text_ctc_lengths": torch.tensor([item["text_ctc"].numel() for item in batch], dtype=torch.long),
        "transcripts": [str(item["transcript"]) for item in batch],
    }


def librispeech_loaders(root: Path, batch_size: int, workers: int, seed: int = 1337) -> tuple[DataLoader, DataLoader]:
    train, validation = LibriSpeechFeatures(root, "train-clean-100"), LibriSpeechFeatures(root, "dev-clean")
    train_options = loader_options(batch_size, workers, seed) | {"collate_fn": audio_collate}
    validation_options = loader_options(batch_size, workers, seed + 1) | {"collate_fn": audio_collate}
    return DataLoader(train, shuffle=True, **train_options), DataLoader(validation, shuffle=False, **validation_options)


def owsm_librispeech_loaders(
    root: Path, batch_size: int, workers: int, seed: int = 1337
) -> tuple[DataLoader, DataLoader]:
    train, validation = OWSMLibriSpeech(root, "train-clean-100"), OWSMLibriSpeech(root, "dev-clean")
    train_options = loader_options(batch_size, workers, seed) | {"collate_fn": owsm_audio_collate}
    validation_options = loader_options(batch_size, workers, seed + 1) | {"collate_fn": owsm_audio_collate}
    return DataLoader(train, shuffle=True, **train_options), DataLoader(validation, shuffle=False, **validation_options)
