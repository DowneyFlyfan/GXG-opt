"""Deterministic packed-token cache for the Qwen3 optimizer study."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG = "sample-10BT"
DEFAULT_SEQUENCE_LENGTH = 2_048


@dataclass(frozen=True)
class QwenTokenCache:
    """Paths and verified metadata for one immutable packed-token cache."""

    root: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    sequence_length: int
    manifest: dict


class _PackedTokenDataset(Dataset):
    def __init__(self, path: Path, token_count: int, sequence_length: int) -> None:
        self.path = path
        self.token_count = token_count
        self.sequence_length = sequence_length
        self._tokens: np.memmap | None = None

    def __len__(self) -> int:
        return (self.token_count - 1) // self.sequence_length

    def _stream(self) -> np.memmap:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, mode="r", dtype=np.dtype("<u4"))
        return self._tokens

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        segment = np.array(
            self._stream()[start : start + self.sequence_length + 1], copy=True
        )
        return (
            torch.from_numpy(segment[:-1]).long(),
            torch.from_numpy(segment[1:]).long(),
        )


def _cache_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    cache_root = root / ".cache" / "qwen3_0p6b_fineweb_edu"
    return (
        cache_root,
        cache_root / "train.uint32",
        cache_root / "validation.uint32",
        cache_root / "manifest.json",
    )


def prepare_qwen_fineweb_cache(
    root: Path,
    *,
    train_tokens: int,
    validation_tokens: int,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    eos_token_id: int,
    source: Iterable[tuple[str, str, Sequence[int]]],
    source_name: str = "injected",
    source_revision: str = "injected",
    selection_seed: int = 1337,
) -> QwenTokenCache:
    """Pack a pre-tokenized, already-disjoint train/validation source.

    ``source`` emits ``(split, document_id, token_ids)``.  Keeping source
    selection outside this pure packing function makes its split rule auditable
    and lets the unit tests avoid a model or network download.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if not 0 <= eos_token_id <= np.iinfo(np.uint32).max:
        raise ValueError("eos_token_id must fit uint32")
    cache_root, train_path, validation_path, manifest_path = _cache_paths(root)
    if any(path.exists() for path in (train_path, validation_path, manifest_path)):
        raise FileExistsError(f"Qwen token cache already exists below {cache_root}")

    targets = {"train": train_tokens, "validation": validation_tokens}
    if any(target <= 1 for target in targets.values()):
        raise ValueError("token targets must leave at least one shifted label")
    temporary_paths = {
        split: path.with_suffix(path.suffix + ".tmp")
        for split, path in (("train", train_path), ("validation", validation_path))
    }
    written = {"train": 0, "validation": 0}
    token_digests = {split: hashlib.sha256() for split in targets}
    document_digests = {split: hashlib.sha256() for split in targets}
    document_counts = {split: 0 for split in targets}
    selected_ids = {split: set() for split in targets}
    cache_root.mkdir(parents=True, exist_ok=True)
    try:
        with (
            temporary_paths["train"].open("wb") as train_handle,
            temporary_paths["validation"].open("wb") as validation_handle,
        ):
            handles = {"train": train_handle, "validation": validation_handle}
            for split, document_id, token_ids in source:
                if all(written[name] == targets[name] for name in targets):
                    break
                if split not in targets:
                    raise ValueError(f"unsupported split: {split}")
                if written[split] == targets[split]:
                    continue
                if document_id in selected_ids[split]:
                    raise ValueError(f"duplicate document id in {split}: {document_id}")
                if document_id in selected_ids["validation" if split == "train" else "train"]:
                    raise ValueError(f"document ids cannot appear in both splits: {document_id}")
                payload = np.asarray([*token_ids, eos_token_id], dtype=np.dtype("<u4"))
                payload = payload[: targets[split] - written[split]]
                encoded = payload.tobytes()
                handles[split].write(encoded)
                token_digests[split].update(encoded)
                document_digests[split].update(document_id.encode("utf-8"))
                document_digests[split].update(b"\0")
                selected_ids[split].add(document_id)
                written[split] += len(payload)
                document_counts[split] += 1
                if all(written[name] == targets[name] for name in targets):
                    break
        if written != targets:
            raise RuntimeError(f"source ended with token counts {written}; expected {targets}")
        os.replace(temporary_paths["train"], train_path)
        os.replace(temporary_paths["validation"], validation_path)
    except BaseException:
        for path in (*temporary_paths.values(), train_path, validation_path):
            path.unlink(missing_ok=True)
        raise
    manifest = {
        "format": "qwen3-packed-token-cache-v1",
        "source": {"dataset": source_name, "revision": source_revision},
        "selection_seed": selection_seed,
        "sequence_length": sequence_length,
        "dtype": "uint32-little-endian",
        "eos_token_id": eos_token_id,
        "requested_tokens": {"train": train_tokens, "validation": validation_tokens},
        "written_tokens": written,
        "sha256": {split: digest.hexdigest() for split, digest in token_digests.items()},
        "document_counts": document_counts,
        "document_id_sha256": {
            split: digest.hexdigest() for split, digest in document_digests.items()
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return QwenTokenCache(
        root=cache_root,
        train_path=train_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        sequence_length=sequence_length,
        manifest=manifest,
    )


def qwen_block_loaders(
    cache: QwenTokenCache,
    *,
    micro_batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Return deterministic train and validation loaders over fixed blocks."""
    if micro_batch_size <= 0 or workers < 0:
        raise ValueError("micro_batch_size must be positive and workers non-negative")
    options = {
        "batch_size": micro_batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    train = _PackedTokenDataset(
        cache.train_path, int(cache.manifest["written_tokens"]["train"]), cache.sequence_length
    )
    validation = _PackedTokenDataset(
        cache.validation_path,
        int(cache.manifest["written_tokens"]["validation"]),
        cache.sequence_length,
    )
    return (
        DataLoader(train, shuffle=True, generator=torch.Generator().manual_seed(seed), **options),
        DataLoader(
            validation,
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 1),
            **options,
        ),
    )


def stream_fineweb_edu_tokens(
    root: Path,
    *,
    tokenizer_path: Path,
    selection_seed: int,
) -> Iterator[tuple[str, str, Sequence[int]]]:
    """Yield a deterministic hash split of FineWeb-Edu without retaining shards."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    stream = load_dataset(
        FINEWEB_EDU_DATASET,
        FINEWEB_EDU_CONFIG,
        split="train",
        streaming=True,
        cache_dir=str(root / ".cache" / "qwen3_0p6b_fineweb_edu" / "hf_temporary"),
    )
    for row_index, row in enumerate(stream):
        document_id = str(row.get("id", row_index))
        bucket = int.from_bytes(
            hashlib.sha256(f"{selection_seed}:{document_id}".encode("utf-8")).digest()[:8],
            "big",
        ) % 100
        split = "validation" if bucket < 5 else "train"
        token_ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
        yield split, document_id, token_ids
