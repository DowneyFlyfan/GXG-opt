from __future__ import annotations

import argparse
import hashlib
import json
from itertools import chain
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepare(
    output_root: Path,
    *,
    model_name: str,
    dataset_name: str,
    dataset_config: str,
    sequence_length: int,
    processes: int,
    model_init: str = "pretrained",
) -> None:
    model_path = output_root / "gpt2-model"
    tokenizer_path = output_root / "gpt2-tokenizer"
    dataset_path = output_root / f"wikitext103-gpt2-{sequence_length}"
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(tokenizer_path)
    if model_init == "random":
        AutoConfig.from_pretrained(model_name).save_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.float().save_pretrained(model_path, safe_serialization=True)

    raw = load_dataset(dataset_name, dataset_config)

    def tokenize(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        texts = [text for text in examples["text"] if text.strip()]
        encoded = tokenizer(texts, add_special_tokens=False)
        return {
            "input_ids": [
                input_ids + [tokenizer.eos_token_id]
                for input_ids in encoded["input_ids"]
            ]
        }

    tokenized = raw.map(
        tokenize,
        batched=True,
        num_proc=processes,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing WikiText-103",
    )

    def group(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        joined = list(chain.from_iterable(examples["input_ids"]))
        usable = len(joined) // sequence_length * sequence_length
        return {
            "input_ids": [
                joined[index : index + sequence_length]
                for index in range(0, usable, sequence_length)
            ]
        }

    grouped = DatasetDict(
        {
            split: tokenized[split].map(
                group,
                batched=True,
                batch_size=1000,
                num_proc=processes,
                desc=f"Packing {split} into {sequence_length}-token blocks",
            )
            for split in ("train", "validation")
        }
    )
    grouped.save_to_disk(dataset_path)
    saved = load_from_disk(dataset_path) if model_init == "random" else grouped
    manifest = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "sequence_length": sequence_length,
        "splits": {
            split: {
                "blocks": len(grouped[split]),
                "tokens": len(grouped[split]) * sequence_length,
                "fingerprint": saved[split]._fingerprint,
            }
            for split in grouped
        },
    }
    if model_init == "random":
        manifest["model_init"] = model_init
        manifest["data_sha256"] = {}
        for path in sorted(dataset_path.rglob("*.arrow")):
            with path.open("rb") as handle:
                manifest["data_sha256"][str(path.relative_to(dataset_path))] = hashlib.file_digest(handle, "sha256").hexdigest()
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / ".cache" / "gpt2_wikitext103",
    )
    parser.add_argument("--model-name", default="openai-community/gpt2")
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--model-init", choices=("pretrained", "random"), default="pretrained")
    args = parser.parse_args()
    prepare(
        args.output_root,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        sequence_length=args.sequence_length,
        processes=args.processes,
        model_init=args.model_init,
    )


if __name__ == "__main__":
    main()
