from pathlib import Path

import data
import torch
from torch.utils.data import DataLoader


def test_librispeech_creates_its_project_cache_directory(tmp_path, monkeypatch):
    observed = {}

    class Dataset:
        def __len__(self):
            return 0

    def fake_librispeech(root: Path, **_):
        observed["root"] = root
        observed["exists"] = root.is_dir()
        return Dataset()

    monkeypatch.setattr(data.torchaudio.datasets, "LIBRISPEECH", fake_librispeech)

    data.LibriSpeechFeatures(tmp_path, "train-clean-100")

    assert observed == {"root": tmp_path / ".cache" / "audio", "exists": True}


def test_seeded_loader_options_reproduce_shuffled_token_batches():
    dataset = data.TokenStreamDataset(torch.arange(41, dtype=torch.int16), context_length=4)
    first = next(iter(DataLoader(dataset, shuffle=True, **data.loader_options(batch_size=2, workers=0, seed=17))))
    second = next(iter(DataLoader(dataset, shuffle=True, **data.loader_options(batch_size=2, workers=0, seed=17))))

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_smollm2_tokenizer_uses_the_local_checkpoint_vocabulary():
    root = Path(__file__).resolve().parents[1]

    token_ids = data.smollm2_tokenize(root, ["A short deterministic sentence."])

    assert token_ids.dtype == torch.long
    assert token_ids.numel() > 1
    assert token_ids.max().item() < 49_152


def test_smollm2_loader_caches_tokenizer_specific_streams():
    root = Path(__file__).resolve().parents[1]
    train, validation = data.smollm2_wikitext_loaders(
        root,
        batch_size=1,
        workers=0,
        train_tokens=4_096,
        validation_tokens=2_048,
    )

    inputs, labels = next(iter(train))

    assert inputs.shape == labels.shape == (1, data.CONTEXT_LENGTH)
    assert len(validation.dataset) == 1


def test_owsm_transcript_labels_include_the_asr_prefix(tmp_path):
    root = Path(__file__).resolve().parents[1]

    labels = data.owsm_transcript_labels(root, "HELLO WORLD")

    assert set(labels) == {"text", "text_prev", "text_ctc"}
    assert labels["text"].ndim == labels["text_prev"].ndim == labels["text_ctc"].ndim == 1
    assert labels["text"].numel() > labels["text_ctc"].numel()
    assert labels["text_prev"].numel() == 3


def test_owsm_ctc_labels_decode_to_the_original_transcript():
    root = Path(__file__).resolve().parents[1]
    transcript = "HELLO WORLD"

    decoded = data.owsm_decode_ctc_ids(root, data.owsm_transcript_labels(root, transcript)["text_ctc"])

    assert decoded == transcript


def test_owsm_loader_emits_the_speech_to_text_model_batch():
    root = Path(__file__).resolve().parents[1]
    train, _ = data.owsm_librispeech_loaders(root, batch_size=1, workers=0)

    batch = next(iter(train))

    assert set(batch) == {
        "speech",
        "speech_lengths",
        "text",
        "text_lengths",
        "text_prev",
        "text_prev_lengths",
        "text_ctc",
        "text_ctc_lengths",
        "transcripts",
    }
    assert batch["speech"].shape[0] == batch["text"].shape[0] == 1
    assert batch["speech_lengths"].item() > 0
