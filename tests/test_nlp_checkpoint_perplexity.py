import json

import pytest
import torch

from nlp_checkpoint_perplexity import model_state_from_checkpoint, write_recovery_record


def test_model_state_from_checkpoint_reads_common_model_key():
    tensor = torch.tensor([1.0])

    assert model_state_from_checkpoint({"model": {"weight": tensor}}) == {"weight": tensor}


def test_model_state_from_checkpoint_rejects_checkpoint_without_model_state():
    with pytest.raises(ValueError, match="model state"):
        model_state_from_checkpoint({"optimizer": {}})


def test_write_recovery_record_preserves_checkpoint_and_perplexity(tmp_path):
    output = tmp_path / "recovered.ppl.json"

    write_recovery_record(
        output,
        checkpoint=".cache/nlp/checkpoints/example.checkpoint.pt",
        validation_batches=64,
        validation_nll=2.0,
    )

    assert json.loads(output.read_text()) == {
        "checkpoint": ".cache/nlp/checkpoints/example.checkpoint.pt",
        "perplexity": pytest.approx(7.38905609893065),
        "validation_batches": 64,
        "validation_nll": 2.0,
    }
