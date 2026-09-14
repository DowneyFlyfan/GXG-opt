from __future__ import annotations


def test_prepare_cache_writes_disjoint_uint32_streams_and_manifest(tmp_path):
    from qwen3_data import prepare_qwen_fineweb_cache

    cache = prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=16,
        validation_tokens=8,
        sequence_length=4,
        eos_token_id=9,
        source=[("train", "a", [1] * 16), ("validation", "b", [2] * 8)],
    )

    assert cache.train_path.stat().st_size == 16 * 4
    assert cache.validation_path.stat().st_size == 8 * 4
    assert cache.manifest["written_tokens"] == {"train": 16, "validation": 8}
    assert cache.manifest["document_counts"] == {"train": 1, "validation": 1}
    assert set(cache.manifest["document_id_sha256"]) == {"train", "validation"}


def test_block_loader_returns_shifted_fixed_width_tokens(tmp_path):
    from qwen3_data import prepare_qwen_fineweb_cache, qwen_block_loaders

    cache = prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=17,
        validation_tokens=9,
        sequence_length=4,
        eos_token_id=99,
        source=[("train", "a", list(range(17))), ("validation", "b", list(range(9)))],
    )

    train_loader, _ = qwen_block_loaders(cache, micro_batch_size=1, workers=0, seed=1)
    inputs, labels = next(iter(train_loader))

    assert inputs.tolist() == [[0, 1, 2, 3]]
    assert labels.tolist() == [[1, 2, 3, 4]]


def test_preparation_stops_streaming_once_both_targets_are_full(tmp_path):
    from qwen3_data import prepare_qwen_fineweb_cache

    def source():
        yield "train", "a", [1] * 8
        yield "validation", "b", [2] * 8
        raise AssertionError("the source must not be consumed after both targets are full")

    cache = prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=8,
        validation_tokens=8,
        sequence_length=4,
        eos_token_id=9,
        source=source(),
    )

    assert cache.manifest["written_tokens"] == {"train": 8, "validation": 8}


def test_load_cache_rejects_a_token_file_with_the_wrong_digest(tmp_path):
    from qwen3_data import load_qwen_token_cache, prepare_qwen_fineweb_cache

    cache = prepare_qwen_fineweb_cache(
        tmp_path,
        train_tokens=8,
        validation_tokens=8,
        sequence_length=4,
        eos_token_id=9,
        source=[("train", "a", [1] * 8), ("validation", "b", [2] * 8)],
    )
    cache.train_path.write_bytes(b"\0" * 32)

    try:
        load_qwen_token_cache(tmp_path)
    except ValueError as error:
        assert "digest" in str(error)
    else:
        raise AssertionError("corrupt cache was accepted")


def test_fineweb_revision_comes_from_the_hub_dataset_metadata(monkeypatch):
    import huggingface_hub
    from qwen3_data import fineweb_edu_revision

    class Api:
        def dataset_info(self, dataset_id):
            assert dataset_id == "HuggingFaceFW/fineweb-edu"
            return type("Info", (), {"sha": "pinned-sha"})()

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)

    assert fineweb_edu_revision() == "pinned-sha"
