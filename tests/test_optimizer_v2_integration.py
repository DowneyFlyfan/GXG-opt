import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.func import functional_call
from transformers import GPT2Config, GPT2LMHeadModel

from scripts.run_gpt2_v2_comparison import load_config
from gpt2_v2_experiment import (
    checkpoint_path, comparison_plan, epoch_batches, epoch_order, evaluate, load_checkpoint,
    resolved_config, run, run_one, save_checkpoint, trim_log,
)
from optimizer_v2.adapter import ProposalAdapter
from optimizer_v2.layernorm import make_local_basis, reduced_ggn
from optimizer_v2.linalg import ProbeFailure
from optimizer_v2.optimizer import METHODS, OptimizerV2
from optimizer_v2.probes import collect_factors, collect_head_probe, lm_loss


pytestmark = pytest.mark.skipif(
    torch.__version__.split("+")[0] != "2.11.0",
    reason="optimizer 2.0 integration requires the isolated PyTorch 2.11 environment",
)


@pytest.fixture(autouse=True)
def deterministic_runtime():
    torch.manual_seed(31)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_model():
    return GPT2LMHeadModel(GPT2Config(vocab_size=19, n_embd=8, n_layer=1, n_head=2,
                                      n_positions=16, attn_implementation="eager", use_cache=False))


def assert_tree(first, second, exact=True):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0 if exact else 1e-6, atol=0 if exact else 1e-7)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree(first[key], second[key], exact)
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_tree(left, right, exact)
    else:
        assert first == second


def backward(model, ids):
    model.zero_grad(set_to_none=True)
    loss = lm_loss(model(input_ids=ids).logits, ids)
    loss.backward()
    return float(loss.detach())


@pytest.mark.parametrize("adam_only", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adapter_proposals_are_pure_and_match_stock_commits(adam_only, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = tiny_model().to(device)
    stock_model = copy.deepcopy(model)
    adapter = ProposalAdapter(model, adam_only=adam_only)
    stock = ProposalAdapter(stock_model, adam_only=adam_only)
    for _ in range(3):
        for parameter, reference in zip(model.parameters(), stock_model.parameters()):
            parameter.grad = torch.randn_like(parameter)
            reference.grad = parameter.grad.clone()
        saved_parameters = copy.deepcopy(model.state_dict())
        saved_state = copy.deepcopy(adapter.state_dict())
        first, second = adapter.propose(), adapter.propose()
        assert_tree(model.state_dict(), saved_parameters)
        assert_tree(adapter.state_dict(), saved_state)
        for name in first:
            assert_tree(first[name].next_state, second[name].next_state)
        adapter.commit(first)
        stock.step()
        assert_tree(model.state_dict(), stock_model.state_dict())
        assert_tree(adapter.state_dict(), stock.state_dict())
    assert model.transformer.wte.weight is model.lm_head.weight
    assert "transformer.wte.weight" not in adapter.matrix_names
    assert "transformer.h.0.attn.c_attn.weight" in adapter.matrix_names


@pytest.mark.parametrize("method", METHODS[2:])
def test_disabled_methods_reproduce_muon_without_auxiliary_state(method):
    model = tiny_model()
    reference = copy.deepcopy(model)
    candidate = OptimizerV2(model, method, {"enabled": False})
    baseline = OptimizerV2(reference, "muon")
    for _ in range(3):
        for p, q in zip(model.parameters(), reference.parameters()):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        candidate.step()
        baseline.step()
    assert_tree(model.state_dict(), reference.state_dict())
    assert_tree(candidate.adapter.state_dict(), baseline.adapter.state_dict())
    assert candidate.state == candidate.signs == {}


def test_probe_gradient_factorization_modes_rng_and_training_grads_are_preserved():
    model = tiny_model().train()
    ids = torch.randint(19, (1, 8))
    backward(model, ids)
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    rng = torch.random.get_rng_state().clone()
    name = "transformer.h.0.mlp.c_proj"
    factors = collect_factors(model, ids, [name])
    probe = collect_head_probe(model, ids, 0, 1, 4, torch.Generator().manual_seed(17), need_error=True)
    assert model.training and all(module.training for module in model.modules())
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    for parameter_name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[parameter_name], rtol=0, atol=0)
    model.eval()
    loss = lm_loss(model(input_ids=ids).logits, ids)
    expected, = torch.autograd.grad(loss, model.transformer.h[0].mlp.c_proj.weight)
    torch.testing.assert_close(factors[name][0].T @ factors[name][1], expected)
    assert probe["x"].shape == (8, 8) and probe["q"].shape == (8, 4)


def test_reduced_ggn_matches_independent_dense_coordinate_jacobian_at_provisional_model():
    model = tiny_model().double().eval()
    ids = torch.randint(19, (1, 5))
    backward(model, ids)
    adapter = ProposalAdapter(model, lr=0.01)
    proposals = adapter.propose()
    basis, x_size, _, _ = make_local_basis(model, proposals, 0)
    provisional = {name: proposals[name].value for name, _ in model.named_parameters()}
    gradient, metric = reduced_ggn(model, provisional, basis, ids)
    def output(coordinates):
        parameters = dict(provisional)
        for index, column in enumerate(basis):
            for name, direction in column.items():
                parameters[name] = parameters[name] + coordinates[index] * direction
        return functional_call(model, parameters, (), {"input_ids": ids}).logits[:, :-1]
    zero = torch.zeros(len(basis), dtype=torch.float64, requires_grad=True)
    logits = output(zero)
    jacobian = torch.autograd.functional.jacobian(output, zero).reshape(-1, len(basis))
    probabilities = logits.detach().softmax(-1).reshape(-1, 19)
    covariance = torch.block_diag(*[(torch.diag(p) - p[:, None] * p[None, :]) / len(probabilities) for p in probabilities])
    expected = jacobian.T @ covariance @ jacobian
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 19), ids[:, 1:].reshape(-1))
    g, = torch.autograd.grad(loss, zero)
    torch.testing.assert_close(metric, expected, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(gradient, g, atol=1e-10, rtol=1e-8)
    assert float(metric[:x_size, x_size:].norm()) > 1e-12


@pytest.mark.parametrize("method", METHODS[2:])
def test_all_candidates_checkpoint_resume_next_update(method):
    options = {"interval": 1} if method != "proposal_notch_v1" else {}
    model = tiny_model().eval()
    optimizer = OptimizerV2(model, method, options)
    fit, check = torch.randint(19, (1, 8)), torch.randint(19, (1, 8))
    backward(model, fit)
    optimizer.step(fit, check)
    model_saved, optimizer_saved = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    restored = tiny_model().eval()
    restored.load_state_dict(model_saved)
    continuation = OptimizerV2(restored, method, options)
    continuation.load_state_dict(optimizer_saved)
    backward(model, check)
    backward(restored, check)
    optimizer.step(fit, check)
    continuation.step(fit, check)
    assert_tree(model.state_dict(), restored.state_dict())
    assert_tree(optimizer.adapter.state_dict(), continuation.adapter.state_dict())
    assert_tree(optimizer.state, continuation.state)


def test_nonfinite_training_and_failed_auxiliary_are_transactional(monkeypatch):
    model = tiny_model()
    optimizer = OptimizerV2(model, "qk_defect_v1", {"interval": 1})
    reference = copy.deepcopy(model)
    baseline = OptimizerV2(reference, "muon")
    for p, q in zip(model.parameters(), reference.parameters()):
        p.grad = torch.randn_like(p)
        q.grad = p.grad.clone()
    def fail(*args, **kwargs):
        raise ProbeFailure("injected solve failure")
    monkeypatch.setattr(optimizer, "_head_correction", fail)
    optimizer.step(torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long))
    baseline.step()
    assert_tree(model.state_dict(), reference.state_dict())
    assert_tree(optimizer.adapter.state_dict(), baseline.adapter.state_dict())
    saved = copy.deepcopy(optimizer.state_dict())
    next(model.parameters()).grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        optimizer.step()
    assert_tree(optimizer.state_dict(), saved)


def test_tied_cache_age_failure_and_single_physical_update(monkeypatch):
    import optimizer_v2.optimizer as module
    model = tiny_model().eval()
    optimizer = OptimizerV2(model, "tied_path_curvature_v1", {"interval": 2, "max_age": 2})
    ids = torch.randint(19, (1, 8))
    backward(model, ids)
    optimizer.step(ids, ids)
    assert optimizer.state["cache_step"] == 1
    backward(model, ids)
    optimizer.step(ids, ids)
    assert optimizer.last_diagnostics["metric_age"] == 1
    reference = copy.deepcopy(model)
    baseline = ProposalAdapter(reference)
    baseline.load_state_dict(copy.deepcopy(optimizer.adapter.state_dict()))
    def fail(*args, **kwargs):
        raise ProbeFailure("failed refresh")
    monkeypatch.setattr(module, "paired_embedding_sketch", fail)
    backward(model, ids)
    backward(reference, ids)
    optimizer.step(ids, ids)
    baseline.step()
    assert optimizer.last_diagnostics["stale_cache_bypass"]
    assert_tree(model.state_dict(), reference.state_dict())


def test_feature_failed_audit_does_not_control_maps_or_updates(monkeypatch):
    model = tiny_model().eval()
    reference = copy.deepcopy(model)
    options = {"interval": 1, "audit_interval": 1}
    candidate = OptimizerV2(model, "feature_remap_cohort_v1", options)
    control = OptimizerV2(reference, "feature_remap_cohort_v1", options)
    fit, check, audit = [torch.randint(19, (1, 8)) for _ in range(3)]
    original = candidate._factors
    def factors(ids):
        if ids is audit:
            raise ProbeFailure("diagnostic-only audit failed")
        return original(ids)
    monkeypatch.setattr(candidate, "_factors", factors)
    for _ in range(2):
        backward(model, fit)
        backward(reference, fit)
        candidate.step(fit, check, audit)
        control.step(fit, check)
    assert candidate.last_diagnostics["audit_failure"]
    assert candidate.cost["fallback_events"] == 0
    assert_tree(model.state_dict(), reference.state_dict())
    assert_tree(candidate.adapter.state_dict(), control.adapter.state_dict())


def test_nonfinite_probe_counts_only_attempted_passes_and_preserves_gradients(monkeypatch):
    from types import SimpleNamespace
    model = tiny_model().train()
    ids = torch.randint(19, (1, 8))
    backward(model, ids)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    original = model.forward
    def invalid(*args, **kwargs):
        return SimpleNamespace(logits=original(*args, **kwargs).logits * float("nan"))
    monkeypatch.setattr(model, "forward", invalid)
    calls = []
    with pytest.raises(ProbeFailure):
        collect_factors(model, ids, ["transformer.h.0.mlp.c_proj"], counter=lambda **count: calls.append(count))
    assert calls == [{"forward": 1}]
    assert model.training
    for parameter, gradient in zip(model.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)


def test_epoch_coverage_partial_accumulation_and_plan():
    first = []
    for epoch in range(5):
        batches = list(epoch_batches(17, 0, epoch, 4, 2))
        flattened = [index for _, chunks in batches for chunk in chunks for index in chunk]
        assert sorted(flattened) == list(range(17))
        assert [sum(map(len, chunks)) for _, chunks in batches] == [8, 8, 1]
        assert [offset for offset, _ in batches] == [8, 16, 17]
        resumed = [index for _, chunks in epoch_batches(17, 0, epoch, 4, 2, 8) for chunk in chunks for index in chunk]
        assert resumed == flattened[8:]
        first.append(flattened)
    assert first[0] != first[1]
    config = load_config(Path("configs/experiments/gpt2_v2_5090.yaml"))
    plan = comparison_plan(config, 232000)
    assert plan["runs"] == 8 and plan["steps_per_run"] == 36250
    assert plan["full_budget_input_tokens_per_run"] == 593_920_000
    selected = resolved_config(config, ["muon"], [7], 3)
    assert comparison_plan(selected, 232000)["steps_per_run"] == 3
    assert selected["training"]["seeds"] == [7]
    with pytest.raises(ValueError):
        resolved_config(config, steps=0)


@pytest.mark.parametrize("initialization", ["random", "pretrained"])
def test_asset_preparation_preserves_pretrained_default_and_random_has_no_weights(tmp_path, monkeypatch, initialization):
    from unittest.mock import Mock
    from scripts import prepare_gpt2_wikitext as preparation
    datasets = pytest.importorskip("datasets")

    tokenizer = Mock(eos_token="<eos>", eos_token_id=0)
    tokenizer.side_effect = lambda texts, **kwargs: {"input_ids": [[1, 2, 3] for _ in texts]}
    monkeypatch.setattr(preparation.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer))
    monkeypatch.setattr(preparation.AutoConfig, "from_pretrained", Mock(return_value=tiny_model().config))
    pretrained = Mock(return_value=tiny_model())
    monkeypatch.setattr(preparation.AutoModelForCausalLM, "from_pretrained", pretrained)
    raw = datasets.DatasetDict({split: datasets.Dataset.from_dict({"text": ["first", "second"]})
                                for split in ("train", "validation")})
    monkeypatch.setattr(preparation, "load_dataset", Mock(return_value=raw))
    options = {"model_init": "random"} if initialization == "random" else {}
    preparation.prepare(tmp_path, model_name="local-test", dataset_name="local-test",
                        dataset_config="test", sequence_length=4, processes=None, **options)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    weights = list((tmp_path / "gpt2-model").glob("*.safetensors"))
    assert (tmp_path / "gpt2-model/config.json").is_file()
    assert manifest["splits"]["train"]["blocks"] == 2
    if initialization == "random":
        pretrained.assert_not_called()
        assert not weights and manifest["model_init"] == "random"
        saved = datasets.load_from_disk(tmp_path / "wikitext103-gpt2-4")
        assert manifest["splits"]["train"]["fingerprint"] == saved["train"]._fingerprint
        for relative, expected in manifest["data_sha256"].items():
            assert hashlib.sha256((tmp_path / "wikitext103-gpt2-4" / relative).read_bytes()).hexdigest() == expected
    else:
        pretrained.assert_called_once_with("local-test")
        assert weights and "model_init" not in manifest and "data_sha256" not in manifest


def test_partial_accumulation_matches_whole_logical_batch():
    model = tiny_model().double().eval()
    clone = copy.deepcopy(model)
    ids = torch.randint(19, (5, 8))
    for chunk in ids.split(2):
        (lm_loss(model(input_ids=chunk).logits, chunk) * len(chunk) / len(ids)).backward()
    lm_loss(clone(input_ids=ids).logits, ids).backward()
    for p, q in zip(model.parameters(), clone.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-9)


def test_checkpoint_rng_and_duplicate_log_recovery(tmp_path):
    model = tiny_model().train()
    optimizer = OptimizerV2(model, "muon")
    ids = torch.randint(19, (1, 8))
    backward(model, ids)
    optimizer.step()
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, model, optimizer, {"step": 1, "epoch": 0, "offset": 8})
    rng = torch.random.get_rng_state().clone()
    backward(model, ids)
    optimizer.step()
    expected = copy.deepcopy(model.state_dict())
    restored = tiny_model().train()
    continuation = OptimizerV2(restored, "muon")
    progress = load_checkpoint(checkpoint, restored, continuation, torch.device("cpu"))
    assert progress["offset"] == 8
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    backward(restored, ids)
    continuation.step()
    assert_tree(restored.state_dict(), expected)
    log = tmp_path / "metrics.jsonl"
    log.write_text('{"step": 0}\n{"step": 1}\n{"step": 2}\n{"step":')
    trim_log(log, 1)
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [0, 1]


class TokenBlocks:
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, indices):
        return {"input_ids": [self.rows[index] for index in indices]}


def test_full_validation_weights_tail_and_does_not_wrap():
    model = tiny_model().eval()
    rows = torch.randint(19, (5, 8))
    dataset = TokenBlocks(rows.tolist())
    result = evaluate(model, dataset, 2, torch.device("cpu"))
    expected = float(lm_loss(model(input_ids=rows).logits, rows).detach())
    assert result["validation_nll"] == pytest.approx(expected, abs=1e-6)
    assert result["validation_prediction_tokens"] == 35


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
def test_tiny_gpu_epoch_runner_all_eight_methods(tmp_path):
    tiny_model().config.save_pretrained(tmp_path / "model")
    config = resolved_config(load_config(Path("configs/experiments/gpt2_v2_5090.yaml")))
    config["model"].update(local_path=str(tmp_path / "model"), sequence_length=8)
    config["training"].update(epochs=2, batch_size=2, gradient_accumulation_steps=2,
                              evaluation_batch_size=2, evaluation_interval=2, checkpoint_interval=2,
                              probe_sequence_length=8)
    train = TokenBlocks(torch.randint(19, (9, 8)).tolist())
    validation = TokenBlocks(torch.randint(19, (3, 8)).tolist())
    results = []
    for method in config["methods"]:
        if method["id"] not in ("adamw", "muon", "proposal_notch_v1"):
            method["options"]["interval"] = 1
        output = tmp_path / method["id"]
        result = run_one(config, method, 0, train, validation, output, {"requested": False})
        assert result["steps"] == 6 and result["epochs_completed"] == 2
        assert result["input_tokens"] == 9 * 8 * 2
        assert checkpoint_path(output).exists()
        assert result["fallback_events"] == 0
        results.append(result)
    assert len({result["initialization_hash"] for result in results}) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
def test_outer_runner_serializes_provenance_and_verifies_completed_resume(tmp_path):
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
        pytest.skip("requires the registered local RTX 5090")
    datasets = pytest.importorskip("datasets")
    assets = tmp_path / "assets"
    model_path, dataset_path = assets / "gpt2-model", assets / "tokens"
    tiny_model().config.save_pretrained(model_path)
    blocks = datasets.DatasetDict({
        split: datasets.Dataset.from_dict({"input_ids": torch.randint(19, (count, 8)).tolist()})
        for split, count in (("train", 5), ("validation", 3))
    })
    blocks.save_to_disk(dataset_path)
    loaded = datasets.load_from_disk(dataset_path)
    digests = {}
    for path in dataset_path.rglob("*.arrow"):
        with path.open("rb") as handle:
            digests[str(path.relative_to(dataset_path))] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"sequence_length": 8, "model_init": "random", "data_sha256": digests,
                "splits": {split: {"blocks": len(data), "fingerprint": data._fingerprint} for split, data in loaded.items()}}
    (assets / "manifest.json").write_text(json.dumps(manifest))
    config = resolved_config(load_config(Path("configs/experiments/gpt2_v2_5090.yaml")), ["adamw"], [0], 1)
    config["model"].update(local_path=str(model_path), sequence_length=8)
    config["dataset"]["local_path"] = str(dataset_path)
    config["training"].update(batch_size=2, gradient_accumulation_steps=2, evaluation_batch_size=2,
                              probe_sequence_length=8)
    output = tmp_path / "comparison"
    assert run(config, output) == 0
    assert (output / "gpt2_wikitext103_validation_nll_steps.png").is_file()
    assert (output / "gpt2_wikitext103_validation_nll_time.png").is_file()
    checkpoint = checkpoint_path(output / "runs/adamw/seed_0")
    modified = checkpoint.stat().st_mtime_ns
    assert run(config, output) == 0
    assert checkpoint.stat().st_mtime_ns == modified
    changed = copy.deepcopy(config)
    changed["training"]["max_steps"] = 2
    with pytest.raises(ValueError, match="resume configuration"):
        run(changed, output)
    with checkpoint.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        run(config, output)


def runner_case(tmp_path):
    tiny_model().config.save_pretrained(tmp_path / "model")
    config = resolved_config(load_config(Path("configs/experiments/gpt2_v2_5090.yaml")), ["adamw"])
    config["model"].update(local_path=str(tmp_path / "model"), sequence_length=8)
    config["training"].update(epochs=2, batch_size=2, gradient_accumulation_steps=2,
                              evaluation_batch_size=2, evaluation_interval=200,
                              checkpoint_interval=400, probe_sequence_length=8)
    return config, TokenBlocks(torch.randint(19, (9, 8)).tolist()), TokenBlocks(torch.randint(19, (3, 8)).tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
def test_nonfinite_microbatch_restores_committed_rng_and_exact_resume(tmp_path, monkeypatch):
    import gpt2_v2_experiment as runner
    config, train, validation = runner_case(tmp_path)
    method = config["methods"][0]
    original_loss = runner.lm_loss
    calls = 0
    def loss(logits, ids):
        nonlocal calls
        value = original_loss(logits, ids)
        if torch.is_grad_enabled():
            calls += 1
            if calls == 3:
                return value * float("nan")
        return value
    monkeypatch.setattr(runner, "lm_loss", loss)
    with pytest.raises(FloatingPointError):
        run_one(config, method, 0, train, validation, tmp_path / "failed", {"requested": False})
    failed = torch.load(checkpoint_path(tmp_path / "failed"), map_location="cpu", weights_only=False)
    assert failed["progress"]["step"] == 1
    monkeypatch.setattr(runner, "lm_loss", original_loss)
    capped = copy.deepcopy(config)
    capped["training"]["max_steps"] = 1
    run_one(capped, method, 0, train, validation, tmp_path / "one", {"requested": False})
    one = torch.load(checkpoint_path(tmp_path / "one"), map_location="cpu", weights_only=False)
    assert_tree(failed["model"], one["model"])
    assert_tree(failed["optimizer"]["adapter"], one["optimizer"]["adapter"])
    assert_tree(failed["torch_rng"], one["torch_rng"])
    assert_tree(failed["cuda_rng"], one["cuda_rng"])
    run_one(config, method, 0, train, validation, tmp_path / "failed", {"requested": False})
    run_one(config, method, 0, train, validation, tmp_path / "uninterrupted", {"requested": False})
    resumed = torch.load(checkpoint_path(tmp_path / "failed"), map_location="cpu", weights_only=False)
    control = torch.load(checkpoint_path(tmp_path / "uninterrupted"), map_location="cpu", weights_only=False)
    assert_tree(resumed["model"], control["model"], exact=False)
    assert_tree(resumed["optimizer"]["adapter"], control["optimizer"]["adapter"], exact=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires local CUDA")
def test_resume_retries_failed_final_validation_before_reporting_completion(tmp_path, monkeypatch):
    import gpt2_v2_experiment as runner
    config, train, validation = runner_case(tmp_path)
    method = config["methods"][0]
    original_evaluate = runner.evaluate
    calls = 0
    def evaluate(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise FloatingPointError("injected final validation failure")
        return original_evaluate(*args)
    monkeypatch.setattr(runner, "evaluate", evaluate)
    output = tmp_path / "failed_validation"
    with pytest.raises(FloatingPointError):
        run_one(config, method, 0, train, validation, output, {"requested": False})
    saved = torch.load(checkpoint_path(output), map_location="cpu", weights_only=False)
    assert saved["progress"]["step"] == 6 and saved["progress"]["last_evaluation_step"] == 3
    monkeypatch.setattr(runner, "evaluate", original_evaluate)
    result = run_one(config, method, 0, train, validation, output, {"requested": False})
    restored = torch.load(checkpoint_path(output), map_location="cpu", weights_only=False)
    assert restored["progress"]["last_evaluation_step"] == 6
    assert result["steps"] == 6
