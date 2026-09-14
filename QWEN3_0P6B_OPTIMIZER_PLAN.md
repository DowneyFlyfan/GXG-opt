# Qwen3-0.6B Optimizer Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Qwen3-0.6B continued-pretraining benchmark with matched AdamW, Muon, Muown, and each active optimizer proposal.

**Architecture:** A Qwen-specific token-cache/data module produces deterministic fixed-width blocks and a manifest.  A compact causal-language-model runner owns model loading, routing, checkpointing, metrics, and PNG rendering.  Proposal adapters consume one non-mutating baseline proposal interface so their safety tests are separable from the training harness.

**Tech Stack:** Python, PyTorch, Transformers, Datasets streaming, NumPy memory maps, matplotlib, pytest, SSH to ABA A100.

**Spec:** `QWEN3_0P6B_OPTIMIZER_DESIGN.md`

## Global Constraints

- Use `Qwen/Qwen3-0.6B`, bfloat16, Qwen tokenizer, 2,048-token blocks, and FineWeb-Edu `sample-10BT`.
- Store datasets, Hugging Face cache, checkpoints, and model snapshots only under `.cache/`.
- Write 2,000,000,000 train and 100,000,000 validation `uint32` tokens, discard partial blocks, and record digests.
- Three formal epochs, same initialization, schedule, precision, deterministic stream, and effective token batch for all baselines.
- Route Qwen interior blocks 1–26 two-dimensional linears to Muon/Muown; route embeddings, head, boundary blocks, and one-dimensional tensors to AdamW.
- Tune AdamW/Muon learning rates and Muown direction/gain learning rates under equal screen budgets; record decay choices.
- All completed final candidates get validation-perplexity-versus-step and versus-time PNGs containing AdamW, Muon, and Muown.
- Preserve unrelated dirty files and use explicit-path commits only.

---

### Task 1: deterministic Qwen token cache

**Files:**
- Create: `src/qwen3_data.py`
- Create: `tests/test_qwen3_data.py`

**Interfaces:**
- Produces `QwenTokenCache`, `prepare_qwen_fineweb_cache`, and `qwen_block_loaders`.
- `prepare_qwen_fineweb_cache(root: Path, *, train_tokens: int, validation_tokens: int, sequence_length: int, source) -> QwenTokenCache` returns paths and a manifest dictionary.
- `qwen_block_loaders(cache: QwenTokenCache, *, micro_batch_size: int, workers: int, seed: int) -> tuple[DataLoader, DataLoader]` returns shifted fixed-width batches.

- [ ] **Step 1: Write failing tests**

```python
def test_prepare_cache_writes_disjoint_uint32_streams_and_manifest(tmp_path):
    cache = prepare_qwen_fineweb_cache(tmp_path, train_tokens=16, validation_tokens=8,
        sequence_length=4, source=[("train", "a", [1, 2]), ("validation", "b", [3, 4])])
    assert cache.train_path.stat().st_size == 16 * 4
    assert cache.manifest["written_tokens"] == {"train": 16, "validation": 8}

def test_block_loader_returns_shifted_fixed_width_tokens(tmp_path):
    cache = write_tiny_cache(tmp_path, train=list(range(17)), validation=list(range(9)), sequence_length=4)
    inputs, labels = next(iter(qwen_block_loaders(cache, micro_batch_size=1, workers=0, seed=1)[0]))
    assert inputs.tolist() == [[0, 1, 2, 3]]
    assert labels.tolist() == [[1, 2, 3, 4]]
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_data.py -v`

Expected: import failure because `qwen3_data` does not exist.

- [ ] **Step 3: Implement the smallest cache and loader**

Implement a `uint32` NumPy memory-map writer that appends end-of-sequence tokens, allocates only rounded full-block targets, and calculates SHA-256 while writing.  Keep the source iterator injectable for the tests; production source streams FineWeb-Edu through `datasets.load_dataset(..., streaming=True)`.  Persist a JSON manifest atomically after both files verify.

- [ ] **Step 4: Run focused and existing cache tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_data.py tests/test_data_cache.py -v`

Expected: PASS.

- [ ] **Step 5: Commit only this task**

Run: `git add src/qwen3_data.py tests/test_qwen3_data.py && git commit -m "feat: add deterministic Qwen token cache"`

### Task 2: Qwen model loading and optimizer routing

**Files:**
- Create: `src/qwen3_model.py`
- Modify: `src/optimizers.py`
- Create: `tests/test_qwen3_model.py`

**Interfaces:**
- Produces `load_qwen3_model(root: Path) -> PreTrainedModel` and `qwen_muon_parameter_names(model) -> set[str]`.
- `build_optimizers(..., parameter_selector=qwen_muon_parameter_names)` accepts an optional selector and preserves legacy behavior when omitted.

- [ ] **Step 1: Write failing routing tests**

```python
def test_qwen_routing_uses_only_interior_linear_matrices():
    model = tiny_qwen_with_28_layers(tied_embeddings=True)
    selected = qwen_muon_parameter_names(model)
    assert "model.layers.1.self_attn.q_proj.weight" in selected
    assert "model.layers.0.self_attn.q_proj.weight" not in selected
    assert "model.layers.27.mlp.down_proj.weight" not in selected
    assert all("embed_tokens" not in name and "lm_head" not in name for name in selected)

def test_muown_preserves_independent_direction_and_gain_rates_for_qwen():
    optimizers = build_qwen_optimizers(tiny_qwen_with_28_layers(), "muown", direction_lr=0.02, gain_lr=0.001)
    assert optimizers["muown"].param_groups[0]["direction_lr"] == 0.02
    assert optimizers["muown"].param_groups[0]["gain_lr"] == 0.001
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_model.py -v`

Expected: import failure because Qwen helpers are absent.

- [ ] **Step 3: Implement model and routing helpers**

Load only from `.cache/huggingface/models/Qwen3-0.6B` with `local_files_only=True`; emit a clear missing-cache error.  Select named two-dimensional projection weights in layers 1–26, exclude tied embeddings/head and boundary layers by exact name, and route every complement parameter to AdamW.  Reuse existing Muon/Muown classes without changing their numerical update.

- [ ] **Step 4: Run routing regression suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_model.py tests/test_muown.py tests/test_baseline_contract.py -v`

Expected: PASS.

- [ ] **Step 5: Commit only this task**

Run: `git add src/qwen3_model.py src/optimizers.py tests/test_qwen3_model.py && git commit -m "feat: route Qwen matrices to Muon optimizers"`

### Task 3: matched Qwen perplexity runner

**Files:**
- Create: `src/qwen3_ppl_experiment.py`
- Create: `src/run_qwen3_ppl.py`
- Create: `tests/test_qwen3_ppl_experiment.py`

**Interfaces:**
- Produces `QwenTrialConfig`, `run_qwen_trial(config: QwenTrialConfig) -> dict`, and `render_qwen_comparison(root: Path, labels: list[str]) -> tuple[Path, Path]`.
- Checkpoint payload includes model, every optimizer/scheduler state, epoch/update/token cursors, elapsed seconds, data-manifest digest, and route checksum.

- [ ] **Step 1: Write failing runner tests**

```python
def test_qwen_paths_keep_checkpoints_under_project_cache(tmp_path):
    paths = qwen_trial_paths(tmp_path, "muown", "screen")
    assert paths.checkpoint.parent == tmp_path / ".cache" / "qwen3_0p6b" / "checkpoints"

def test_renderer_uses_perplexity_and_completed_optimizer_steps(tmp_path):
    write_metric_records(tmp_path, labels=("AdamW", "Muon", "Muown"), values=(3.0, 2.5, 2.0))
    step_png, time_png = render_qwen_comparison(tmp_path, ["AdamW", "Muon", "Muown"])
    assert step_png.is_file() and time_png.is_file()
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_ppl_experiment.py -v`

Expected: import failure because the Qwen runner is absent.

- [ ] **Step 3: Implement the runner**

Implement autocast training, loss on shifted labels, accumulation, one optimizer step per matched global batch, periodic fixed-validation perplexity, resumable cursors, and metrics with `step`, `elapsed_seconds`, and `token_exposure`.  The command-line script supports prepare, admission, screen, formal, resume, and render modes.  It rejects a cache manifest mismatch rather than resuming on altered data.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_ppl_experiment.py tests/test_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit only this task**

Run: `git add src/qwen3_ppl_experiment.py src/run_qwen3_ppl.py tests/test_qwen3_ppl_experiment.py && git commit -m "feat: add matched Qwen perplexity runner"`

### Task 4: baseline admission, tuning, and formal execution

**Files:**
- Create: `records/2026-09-13_qwen3_0p6b_baseline_protocol.md`
- Create after completion: `results/nlp/qwen3_0p6b_baselines_metric_steps.png`
- Create after completion: `results/nlp/qwen3_0p6b_baselines_metric_time.png`

**Interfaces:**
- Consumes `run_qwen3_ppl.py` and a verified cache manifest.
- Produces three baseline result JSON documents, JSONL traces, cache checkpoints, two comparison PNGs, and one record.

- [ ] **Step 1: Write the protocol record before the run**

Record host/GPU identity, model and tokenizer revision, data-manifest digest, sequence length, microbatch admission result, effective batch, fixed seed, screen token budget, candidate grids, formal three-epoch budget, and success/failure criteria.

- [ ] **Step 2: Synchronize tested local code to ABA and inspect A100 ownership**

Run: `ssh ABA 'nvidia-smi; cd /home/yufan/New_Optimizer && df -h .'`

Expected: selected device is idle or has only an explicitly identified disposable filler; preserve every other process.

- [ ] **Step 3: Build cache and run admission**

Run the prepare mode with `nohup` on ABA, verify manifest digests and token counts, remove temporary source cache, then run a 20-step memory admission for each optimizer route.  Choose the largest microbatch whose peak allocation remains below 90% of device memory and use accumulation to keep one shared global batch.

- [ ] **Step 4: Run equal-budget screens**

Run AdamW and Muon in parallel on the two A100 devices, then Muown.  Screen AdamW/Muon learning rates and Muown `(direction_lr, gain_lr)` pairs for the identical token budget.  Reject nonfinite candidates.  Record all trial data but do not plot screens.

- [ ] **Step 5: Run and monitor three matched formal baselines**

Launch one `nohup` job per A100 with distinct labels and cache checkpoint paths.  Monitor process command, checkpoint modification time, last JSONL record, and `nvidia-smi` utilization.  Do not alter a live job except for a logged failure.

- [ ] **Step 6: Render, validate, record, commit, and push**

Verify three equal token exposures/epochs and the exact route checksum.  Render the two required PNGs and commit only baseline code-independent evidence plus the record.  Run `git push`; on non-fast-forward, report it and leave history unchanged.

### Task 5: proposal adapter and shared safety protocol

**Files:**
- Create: `src/qwen3_proposals.py`
- Create: `tests/test_qwen3_proposals.py`
- Create: `records/2026-09-13_qwen3_0p6b_proposal_protocol.md`

**Interfaces:**
- Produces `BaselineProposal`, `ProposalAdapter`, `ProposalEvent`, and `run_proposal_screen`.
- `ProposalAdapter.propose(parameters, gradients, state) -> BaselineProposal` never mutates model parameters or state.
- `ProposalAdapter.commit(proposal: BaselineProposal) -> None` commits each base optimizer state once.

- [ ] **Step 1: Write failing adapter tests**

```python
def test_disabled_adapter_matches_one_baseline_step_bitwise():
    baseline, adapted = matching_tiny_models()
    one_muon_step(baseline)
    run_adapter_step(adapted, strength=0.0)
    assert_state_and_parameters_equal(baseline, adapted)

def test_failed_proposal_solve_commits_the_unmodified_baseline_once():
    result = run_adapter_step(matching_tiny_models()[1], force_solver_failure=True)
    assert result.used_baseline_fallback and result.committed_states == 1
```

- [ ] **Step 2: Run and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_proposals.py -v`

Expected: import failure because the common proposal adapter is absent.

- [ ] **Step 3: Implement the adapter**

Expose baseline learning increments separately from decay increments, retain a deterministic event RNG state, validate every proposal before commit, and write per-event diagnostics.  Keep it independent of the four mathematical implementations.

- [ ] **Step 4: Run tests and commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_proposals.py tests/test_muown.py -v`

Then: `git add src/qwen3_proposals.py tests/test_qwen3_proposals.py records/2026-09-13_qwen3_0p6b_proposal_protocol.md && git commit -m "feat: add Qwen proposal safety adapter"`

### Task 6: routing-resistance proposal

**Files:**
- Create: `src/qwen3_routing_resistance.py`
- Create: `tests/test_qwen3_routing_resistance.py`

**Interfaces:**
- Produces `sample_unordered_edges`, `weighted_factor_gram`, and `filter_qk_proposal`.
- Consumes one selected Qwen query/key head, FP32 attention probabilities, and a non-mutating Muon proposal.

- [ ] **Step 1: Write failing equation tests**

```python
def test_resistance_pair_probabilities_sum_to_one():
    probabilities = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    assert sum(edge_sampling_probability(probabilities, *edge, epsilon=0.05) for edge in combinations(range(3), 2)) == pytest.approx(1.0)

def test_factorized_woodbury_filter_matches_dense_reference():
    assert torch.allclose(filter_qk_proposal(tiny_factors, proposal, rho=1.0), dense_filter(tiny_factors, proposal, rho=1.0), atol=1e-10)
```

- [ ] **Step 2: Run failure, implement, and pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_routing_resistance.py -v`

Implement the exact resistance-mixture sampler and FP64 Gram solve required by idea 2.  Then rerun the command until PASS.

- [ ] **Step 3: Screen, formalize if competitive, and commit evidence**

Run a 100-step safety screen, then equal-budget learning-rate screen.  Run three epochs only if diagnostics are finite and screen perplexity is competitive with Muon.  Render final graphs only for the formal candidate; commit record and exact evidence paths.

### Task 7: tied-path curvature proposal

**Files:**
- Create: `src/qwen3_tied_path_curvature.py`
- Create: `tests/test_qwen3_tied_path_curvature.py`

**Interfaces:**
- Produces `paired_tied_path_sketch` and `filter_tied_embedding_proposal`.
- Consumes the one shared Qwen embedding/head parameter and creates one physical state/update.

- [ ] **Step 1: Write failing joint-path tests**

```python
def test_paired_sketch_keeps_the_cross_path_term():
    joint = paired_tied_path_sketch(tiny_tied_model(), probes=2)
    assert not torch.allclose(joint.metric, joint.input_metric + joint.output_metric)

def test_tied_parameter_has_one_optimizer_state_after_commit():
    result = commit_tied_proposal(tiny_tied_model())
    assert result.physical_parameter_count == 1 and result.optimizer_state_count == 1
```

- [ ] **Step 2: Run failure, implement, and pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_tied_path_curvature.py -v`

Implement paired probes and the small positive-semidefinite solve, then rerun until PASS.

- [ ] **Step 3: Run safety/screen/formal protocol and commit evidence**

Use the exact Task 6 stage gates and record probe count, solve residual, overhead, and whether the joint cross term is active.

### Task 8: resonance-selective and feature-remapping proposals

**Files:**
- Create: `src/qwen3_resonance_filter.py`
- Create: `src/qwen3_feature_remap.py`
- Create: `tests/test_qwen3_resonance_filter.py`
- Create: `tests/test_qwen3_feature_remap.py`

**Interfaces:**
- Produces `filter_resonant_proposal` and `remap_feature_momentum`.
- Both consume `BaselineProposal` and return a validated proposal without committing optimizer state.

- [ ] **Step 1: Write failing causal-history and disabled-equivalence tests**

```python
def test_resonance_filter_uses_only_prior_events():
    state = ResonanceState.empty()
    assert filter_resonant_proposal(proposal, state).history_length == 0

def test_feature_remap_strength_zero_matches_muon():
    assert_proposals_equal(remap_feature_momentum(proposal, strength=0.0), proposal)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_qwen3_resonance_filter.py tests/test_qwen3_feature_remap.py -v`

Expected: import failures because both proposal implementations are absent.

- [ ] **Step 3: Implement and pass focused tests**

Implement each proposal exactly from its active idea document, including persistent state serialization, disabled direct bypass, and nonfinite fallback to baseline.  Rerun the focused suite until PASS.

- [ ] **Step 4: Stage-gated experiments, reporting, commits, and pushes**

For each independent method run the Task 6 safety/screen/formal protocol.  After each completed experiment, write a detailed `records/` report, create required graphs for formal runs, commit explicit paths, and attempt one push without force or history rewrite.

## Plan self-review

- Spec coverage: Tasks 1–4 implement cache, fair baselines, tuning, execution, records, and figures; Tasks 5–8 implement the four active proposals independently with safety gates.
- No placeholder scan: no unresolved placeholders or deferred undefined work remains; candidates that fail safety are explicitly reported as failed rather than silently replaced.
- Interface consistency: all proposal tasks consume `BaselineProposal` from Task 5 and all formal tasks consume `run_qwen_trial` from Task 3.
