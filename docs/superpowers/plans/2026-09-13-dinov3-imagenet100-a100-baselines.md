# DINOv3 ImageNet-100 A100 Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish reproducible five-epoch DINOv3 ViT-B ImageNet-100 classification baselines for AdamW, Muon, and two-rate Muown on A100 accelerators.

**Architecture:** Add a cached Hugging Face ImageNet-100 loader and a dedicated DINOv3 CV runner while preserving historical CIFAR-100 evidence. Extend the shared training and artifact records with true completed optimizer steps and elapsed timestamps, then restrict Muon and Muown to trainable interior DINOv3 blocks 8–10; all boundary layers use AdamW.

**Tech Stack:** Python 3, PyTorch, torchvision, Hugging Face `datasets`, Transformers DINOv3 ViT-B, matplotlib, pytest, `nohup`, SSH, Kubernetes.

**Spec:** `docs/superpowers/specs/2026-09-13-dinov3-imagenet100-a100-baselines-design.md`

## Global Constraints

- Model: DINOv3 ViT-B, 85,737,316 parameters, 100-class classifier, frozen embeddings and blocks 0–7.
- Data: `clane9/imagenet-100`, 126,689 train and 5,000 validation examples, cached only below `.cache/cv/imagenet100`.
- Train five final epochs with complete 5,000-image top-1 validation after each epoch; target at most two hours per cached final run and reject an out-of-memory batch before launch.
- Keep patch projection, final transformer block, final normalization, classifier, and all one-dimensional parameters in AdamW. Muon/Muown receive only trainable matrix weights of blocks 8–10.
- Muown uses distinct positive `direction_lr` and `gain_lr`; no shared-rate fallback in the CV runner.
- Metrics retain `epoch`, `step`, `elapsed_seconds`, and `metric`; PNG horizontal axes must use recorded steps/times, not epoch labels or interpolated times.
- Checkpoints are below `.cache/cv/checkpoints`; source and historical CIFAR-100 artifacts are not overwritten.
- Do not stage or commit unrelated dirty-worktree changes.

---

### Task 1: Add the cached ImageNet-100 data contract

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data_cache.py`

**Interfaces:**
- Produces `dinov3_imagenet100_loaders(root: Path, batch_size: int, workers: int, seed: int = 1337) -> tuple[DataLoader, DataLoader]`.
- Produces an `ImageNet100TransformDataset` that exposes `__len__` and returns `(Tensor[3,224,224], int)`.
- Consumes by `training._loaders` in Task 3.

- [ ] **Step 1: Write failing loader tests**

```python
def test_imagenet100_transform_dataset_preserves_image_and_label_contract(monkeypatch, tmp_path):
    import data
    from PIL import Image

    class Split:
        features = {"label": type("Labels", (), {"names": ["zero", "one"]})()}
        def __len__(self): return 2
        def __getitem__(self, index): return {"image": Image.new("RGB", (240, 260)), "label": index}

    dataset = data.ImageNet100TransformDataset(Split(), data._dinov3_validation_transform())
    image, label = dataset[1]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label == 1


def test_dinov3_imagenet100_loader_uses_project_cache(monkeypatch, tmp_path):
    import data
    calls = []
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: calls.append((args, kwargs)) or {"train": [], "validation": []})
    data.dinov3_imagenet100_loaders(tmp_path, batch_size=2, workers=0)
    assert calls[0][0] == ("clane9/imagenet-100",)
    assert calls[0][1]["cache_dir"] == str(tmp_path / ".cache/cv/imagenet100/hf")
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_data_cache.py -k imagenet100 -v`

Expected: FAIL because `ImageNet100TransformDataset` and `dinov3_imagenet100_loaders` do not exist.

- [ ] **Step 3: Implement the minimum loader**

```python
class ImageNet100TransformDataset(Dataset):
    def __init__(self, split, transform) -> None:
        self.split, self.transform = split, transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.split[index]
        return self.transform(row["image"].convert("RGB")), int(row["label"])


def dinov3_imagenet100_loaders(root: Path, batch_size: int, workers: int, seed: int = 1337):
    from datasets import load_dataset
    cache = root / ".cache/cv/imagenet100/hf"
    splits = load_dataset("clane9/imagenet-100", cache_dir=str(cache))
    return (
        DataLoader(ImageNet100TransformDataset(splits["train"], _dinov3_train_transform()), shuffle=True, **loader_options(batch_size, workers, seed)),
        DataLoader(ImageNet100TransformDataset(splits["validation"], _dinov3_validation_transform()), shuffle=False, **loader_options(batch_size, workers, seed + 1)),
    )
```

Keep `_dinov3_train_transform` and `_dinov3_validation_transform` as small helpers using ImageNet mean/std and the existing 224-pixel transforms.

- [ ] **Step 4: Run loader tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_data_cache.py -k 'imagenet100 or dinov3_cifar100' -v`

Expected: PASS.

- [ ] **Step 5: Commit only this data slice**

```bash
git add src/data.py tests/test_data_cache.py
git commit -m "feat: add cached DINOv3 ImageNet-100 loader"
```

### Task 2: Enforce DINOv3 boundary-layer optimizer routing and two-rate Muown

**Files:**
- Modify: `src/optimizers.py`
- Modify: `tests/test_muown.py`
- Modify: `tests/test_baseline_contract.py`

**Interfaces:**
- `muon_parameter_names(model: nn.Module) -> set[str]` returns only `backbone.layer.8`, `.9`, and `.10` matrix weights for DINOv3.
- `Muown(params: Iterable[nn.Parameter], direction_lr: float, gain_lr: float, weight_decay: float, momentum: float = 0.95)` stores both rates in its parameter group.
- `build_optimizers(model: nn.Module, optimizer: str, lr: float, weight_decay: float, auxiliary_lr: float = 3e-4, muown_direction_lr: float | None = None, muown_gain_lr: float | None = None)` passes rates to `Muown` and leaves boundary tensors in `adamw_aux`.

- [ ] **Step 1: Write failing routing and two-rate tests**

```python
def test_dinov3_routes_only_trainable_interior_matrices_to_muon():
    names = muon_parameter_names(create_cv_model("dinov3_vitb16"))
    assert "backbone.layer.8.attention.q_proj.weight" in names
    assert "backbone.layer.10.mlp.up_proj.weight" in names
    assert not any(name.startswith("backbone.layer.11.") for name in names)
    assert "backbone.embeddings.patch_embeddings.projection.weight" not in names
    assert "classifier.weight" not in names


def test_muown_uses_distinct_direction_and_gain_learning_rates():
    parameter = nn.Parameter(torch.eye(2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = Muown([parameter], direction_lr=0.2, gain_lr=0.01, weight_decay=0.0, ns_steps=1)
    optimizer.step()
    assert optimizer.param_groups[0]["direction_lr"] == 0.2
    assert optimizer.param_groups[0]["gain_lr"] == 0.01
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_muown.py tests/test_baseline_contract.py -k 'two_rate or interior_matrices' -v`

Expected: FAIL because the last DINOv3 block is currently selected and `Muown` accepts one `lr`.

- [ ] **Step 3: Implement routing and separate rates**

```python
def _dino_interior_matrix_names(model: nn.Module) -> set[str]:
    names = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.ndim < 2:
            continue
        if any(name.startswith(f"backbone.layer.{index}.") for index in (8, 9, 10)):
            names.add(name)
    return names
```

Detect DINOv3 by the presence of `backbone.layer` and `backbone.embeddings`; return this set before generic selection. In `Muown.step`, scale the orthogonalized direction with `direction_lr` and the Adam row-gain update with `gain_lr`. Keep `lr` as a backward-compatible alias only for legacy callers; the new CV path must validate both explicit rates.

- [ ] **Step 4: Run unit tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_muown.py tests/test_baseline_contract.py -v`

Expected: PASS, including existing generic Muown compatibility coverage.

- [ ] **Step 5: Commit only the routing slice**

```bash
git add src/optimizers.py tests/test_muown.py tests/test_baseline_contract.py
git commit -m "feat: route DINO boundary layers through AdamW"
```

### Task 3: Record genuine step/time metrics and add the ImageNet-100 task

**Files:**
- Modify: `src/config.py`
- Modify: `src/training.py`
- Modify: `src/artifacts.py`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_baseline_contract.py`

**Interfaces:**
- New formal task identifier: `cv_dinov3_vitb16_imagenet100`, model `dinov3_vitb16_imagenet100`, five epochs, full validation.
- `write_metric_plot` reads `record["step"]` when present and labels the x-axis `Completed optimizer step`.
- `write_metric_time_plot` reads `record["elapsed_seconds"]` when present and labels the x-axis `Wall-clock time (minutes)`.
- `run_trial` writes `{"epoch", "step", "elapsed_seconds", "metric"}` at every completed epoch.

- [ ] **Step 1: Write failing metric tests**

```python
def test_metric_plot_uses_recorded_optimizer_steps(tmp_path):
    for name in ("adamw", "muon"):
        (tmp_path / f"{name}.jsonl").write_text('{"epoch": 1, "step": 491, "elapsed_seconds": 11.5, "metric": 0.6}\n')
    output = write_metric_plot(tmp_path / "adamw.jsonl", tmp_path / "muon.jsonl", tmp_path / "steps.png", "Accuracy")
    assert output.exists() and output.stat().st_size > 0


def test_metric_time_plot_uses_recorded_elapsed_seconds(tmp_path):
    import artifacts
    observed = {}
    class Axis:
        def plot(self, x, y, label): observed[label] = list(x)
        def set(self, **_): pass
        def legend(self): pass
    class Figure:
        def tight_layout(self): pass
        def savefig(self, output, dpi): output.write_bytes(b"png")
    for name, seconds in (("adamw", 12.0), ("muon", 24.0)):
        (tmp_path / f"{name}.jsonl").write_text(json.dumps({"epoch": 1, "step": 491, "elapsed_seconds": seconds, "metric": 0.6}) + "\\n")
    monkeypatch.setattr(artifacts.plot, "subplots", lambda **_: (Figure(), Axis()))
    write_metric_time_plot(tmp_path / "adamw.jsonl", tmp_path / "muon.jsonl", tmp_path / "time.png", "Accuracy", {"AdamW": 999.0, "Muon": 999.0})
    assert observed == {"AdamW": [0.2], "Muon": [0.4]}
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_artifacts.py -v`

Expected: FAIL after asserting the new axis/record behavior because the current code reads `epoch` and interpolates time.

- [ ] **Step 3: Implement metric and task changes**

Add `evaluation_batches: int | None = 64` to `FormalTask`; set it to `None` for the new ImageNet-100 task and use it in `_evaluate`. Route `dinov3_vitb16_imagenet100` to the new loader. Track `completed_steps`, incrementing only when all accumulated gradients are stepped, and append elapsed time from a single run origin. Preserve fallback epoch/interpolated behavior in artifact writers for historical JSONL files that lack the new keys.

- [ ] **Step 4: Run regression tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_artifacts.py tests/test_baseline_contract.py tests/test_data_cache.py -v`

Expected: PASS; the legacy CIFAR task continues to load and historical plots remain renderable.

- [ ] **Step 5: Commit only the metric/task slice**

```bash
git add src/config.py src/training.py src/artifacts.py tests/test_artifacts.py tests/test_baseline_contract.py
git commit -m "feat: record DINO optimizer steps and timestamps"
```

### Task 4: Add a dedicated CV runner and three-way final renderer

**Files:**
- Create: `src/run_cv_dinov3.py`
- Create: `src/cv_dinov3_experiment.py`
- Create: `tests/test_cv_dinov3_experiment.py`

**Interfaces:**
- CLI: `run_cv_dinov3.py --optimizer {adamw,muon,muown} --label NAME --epochs N --micro-batch-size N --gradient-accumulation N`.
- Muon CLI values: `--matrix-learning-rate`, `--auxiliary-learning-rate`.
- Muown CLI values: `--direction-learning-rate`, `--gain-learning-rate`, `--auxiliary-learning-rate`.
- Renderer: `write_cv_dinov3_baseline_plots(root: Path, label: str) -> tuple[Path, Path]` writes the two spec-named PNGs after three completed final result files exist.

- [ ] **Step 1: Write failing CLI and plot tests**

```python
def test_cv_runner_requires_two_muown_rates(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_cv_dinov3.py", "--optimizer", "muown", "--label", "screen"])
    with pytest.raises(SystemExit):
        run_cv_dinov3.main()


def test_cv_renderer_writes_three_way_step_and_time_pngs(tmp_path):
    for optimizer in ("adamw", "muon", "muown"):
        write_metric(tmp_path / f"metrics/cv/cv_dinov3_vitb16_imagenet100__final__{optimizer}.jsonl", {"epoch": 1, "step": 491, "elapsed_seconds": 12.0, "metric": 0.5})
        (tmp_path / f"results/cv/cv_dinov3_vitb16_imagenet100__final__{optimizer}.json").write_text('{"seconds": 12.0}')
    assert all(path.exists() for path in write_cv_dinov3_baseline_plots(tmp_path, "final"))
```

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_cv_dinov3_experiment.py -v`

Expected: FAIL because the runner and renderer do not exist.

- [ ] **Step 3: Implement the dedicated runner**

Use `argparse` to validate positive rates and forbid omitted Muown direction/gain rates. Build a `FormalTask` replacement containing CLI batch settings, call `run_trial`, and write the combined plots only if all three matching final metrics/results exist. Do not modify `src/run.py`; its old task remains historical.

- [ ] **Step 4: Run dedicated and full unit tests**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_cv_dinov3_experiment.py tests/test_muown.py tests/test_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit only the runner slice**

```bash
git add src/run_cv_dinov3.py src/cv_dinov3_experiment.py tests/test_cv_dinov3_experiment.py
git commit -m "feat: add DINOv3 CV baseline runner"
```

### Task 5: Verify locally, synchronize, profile, tune, and launch A100 trials

**Files:**
- Create: `scripts/run_aba_dinov3_imagenet100.sh`
- Create: `records/2026-09-13-dinov3-imagenet100-a100-baselines.md`
- Modify: `README.md` only after all three final figures exist

**Interfaces:**
- Script accepts an optimizer label and launches exactly one uniquely labelled `nohup` trial into `.cache/cv/checkpoints`.
- Record reports model/data size, selected batch, screen settings, accelerator, runtime, validation accuracy, and artifact paths.

- [ ] **Step 1: Write the preflight assertions into a runnable script**

```bash
set -euo pipefail
test -f .cache/huggingface/models/DINOv3-ViT-B/model.safetensors
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
PYTHONPATH=src .venv/bin/python -c 'from models import create_cv_model; assert sum(p.numel() for p in create_cv_model("dinov3_vitb16").parameters()) == 85737316'
```

- [ ] **Step 2: Run a local syntax and unit-test gate**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_data_cache.py tests/test_muown.py tests/test_baseline_contract.py tests/test_artifacts.py tests/test_cv_dinov3_experiment.py -v`

Expected: PASS.

- [ ] **Step 3: Synchronize only required source to ABA and stage data there**

```bash
rsync -a --relative src/data.py src/optimizers.py src/config.py src/training.py src/artifacts.py src/run_cv_dinov3.py src/cv_dinov3_experiment.py scripts/run_aba_dinov3_imagenet100.sh tests/test_data_cache.py tests/test_muown.py tests/test_baseline_contract.py tests/test_artifacts.py tests/test_cv_dinov3_experiment.py ABA:/home/yufan/New_Optimizer/
ssh ABA 'cd /home/yufan/New_Optimizer && .venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"'
```

Download `clane9/imagenet-100` through the new loader on ABA into `/home/yufan/New_Optimizer/.cache/cv/imagenet100/hf`; verify both splits and the 100-label feature before profiling.

- [ ] **Step 4: Establish the largest safe A100 microbatch**

Try microbatches 256, 128, then 64 in that order using one forward/backward/update with block 8–11 trainable. Select the first finite setting that leaves at least 10 GiB free; set gradient accumulation so all three methods use the same effective batch. Delete temporary profile checkpoints after recording the result.

- [ ] **Step 5: Run independent aggressive one-epoch screens**

Launch AdamW and Muon on ABA GPU 0 and GPU 1 with unique labels. Launch Muown on Nautilus when its one-A100 pod is Running. Screen rate candidates separately: AdamW `1e-4, 3e-4, 1e-3`; Muon matrix `3e-4, 1e-3, 3e-3` with auxiliary `1e-4, 3e-4`; Muown `(direction,gain)` pairs `(3e-4,1e-4)`, `(1e-3,1e-4)`, `(1e-3,3e-4)`, with auxiliary `1e-4`. Do not launch a candidate that shares a checkpoint label with another run.

- [ ] **Step 6: Select winners and launch three final five-epoch trials**

Select only finite screen traces by full-validation top-1 accuracy. Run each winner with `nohup`, record `nvidia-smi` ownership and log path, and monitor every completed epoch. If Nautilus remains pending, wait until either it schedules or one ABA final run completes; then launch Muown on the released A100 and label it delayed.

- [ ] **Step 7: Verify and publish evidence**

Check each final JSON says `status: completed`, has five full-validation records containing nondecreasing `step` and elapsed timestamps, and total runtime no longer than two hours. Render the required step/time PNGs, inspect them visually, write the record, add the final figure to `README.md`, then commit and push only the CV code, record, metrics, results, and README changes.

- [ ] **Step 8: Final artifact test and commit**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_cv_dinov3_experiment.py tests/test_artifacts.py -v`

Expected: PASS, followed by:

```bash
git status --short src/data.py src/optimizers.py src/config.py src/training.py src/artifacts.py src/run_cv_dinov3.py src/cv_dinov3_experiment.py scripts/run_aba_dinov3_imagenet100.sh tests/test_data_cache.py tests/test_muown.py tests/test_baseline_contract.py tests/test_artifacts.py tests/test_cv_dinov3_experiment.py records/2026-09-13-dinov3-imagenet100-a100-baselines.md results/cv metrics/cv README.md
git add src/data.py src/optimizers.py src/config.py src/training.py src/artifacts.py src/run_cv_dinov3.py src/cv_dinov3_experiment.py scripts/run_aba_dinov3_imagenet100.sh tests/test_data_cache.py tests/test_muown.py tests/test_baseline_contract.py tests/test_artifacts.py tests/test_cv_dinov3_experiment.py records/2026-09-13-dinov3-imagenet100-a100-baselines.md results/cv/cv_dinov3_vitb16_imagenet100_baselines_metric_steps.png results/cv/cv_dinov3_vitb16_imagenet100_baselines_metric_time.png README.md
git commit -m "results: add DINOv3 ImageNet-100 A100 baselines"
git push origin main
```
