"""Controlled GPT experiment for recycled low-rank Gauss--Newton solves."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Iterator

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plot
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from artifacts import write_metric
from gn_experiment import (
    LANGUAGE_MODEL_GN_TASK,
    _comparison_records,
    _read_metrics,
    artifact_paths,
)
from kronecker_ggn_common.curvature_operator import (
    FunctionalCurvatureBatch,
    GGNFullOperator,
)
from models import CONTEXT_LENGTH, parameter_count
from paper_full_gn_experiment import (
    _infinite_batches,
    _load_complete_warmup,
    common_warmup_path,
    language_model_curvature_batch,
    paper_gn_time_limit_reached,
)
from recycled_low_rank_gn import (
    FactoredKroneckerAccumulator,
    LowRankResidualCorrection,
    RecycledResidualState,
    build_factored_kronecker_diagonal,
    pcg_solve,
)
from training import _evaluate, _loaders, _model, configure_reproducibility


DEFAULT_OUTER_EFFECTIVE_BATCH_SIZE = 3_904
DEFAULT_CURVATURE_BATCH_SIZE = DEFAULT_OUTER_EFFECTIVE_BATCH_SIZE
DEFAULT_POST_WARMUP_SECONDS = 13_189.759907


def validate_kron_rpcg_contract(
    *,
    physical_batch_size: int,
    sequence_length: int,
    outer_effective_batch_size: int,
    curvature_batch_size: int,
) -> dict[str, int]:
    if physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be positive")
    if sequence_length != CONTEXT_LENGTH:
        raise ValueError(f"sequence_length must remain {CONTEXT_LENGTH}")
    if outer_effective_batch_size <= 0:
        raise ValueError("outer_effective_batch_size must be positive")
    if not 0 < curvature_batch_size <= outer_effective_batch_size:
        raise ValueError("curvature batch must be within the outer effective batch")
    return {
        "sequence_length": sequence_length,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation": math.ceil(
            outer_effective_batch_size / physical_batch_size
        ),
        "final_physical_batch_size": (
            outer_effective_batch_size % physical_batch_size or physical_batch_size
        ),
        "outer_effective_batch_size": outer_effective_batch_size,
        "outer_effective_tokens": outer_effective_batch_size * sequence_length,
        "curvature_batch_size": curvature_batch_size,
        "curvature_accumulation": math.ceil(
            curvature_batch_size / physical_batch_size
        ),
        "final_curvature_batch_size": (
            curvature_batch_size % physical_batch_size or physical_batch_size
        ),
        "line_search_effective_batch_size": outer_effective_batch_size,
    }


class SampleWeightedGGNOperator:
    """Accumulate mini-batch GGN products with exact per-sample weights."""

    def __init__(
        self,
        operators: tuple[GGNFullOperator, ...],
        *,
        sample_counts: tuple[int, ...],
    ) -> None:
        if not operators or len(operators) != len(sample_counts):
            raise ValueError("operators and sample_counts must be non-empty and aligned")
        if any(count <= 0 for count in sample_counts):
            raise ValueError("sample_counts must be positive")
        first = operators[0]
        parameter_ids = tuple(id(parameter) for parameter in first.parameters)
        if any(
            tuple(id(parameter) for parameter in operator.parameters) != parameter_ids
            for operator in operators[1:]
        ):
            raise ValueError("All weighted GGN operators must share parameters")
        self.operators = operators
        self.sample_counts = sample_counts
        self.total_samples = sum(sample_counts)
        self.parameters = first.parameters
        self._sizes = first._sizes

    @property
    def numel(self) -> int:
        return self.operators[0].numel

    def matvec(self, vector: Tensor) -> Tensor:
        total = torch.zeros_like(vector)
        for operator, count in zip(
            self.operators, self.sample_counts, strict=True
        ):
            total.add_(operator.matvec(vector).to(total), alpha=count)
        return total / self.total_samples


class ExactBatchStream:
    """Take an exact sample count while retaining a raw batch's unused rows."""

    def __init__(self, batches: Iterator[tuple[Tensor, Tensor]]) -> None:
        self.batches = batches
        self.pending: tuple[Tensor, Tensor] | None = None
        self.samples_consumed = 0
        self.raw_batches_consumed = 0

    def take(self, sample_count: int) -> tuple[tuple[Tensor, Tensor], ...]:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        selected: list[tuple[Tensor, Tensor]] = []
        remaining = sample_count
        while remaining:
            if self.pending is None:
                self.pending = next(self.batches)
                self.raw_batches_consumed += 1
            token_ids, targets = self.pending
            take = min(remaining, token_ids.shape[0])
            selected.append((token_ids[:take], targets[:take]))
            if take == token_ids.shape[0]:
                self.pending = None
            else:
                self.pending = (token_ids[take:], targets[take:])
            remaining -= take
            self.samples_consumed += take
        return tuple(selected)

    def skip(self, sample_count: int) -> None:
        while sample_count:
            take = min(sample_count, 1_024)
            self.take(take)
            sample_count -= take


def _trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def _flat_parameter_shapes(model: nn.Module) -> tuple[torch.Size, ...]:
    return tuple(parameter.shape for parameter in _trainable_parameters(model))


def _flatten_gradients(model: nn.Module) -> Tensor:
    values = []
    for parameter in _trainable_parameters(model):
        if parameter.grad is None:
            raise RuntimeError("Every trainable parameter must have an outer gradient")
        values.append(parameter.grad.detach().reshape(-1).float())
    return torch.cat(tuple(values))


def _restore_flat_parameters(
    parameters: tuple[nn.Parameter, ...], originals: tuple[Tensor, ...]
) -> None:
    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            parameter.copy_(original)


def _assign_flat_direction(
    parameters: tuple[nn.Parameter, ...],
    originals: tuple[Tensor, ...],
    direction: Tensor,
    step_size: float,
) -> None:
    offset = 0
    with torch.no_grad():
        for parameter, original in zip(parameters, originals, strict=True):
            size = parameter.numel()
            parameter.copy_(
                original
                + step_size
                * direction[offset : offset + size]
                .reshape_as(parameter)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            offset += size
    if offset != direction.numel():
        raise ValueError("direction has the wrong number of parameters")


def _average_true_loss(
    model: nn.Module, batches: tuple[FunctionalCurvatureBatch, ...]
) -> float:
    total = 0.0
    total_samples = sum(batch.args[0].shape[0] for batch in batches)
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = model(*batch.args, **batch.kwargs)
            total += float(batch.loss_fn(output).item()) * batch.args[0].shape[0]
    return total / total_samples


@dataclass(frozen=True)
class DirectionLineSearchResult:
    step_size: float
    loss: float
    candidate_losses: tuple[tuple[float, float], ...]
    screening_sequences: int
    finalist_sequences: int


def _batch_sample_count(batches: tuple[FunctionalCurvatureBatch, ...]) -> int:
    return sum(batch.args[0].shape[0] for batch in batches)


def held_out_direction_line_search(
    model: nn.Module,
    direction: Tensor,
    batches: tuple[FunctionalCurvatureBatch, ...],
    *,
    search_range: int,
    initial_step_scale: float = 1.0,
    screening_sequences: int | None = None,
    finalists: int = 2,
    include_zero_step: bool = False,
) -> DirectionLineSearchResult:
    """Apply the paper's held-out line search directly to one flat direction."""
    if not batches:
        raise ValueError("Held-out line search requires at least one batch")
    if search_range <= 0 or initial_step_scale <= 0:
        raise ValueError("line-search range and initial scale must be positive")
    if screening_sequences is not None and screening_sequences <= 0:
        raise ValueError("screening_sequences must be positive when provided")
    candidate_count = search_range + int(include_zero_step)
    if not 0 < finalists <= candidate_count:
        raise ValueError("finalists must lie within the candidate count")
    parameters = _trainable_parameters(model)
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    step_sizes = tuple(
        initial_step_scale / math.sqrt(2.0) ** index
        for index in range(search_range)
    )
    if include_zero_step:
        step_sizes += (0.0,)
    total_sequences = _batch_sample_count(batches)
    use_progressive = (
        screening_sequences is not None and screening_sequences < total_sequences
    )
    if use_progressive:
        split = 0
        selected = 0
        while split < len(batches) and selected < screening_sequences:
            selected += batches[split].args[0].shape[0]
            split += 1
        screen_batches = batches[:split]
        remaining_batches = batches[split:]
    else:
        screen_batches = batches
        remaining_batches = ()
    screen_count = _batch_sample_count(screen_batches)
    losses: list[tuple[float, float]] = []
    try:
        for step_size in step_sizes:
            _assign_flat_direction(parameters, originals, direction, step_size)
            losses.append((step_size, _average_true_loss(model, screen_batches)))
        if use_progressive:
            finalist_indices = sorted(
                range(len(losses)), key=lambda index: losses[index][1]
            )[:finalists]
            remaining_count = _batch_sample_count(remaining_batches)
            for index in finalist_indices:
                step_size, screen_loss = losses[index]
                _assign_flat_direction(parameters, originals, direction, step_size)
                remaining_loss = _average_true_loss(model, remaining_batches)
                full_loss = (
                    screen_count * screen_loss + remaining_count * remaining_loss
                ) / total_sequences
                losses[index] = (step_size, full_loss)
            best_index = min(finalist_indices, key=lambda index: losses[index][1])
            best_step_size, best_loss = losses[best_index]
        else:
            best_step_size, best_loss = min(losses, key=lambda item: item[1])
        _assign_flat_direction(parameters, originals, direction, best_step_size)
    except BaseException:
        _restore_flat_parameters(parameters, originals)
        raise
    return DirectionLineSearchResult(
        best_step_size,
        best_loss,
        tuple(losses),
        screen_count,
        total_sequences,
    )


def _correction_payload(state: RecycledResidualState) -> dict | None:
    if state.correction is None:
        return None
    return {
        "basis": state.correction.basis.detach().cpu(),
        "projected_matrix": state.correction.projected_matrix.detach().cpu(),
        "minimum_relative_eigenvalue": state.correction.minimum_relative_eigenvalue,
    }


def save_kron_rpcg_checkpoint(
    path: Path,
    model: nn.Module,
    recycled_state: RecycledResidualState,
    *,
    completed_outer_steps: int,
    consumed_training_batches: int,
    consumed_training_sequences: int | None = None,
    elapsed_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "completed_outer_steps": completed_outer_steps,
            "consumed_training_batches": consumed_training_batches,
            "consumed_training_sequences": (
                consumed_training_batches
                if consumed_training_sequences is None
                else consumed_training_sequences
            ),
            "elapsed_seconds": elapsed_seconds,
            "recycled_built_at_step": recycled_state.built_at_step,
            "recycled_correction": _correction_payload(recycled_state),
        },
        path,
    )


def load_kron_rpcg_checkpoint(
    path: Path,
    model: nn.Module,
    recycled_state: RecycledResidualState,
) -> tuple[int, int, float]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    correction = payload.get("recycled_correction")
    recycled_state.built_at_step = payload.get("recycled_built_at_step")
    if correction is None:
        recycled_state.correction = None
    else:
        device = next(model.parameters()).device
        recycled_state.correction = LowRankResidualCorrection(
            basis=correction["basis"].to(device),
            projected_matrix=correction["projected_matrix"].to(device),
            minimum_relative_eigenvalue=float(
                correction["minimum_relative_eigenvalue"]
            ),
        )
    return (
        int(payload["completed_outer_steps"]),
        int(
            payload.get(
                "consumed_training_sequences", payload["consumed_training_batches"]
            )
        ),
        float(payload["elapsed_seconds"]),
    )


def _outer_gradient_and_curvature_batches(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], ...],
    *,
    curvature_batch_size: int,
    device: torch.device,
    collect_second_moment: bool = True,
) -> tuple[Tensor, tuple[FunctionalCurvatureBatch, ...], Tensor | None]:
    model.train()
    model.zero_grad(set_to_none=True)
    curvature_batches: list[FunctionalCurvatureBatch] = []
    curvature_samples = 0
    parameters = _trainable_parameters(model)
    statistics = (
        FactoredKroneckerAccumulator(
            parameter_shapes=tuple(parameter.shape for parameter in parameters),
            device=device,
            dtype=torch.float32,
        )
        if collect_second_moment
        else None
    )
    current_batch_weight = [0.0]
    handles = []
    for index, parameter in enumerate(parameters) if statistics is not None else ():

        def accumulate(gradient: Tensor, block_index: int = index) -> Tensor:
            statistics.update_scaled_block(
                block_index,
                gradient,
                batch_weight=current_batch_weight[0],
            )
            return gradient

        handles.append(parameter.register_hook(accumulate))
    total_samples = sum(token_ids.shape[0] for token_ids, _ in batches)
    if total_samples <= 0:
        raise ValueError("outer gradient batches must be non-empty")
    try:
        for token_ids, targets in batches:
            token_ids = token_ids.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            remaining = curvature_batch_size - curvature_samples
            selected = min(token_ids.shape[0], max(remaining, 0))
            if selected:
                curvature_batches.append(
                    language_model_curvature_batch(
                        (token_ids[:selected], targets[:selected]), device
                    )
                )
                curvature_samples += selected
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(token_ids)
                logits = logits.logits if hasattr(logits, "logits") else logits
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            current_batch_weight[0] = token_ids.shape[0] / total_samples
            (loss * current_batch_weight[0]).backward()
    finally:
        for handle in handles:
            handle.remove()
    if curvature_samples != curvature_batch_size:
        raise RuntimeError("Failed to collect the requested curvature subset")
    gradient = _flatten_gradients(model)
    second_moment_diagonal = (
        statistics.diagonal() if statistics is not None else None
    )
    model.zero_grad(set_to_none=True)
    return gradient, tuple(curvature_batches), second_moment_diagonal


def _held_out_batches(
    batches: tuple[tuple[Tensor, Tensor], ...],
    *,
    device: torch.device,
) -> tuple[FunctionalCurvatureBatch, ...]:
    return tuple(
        language_model_curvature_batch(batch, device) for batch in batches
    )


def write_kron_rpcg_comparison_plots(
    root: Path,
    *,
    run_label: str,
    paper_full_gn_run_label: str,
) -> tuple[Path, Path] | None:
    traces: list[tuple[str, list[dict]]] = []
    for label, optimizer, trace_label in (
        ("AdamW", "adamw", None),
        ("Muon", "muon", None),
        ("Paper Full GN", "paper_full_gn", paper_full_gn_run_label),
    ):
        path = artifact_paths(root, optimizer, run_label=trace_label).metric
        if path.exists():
            traces.append(
                (label, _comparison_records(root, optimizer, run_label=trace_label))
            )
    candidate_path = artifact_paths(root, "kron_rpcg", run_label=run_label).metric
    if candidate_path.exists():
        traces.append(("Recycled low-rank GN", _read_metrics(candidate_path)))
    if len(traces) < 4:
        return None
    outputs = (
        root / "results" / "nlp" / f"{run_label}_metric_steps.png",
        root / "results" / "nlp" / f"{run_label}_metric_time.png",
    )
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    for output, key, axis_label in (
        (outputs[0], "step", "Completed optimizer step / outer step"),
        (outputs[1], "elapsed_seconds", "Wall-clock time (hours)"),
    ):
        figure, axis = plot.subplots(figsize=(9, 5))
        for label, records in traces:
            values = [record for record in records if key in record]
            x_values = [
                record[key] / 3600 if key == "elapsed_seconds" else record[key]
                for record in values
            ]
            axis.plot(x_values, [record["metric"] for record in values], label=label)
        if key == "step":
            axis.set_xscale("log")
        axis.set(
            xlabel=axis_label,
            ylabel="Validation next-token accuracy",
            title="Recycled low-rank GN versus controlled optimizer baselines",
        )
        axis.grid(alpha=0.2)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plot.close(figure)
    return outputs


def run_kron_rpcg_trial(
    root: Path,
    *,
    physical_batch_size: int = 4,
    sequence_length: int = CONTEXT_LENGTH,
    outer_effective_batch_size: int = DEFAULT_OUTER_EFFECTIVE_BATCH_SIZE,
    curvature_batch_size: int = DEFAULT_CURVATURE_BATCH_SIZE,
    damping: float = 0.03,
    correction_rank: int = 4,
    correction_refresh_interval: int = 4,
    minimum_relative_eigenvalue: float = 0.25,
    maximum_pcg_iterations: int = 6,
    relative_pcg_tolerance: float = 0.05,
    solver_mode: str = "secant_direct",
    line_search_range: int = 5,
    initial_step_scale: float = 1.0,
    line_search_screening_sequences: int | None = None,
    line_search_finalists: int = 2,
    preconditioner_statistic: str = "mean_gradient_square",
    maximum_outer_steps: int = 10_000,
    maximum_seconds: float = DEFAULT_POST_WARMUP_SECONDS,
    workers: int = 4,
    seed: int = 1337,
    label: str = "kron_rpcg_r4_c64_pcg6_d003",
    paper_full_gn_label: str = "paper_full_gn_template_n122_b32_lr0003",
    basis_dtype: torch.dtype = torch.bfloat16,
    fresh: bool = False,
) -> dict:
    contract = validate_kron_rpcg_contract(
        physical_batch_size=physical_batch_size,
        sequence_length=sequence_length,
        outer_effective_batch_size=outer_effective_batch_size,
        curvature_batch_size=curvature_batch_size,
    )
    if damping <= 0:
        raise ValueError("damping must be positive")
    if correction_rank <= 0 or correction_refresh_interval <= 0:
        raise ValueError("correction rank and refresh interval must be positive")
    if not 0 < minimum_relative_eigenvalue <= 1:
        raise ValueError("minimum_relative_eigenvalue must lie in (0, 1]")
    if maximum_pcg_iterations <= 0 or relative_pcg_tolerance <= 0:
        raise ValueError("PCG controls must be positive")
    if solver_mode not in {"pcg", "secant_direct"}:
        raise ValueError("solver_mode must be 'pcg' or 'secant_direct'")
    if line_search_range <= 0 or initial_step_scale <= 0 or maximum_seconds <= 0:
        raise ValueError("line search, initial scale, and time limit must be positive")
    if (
        line_search_screening_sequences is not None
        and line_search_screening_sequences <= 0
    ):
        raise ValueError("line_search_screening_sequences must be positive")
    if not 0 < line_search_finalists <= line_search_range:
        raise ValueError("line_search_finalists must lie within the candidate count")
    if preconditioner_statistic not in {
        "mean_gradient_square",
        "batch_second_moment",
    }:
        raise ValueError("Unsupported preconditioner statistic")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Kron-RPCG experiment")

    configure_reproducibility(seed)
    device = torch.device("cuda")
    task = replace(
        LANGUAGE_MODEL_GN_TASK,
        micro_batch_size=physical_batch_size,
        gradient_accumulation=1,
    )
    train_loader, validation_loader = _loaders(task, root, workers, seed + 17)
    model = _model(task).to(device)
    warmup = _load_complete_warmup(common_warmup_path(root), model)
    paths = artifact_paths(root, "kron_rpcg", run_label=label)
    if fresh:
        paths.metric.unlink(missing_ok=True)
        paths.result.unlink(missing_ok=True)
        paths.checkpoint.unlink(missing_ok=True)
    recycled_state = RecycledResidualState(
        rank=correction_rank,
        refresh_interval=correction_refresh_interval,
        minimum_relative_eigenvalue=minimum_relative_eigenvalue,
        storage_dtype=basis_dtype,
    )
    completed_outer_steps = 0
    consumed_training_sequences = 0
    elapsed_seconds = 0.0
    if paths.checkpoint.exists():
        completed_outer_steps, consumed_training_sequences, elapsed_seconds = (
            load_kron_rpcg_checkpoint(paths.checkpoint, model, recycled_state)
        )
    elif paths.metric.exists():
        paths.metric.unlink()

    existing_metrics = (
        _read_metrics(paths.metric) if paths.metric.exists() else []
    )

    stream = ExactBatchStream(_infinite_batches(train_loader))
    if consumed_training_sequences:
        stream.skip(consumed_training_sequences)
    parameter_shapes = _flat_parameter_shapes(model)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_averaged_matvecs = sum(
        int(record.get("averaged_curvature_matvecs", 0))
        for record in existing_metrics
    )
    final_metric = (
        float(existing_metrics[-1]["metric"]) if existing_metrics else None
    )
    while completed_outer_steps < maximum_outer_steps:
        outer_step_started = time.perf_counter()
        gradient, curvature_batches, second_moment_diagonal = (
            _outer_gradient_and_curvature_batches(
                model,
                stream.take(outer_effective_batch_size),
                curvature_batch_size=curvature_batch_size,
                device=device,
                collect_second_moment=(
                    preconditioner_statistic == "batch_second_moment"
                ),
            )
        )
        gradient_seconds = time.perf_counter() - outer_step_started
        print(
            f"task={task.identifier} optimizer=kron_rpcg "
            f"step={completed_outer_steps + 1} phase=gradient "
            f"seconds={gradient_seconds:.3f}",
            flush=True,
        )
        consumed_training_sequences += outer_effective_batch_size
        curvature_sample_counts = tuple(
            batch.args[0].shape[0] for batch in curvature_batches
        )
        curvature_operator = SampleWeightedGGNOperator(
            tuple(GGNFullOperator(model, batch) for batch in curvature_batches),
            sample_counts=curvature_sample_counts,
        )
        torch.cuda.empty_cache()
        averaged_matvecs = 0

        def curvature_matvec(vector: Tensor) -> Tensor:
            nonlocal averaged_matvecs
            averaged_matvecs += 1
            return curvature_operator.matvec(vector)

        accumulated_diagonal = (
            second_moment_diagonal
            if preconditioner_statistic == "batch_second_moment"
            else None
        )
        if solver_mode == "secant_direct":
            seed = gradient / gradient.norm().clamp_min(
                torch.finfo(gradient.dtype).eps
            )
            curvature_seed = curvature_matvec(seed)
            base_diagonal = build_factored_kronecker_diagonal(
                gradient,
                parameter_shapes=parameter_shapes,
                curvature_matvec=curvature_matvec,
                curvature_seed=curvature_seed,
                damping=damping,
                accumulated_second_moment_diagonal=accumulated_diagonal,
            )
            correction, solver_relative_residual = (
                recycled_state.update_from_secant(
                    outer_step=completed_outer_steps,
                    base_diagonal=base_diagonal,
                    seed=seed,
                    system_seed=curvature_seed + damping * seed,
                )
            )
            direction = correction.apply(-gradient, base_diagonal)
            pcg_iterations = 0
            pcg_relative_residual = None
            refreshed = True
            del seed, curvature_seed
        else:
            base_diagonal = build_factored_kronecker_diagonal(
                gradient,
                parameter_shapes=parameter_shapes,
                curvature_matvec=curvature_matvec,
                damping=damping,
                accumulated_second_moment_diagonal=accumulated_diagonal,
            )

            def system_matvec(vector: Tensor) -> Tensor:
                return curvature_matvec(vector) + damping * vector

            correction, refreshed = recycled_state.get_or_build(
                outer_step=completed_outer_steps,
                system_matvec=system_matvec,
                base_diagonal=base_diagonal,
                seed=-gradient,
            )

            def preconditioner(vector: Tensor) -> Tensor:
                return correction.apply(vector, base_diagonal)

            solve = pcg_solve(
                system_matvec,
                -gradient,
                preconditioner=preconditioner,
                maximum_iterations=maximum_pcg_iterations,
                relative_tolerance=relative_pcg_tolerance,
            )
            direction = solve.direction
            pcg_iterations = solve.iterations
            pcg_relative_residual = solve.relative_residual
            solver_relative_residual = solve.relative_residual
            del solve, preconditioner, system_matvec
        solver_seconds = time.perf_counter() - outer_step_started - gradient_seconds
        print(
            f"task={task.identifier} optimizer=kron_rpcg "
            f"step={completed_outer_steps + 1} phase=solve "
            f"seconds={solver_seconds:.3f} residual={solver_relative_residual:.6g} "
            f"average_gv={averaged_matvecs}",
            flush=True,
        )
        actual_correction_rank = correction.rank
        if not torch.isfinite(direction).all() or torch.dot(gradient, direction) >= 0:
            raise RuntimeError("Kron-RPCG failed to produce a finite descent direction")
        # These callables close over the full curvature operator and the large
        # base diagonal.  Keeping them until the next loop iteration leaked
        # enough memory to OOM on step three even though step one fitted.
        del (
            curvature_matvec,
            curvature_operator,
            curvature_batches,
            base_diagonal,
            second_moment_diagonal,
            gradient,
        )
        held_out_batches = _held_out_batches(
            stream.take(outer_effective_batch_size),
            device=device,
        )
        consumed_training_sequences += outer_effective_batch_size
        line_search = held_out_direction_line_search(
            model,
            direction,
            held_out_batches,
            search_range=line_search_range,
            initial_step_scale=initial_step_scale,
            screening_sequences=line_search_screening_sequences,
            finalists=line_search_finalists,
            include_zero_step=(solver_mode == "secant_direct"),
        )
        line_search_seconds = (
            time.perf_counter()
            - outer_step_started
            - gradient_seconds
            - solver_seconds
        )
        del held_out_batches, direction
        completed_outer_steps += 1
        total_averaged_matvecs += averaged_matvecs
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        final_metric = _evaluate(task, model, validation_loader, device)
        evaluation_seconds = time.perf_counter() - started
        elapsed_seconds += time.perf_counter() - started
        started = time.perf_counter()
        timed_out = paper_gn_time_limit_reached(elapsed_seconds, maximum_seconds)
        metric_record = {
            "step": completed_outer_steps,
            "metric": final_metric,
            "elapsed_seconds": float(warmup["elapsed_seconds"]) + elapsed_seconds,
            "post_warmup_elapsed_seconds": elapsed_seconds,
            "line_search_step_size": line_search.step_size,
            "initial_step_scale": initial_step_scale,
            "line_search_loss": line_search.loss,
            "line_search_screening_sequences": line_search.screening_sequences,
            "line_search_finalist_sequences": line_search.finalist_sequences,
            "line_search_finalists": line_search_finalists,
            "line_search_candidate_losses": [
                {"step_size": step_size, "loss": loss}
                for step_size, loss in line_search.candidate_losses
            ],
            "damping": damping,
            "preconditioner_statistic": preconditioner_statistic,
            "solver_mode": solver_mode,
            "correction_rank": actual_correction_rank,
            "minimum_relative_eigenvalue": minimum_relative_eigenvalue,
            "correction_refreshed": refreshed,
            "pcg_iterations": pcg_iterations,
            "pcg_relative_residual": pcg_relative_residual,
            "solver_relative_residual": solver_relative_residual,
            "averaged_curvature_matvecs": averaged_matvecs,
            "sample_curvature_matvecs": averaged_matvecs * curvature_batch_size,
            "curvature_physical_batches": len(curvature_sample_counts),
            "gradient_seconds": gradient_seconds,
            "solver_seconds": solver_seconds,
            "line_search_seconds": line_search_seconds,
            "evaluation_seconds": evaluation_seconds,
            **contract,
        }
        write_metric(paths.metric, metric_record)
        save_kron_rpcg_checkpoint(
            paths.checkpoint,
            model,
            recycled_state,
            completed_outer_steps=completed_outer_steps,
            consumed_training_batches=stream.raw_batches_consumed,
            consumed_training_sequences=consumed_training_sequences,
            elapsed_seconds=elapsed_seconds,
        )
        write_kron_rpcg_comparison_plots(
            root,
            run_label=label,
            paper_full_gn_run_label=paper_full_gn_label,
        )
        print(
            f"task={task.identifier} optimizer=kron_rpcg "
            f"step={completed_outer_steps} metric={final_metric:.6f} "
            f"line_search_step={line_search.step_size:.6g} "
            f"solver_residual={solver_relative_residual:.6g} "
            f"average_gv={averaged_matvecs} refreshed={refreshed}",
            flush=True,
        )
        del line_search, correction
        torch.cuda.empty_cache()
        if timed_out:
            break

    status = (
        "completed"
        if completed_outer_steps >= maximum_outer_steps
        else "time_limit_checkpointed"
    )
    result = {
        "task": task.identifier,
        "domain": task.domain,
        "model": task.model,
        "optimizer": "kron_rpcg",
        "parameters": parameter_count(model),
        "warmup_tokens": int(warmup["processed_tokens"]),
        "damping": damping,
        "correction_rank": correction_rank,
        "correction_refresh_interval": correction_refresh_interval,
        "minimum_relative_eigenvalue": minimum_relative_eigenvalue,
        "maximum_pcg_iterations": maximum_pcg_iterations,
        "relative_pcg_tolerance": relative_pcg_tolerance,
        "solver_mode": solver_mode,
        "line_search_range": line_search_range,
        "initial_step_scale": initial_step_scale,
        "line_search_screening_sequences": line_search_screening_sequences,
        "line_search_finalists": line_search_finalists,
        "preconditioner_statistic": preconditioner_statistic,
        "basis_dtype": str(basis_dtype),
        **contract,
        "completed_outer_steps": completed_outer_steps,
        "consumed_training_batches": stream.raw_batches_consumed,
        "consumed_training_sequences": consumed_training_sequences,
        "averaged_curvature_matvecs": total_averaged_matvecs,
        "sample_curvature_matvecs": total_averaged_matvecs * curvature_batch_size,
        "final_metric": final_metric,
        "seconds": float(warmup["elapsed_seconds"]) + elapsed_seconds,
        "post_warmup_seconds": elapsed_seconds,
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "status": status,
    }
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
