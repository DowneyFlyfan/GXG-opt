"""Temporary split-leaf forwards for Qwen's physically tied embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.func import functional_call

from optimizer_v2.linalg import low_rank_prox


@dataclass(frozen=True)
class QwenAdamWProposal:
    """One tied-embedding AdamW proposal before an optional learning correction."""

    learning: torch.Tensor
    decay: torch.Tensor
    value: torch.Tensor
    next_state: dict[str, torch.Tensor | int]


class QwenTiedAdamWProposalAdapter:
    """Compute and commit the pinned tied-embedding AdamW state once."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        beta1, beta2 = betas
        if (
            parameter.ndim != 2
            or learning_rate <= 0
            or weight_decay < 0
            or not 0 <= beta1 < 1
            or not 0 <= beta2 < 1
            or eps <= 0
        ):
            raise ValueError("invalid tied AdamW proposal configuration")
        self.parameter = parameter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state: dict[str, torch.Tensor | int] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.parameter.grad is not None:
            if set_to_none:
                self.parameter.grad = None
            else:
                self.parameter.grad.zero_()

    @torch.no_grad()
    def propose(self) -> QwenAdamWProposal:
        """Return a standard AdamW increment without mutating parameter/state."""
        gradient = self.parameter.grad
        if gradient is None:
            raise RuntimeError("missing tied embedding gradient")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite tied embedding gradient")
        beta1, beta2 = self.betas
        step = int(self.state.get("step", 0)) + 1
        previous_average = self.state.get("exp_avg")
        previous_square = self.state.get("exp_avg_sq")
        exp_avg = torch.zeros_like(gradient) if previous_average is None else previous_average.clone()
        exp_avg_sq = torch.zeros_like(gradient) if previous_square is None else previous_square.clone()
        exp_avg.lerp_(gradient, 1 - beta1)
        exp_avg_sq.lerp_(gradient.square(), 1 - beta2)
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(self.eps)
        learning = torch.zeros_like(self.parameter).addcdiv_(
            exp_avg, denominator, value=-self.learning_rate / bias_correction1
        )
        decay = self.parameter.detach().mul(-self.learning_rate * self.weight_decay)
        value = self.parameter.detach().clone()
        value.mul_(1 - self.learning_rate * self.weight_decay)
        value.add_(learning)
        return QwenAdamWProposal(
            learning=learning,
            decay=decay,
            value=value,
            next_state={"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq},
        )

    @torch.no_grad()
    def commit(self, proposal: QwenAdamWProposal, correction: torch.Tensor | None = None) -> None:
        """Commit one proposal and optionally replace part of its learning increment."""
        if correction is not None and (
            correction.shape != self.parameter.shape or not torch.isfinite(correction).all()
        ):
            raise ValueError("invalid tied embedding learning correction")
        self.parameter.copy_(proposal.value)
        if correction is not None:
            self.parameter.add_(correction)
        self.state = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in proposal.next_state.items()
        }


def split_qwen_tied_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    embedding_input: torch.Tensor,
    embedding_output: torch.Tensor,
) -> torch.Tensor:
    """Evaluate Qwen with separate differentiable input/output embedding leaves.

    ``tie_weights=False`` is intentional: this functional call is solely an
    auxiliary differentiation device.  The trained Qwen model retains one
    physical tied parameter and receives one committed optimizer update.
    """
    if embedding_input.shape != embedding_output.shape:
        raise ValueError("split embedding leaves must have the same shape")
    parameters = {name: parameter.detach() for name, parameter in model.named_parameters()}
    parameters["model.embed_tokens.weight"] = embedding_input
    parameters["lm_head.weight"] = embedding_output
    output = functional_call(
        model,
        parameters,
        (),
        {"input_ids": input_ids, "use_cache": False},
        tie_weights=False,
    )
    logits = getattr(output, "logits", output)
    if not isinstance(logits, torch.Tensor):
        raise TypeError("split Qwen forward did not return logits")
    return logits


def qwen_paired_embedding_sketch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[dict[str, float]]]:
    """Build the paired categorical-probe sketch for Qwen's tied embedding.

    The same probe is differentiated through the input and output paths.  The
    resulting columns therefore retain the cross terms of the joint output-loss
    generalized Gauss--Newton metric and are stored in FP32 for the small
    proximal solve.
    """
    if count <= 0 or input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("a paired sketch needs positive count and two-token sequences")
    embedding = model.model.embed_tokens.weight
    input_leaf = embedding.detach().clone().requires_grad_()
    output_leaf = embedding.detach().clone().requires_grad_()
    logits = split_qwen_tied_forward(model, input_ids, input_leaf, output_leaf)[:, :-1]
    probabilities = logits.detach().float().softmax(dim=-1)
    flat = probabilities.reshape(-1, probabilities.shape[-1])
    columns: list[torch.Tensor] = []
    diagnostics: list[dict[str, float]] = []
    for index in range(count):
        choices = torch.multinomial(flat.cpu(), 1, generator=generator).to(flat.device)
        probe = -flat.clone()
        probe.scatter_add_(1, choices, torch.ones_like(choices, dtype=probe.dtype))
        probe = (probe / math.sqrt(len(flat))).reshape_as(logits).to(logits.dtype)
        left, right = torch.autograd.grad(
            (logits * probe).sum(),
            (input_leaf, output_leaf),
            retain_graph=index + 1 < count,
        )
        columns.append((left.detach().float() + right.detach().float()) / math.sqrt(count))
        diagnostics.append(
            {
                "input_norm": float(left.norm()),
                "output_norm": float(right.norm()),
                "paired_inner_product": float((left * right).sum()),
            }
        )
    return columns, diagnostics


@torch.no_grad()
def qwen_tied_proximal_correction(
    learning_increment: torch.Tensor,
    columns: list[torch.Tensor],
    *,
    rho: float,
) -> tuple[torch.Tensor, dict]:
    """Return the joint-sketch proximal correction before embedding decay.

    The direct zero-strength branch is intentional: it preserves the baseline
    floating-point trajectory rather than merely relying on a zero Woodbury
    coefficient after extra operations.
    """
    if rho < 0 or not columns or any(column.shape != learning_increment.shape for column in columns):
        raise ValueError("invalid tied-path proximal inputs")
    if rho == 0:
        return torch.zeros_like(learning_increment), {"kappa": 0.0, "gram_eigenvalues": []}
    filtered, diagnostics = low_rank_prox(learning_increment.float(), columns, rho)
    return (filtered - learning_increment.float()).to(learning_increment.dtype), diagnostics


class QwenTiedPathOptimizer:
    """Joint tied-path curvature filter for Qwen's one physical embedding.

    The controller owns the tied embedding's AdamW state.  It keeps all other
    AdamW auxiliary parameters in their ordinary optimizer group, so a tied
    alias cannot receive two updates or two moment estimates.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        parameter: torch.nn.Parameter,
        *,
        learning_rate: float,
        weight_decay: float,
        rho: float = 1.0,
        probes: int = 2,
        interval: int = 16,
        max_age: int = 16,
        seed: int = 0,
    ) -> None:
        if rho < 0 or probes <= 0 or interval <= 0 or max_age <= 0:
            raise ValueError("invalid tied-path curvature configuration")
        embedding = getattr(getattr(model, "model", None), "embed_tokens", None)
        if embedding is None or embedding.weight is not parameter:
            raise ValueError("tied-path optimizer must own Qwen's physical embedding parameter")
        self.model = model
        self.adapter = QwenTiedAdamWProposalAdapter(
            parameter, learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.rho = rho
        self.probes = probes
        self.interval = interval
        self.max_age = max_age
        self.seed = seed
        self.steps = 0
        self.columns: list[torch.Tensor] | None = None
        self.cache_step: int | None = None
        self._batch: torch.Tensor | None = None
        self.last_diagnostics: dict = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adapter.zero_grad(set_to_none=set_to_none)

    def prepare_batch(self, input_ids: torch.Tensor) -> None:
        """Retain one training sequence only when a fresh joint sketch is due."""
        self._batch = None
        if self.rho == 0 or self.steps % self.interval:
            return
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
            raise ValueError("tied-path probes require a nonempty two-token training sequence")
        self._batch = input_ids[:1].detach().to(self.adapter.parameter.device)

    def _generator(self) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(self.seed + 104_729 * (self.steps + 1))

    def step(self) -> None:
        """Refresh an optional paired sketch, then commit exactly one AdamW proposal."""
        proposal = self.adapter.propose()
        diagnostics: dict = {"step": self.steps + 1, "disabled": self.rho == 0}
        correction: torch.Tensor | None = None
        try:
            if self.rho != 0:
                if self.steps % self.interval == 0:
                    if self._batch is None:
                        diagnostics["refresh_failure"] = "missing_training_sequence"
                    else:
                        try:
                            columns, path_diagnostics = qwen_paired_embedding_sketch(
                                self.model, self._batch, count=self.probes, generator=self._generator()
                            )
                            if not all(torch.isfinite(column).all() for column in columns):
                                raise FloatingPointError("nonfinite tied-path sketch")
                            self.columns = columns
                            self.cache_step = self.steps
                            diagnostics["refresh"] = True
                            diagnostics["paths"] = path_diagnostics
                        except (RuntimeError, FloatingPointError, torch.linalg.LinAlgError) as error:
                            diagnostics["refresh_failure"] = str(error)
                if self.columns is not None and self.cache_step is not None:
                    age = self.steps - self.cache_step
                    diagnostics["metric_age"] = age
                    if age < self.max_age:
                        correction, metric = qwen_tied_proximal_correction(
                            proposal.learning, self.columns, rho=self.rho
                        )
                        diagnostics.update(metric)
                    else:
                        diagnostics["stale_cache_bypass"] = True
                else:
                    diagnostics["stale_cache_bypass"] = True
            self.adapter.commit(proposal, correction)
            self.steps += 1
            self.last_diagnostics = diagnostics
        finally:
            self._batch = None

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "steps": self.steps,
            "rho": self.rho,
            "probes": self.probes,
            "interval": self.interval,
            "max_age": self.max_age,
            "seed": self.seed,
            "columns": self.columns,
            "cache_step": self.cache_step,
            "adapter_state": self.adapter.state,
        }

    def load_state_dict(self, saved: dict) -> None:
        expected = self.state_dict()
        for key in ("version", "rho", "probes", "interval", "max_age", "seed"):
            if saved.get(key) != expected[key]:
                raise ValueError("tied-path optimizer configuration changed")
        self.steps = int(saved["steps"])
        self.columns = saved["columns"]
        self.cache_step = saved["cache_step"]
        self.adapter.state = saved["adapter_state"]
