from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, jvp, vjp

from .layer_registry import LayerRegistry


class GGNOperatorError(RuntimeError):
    pass


@contextmanager
def _preserve_execution_state(model: nn.Module):
    modes = [(module, module.training) for module in model.modules()]
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    python_rng = random.getstate()
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.train(training)
        with torch.no_grad():
            current_buffers = dict(model.named_buffers())
            for name, value in buffers.items():
                current_buffers[name].copy_(value)
        random.setstate(python_rng)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)


@dataclass(frozen=True)
class FunctionalCurvatureBatch:
    args: tuple[Any, ...]
    loss_fn: Any
    kwargs: dict[str, Any] = field(default_factory=dict)
    batch_id: str | None = None
    output_hvp_fn: Any | None = None


def _tree_is_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (tuple, list)):
        return all(_tree_is_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_tree_is_finite(item) for item in value.values())
    return True


class GGNLinearOperator:
    """Matrix-free layer-local J^T H_loss J for registered nn.Linear weights."""

    def __init__(
        self, model: nn.Module, registry: LayerRegistry, batch: FunctionalCurvatureBatch
    ) -> None:
        self.model = model
        self.registry = registry
        self.batch = batch
        self.matvec_count = 0
        self._parameter_names = {
            id(parameter): name for name, parameter in model.named_parameters()
        }

    def _functions(self, layer_id: str):
        layer = self.registry.by_id(layer_id)
        if not layer.supported:
            raise GGNOperatorError(
                f"Unsupported layer {layer_id}: {layer.fallback_reason}"
            )
        parameter_name = self._parameter_names[id(layer.weight)]
        state = dict(self.model.named_parameters())
        state.update(
            {name: value.detach().clone() for name, value in self.model.named_buffers()}
        )
        base = state[parameter_name]

        def layer_function(weight: Tensor) -> Any:
            candidate = dict(state)
            candidate[parameter_name] = weight
            return functional_call(
                self.model,
                candidate,
                self.batch.args,
                dict(self.batch.kwargs),
                strict=False,
            )

        return layer, base, layer_function

    def matvec(self, layer_id: str, vector: Tensor) -> Tensor:
        layer, base, layer_function = self._functions(layer_id)
        if tuple(vector.shape) != layer.matrix_shape:
            raise ValueError(
                f"Vector for {layer_id} must have shape {layer.matrix_shape}"
            )
        tangent = vector.to(device=base.device, dtype=base.dtype)
        with _preserve_execution_state(self.model):
            try:
                output, pullback = vjp(layer_function, base)
                loss = self.batch.loss_fn(output)
                output_gradient_fn = grad(self.batch.loss_fn)
                output_gradient = output_gradient_fn(output)
                if not torch.isfinite(loss).all() or not _tree_is_finite(
                    output_gradient
                ):
                    raise GGNOperatorError(
                        "Curvature loss or output gradient is non-finite"
                    )
                _, output_tangent = jvp(layer_function, (base,), (tangent,))
                _, output_hvp = jvp(output_gradient_fn, (output,), (output_tangent,))
                product = pullback(output_hvp)[0]
            except RuntimeError as error:
                raise GGNOperatorError(
                    "Exact GGN functional transforms failed; stateful/in-place modules are unsupported"
                ) from error
        dtype = torch.float64 if product.dtype == torch.float64 else torch.float32
        result = (
            product.detach()
            .to(device=vector.device, dtype=dtype)
            .reshape(layer.matrix_shape)
        )
        self.matvec_count += 1
        if not torch.isfinite(result).all():
            raise GGNOperatorError("GGN product is non-finite")
        return result.to(dtype=vector.dtype)

    def double_autograd_matvec(self, layer_id: str, vector: Tensor) -> Tensor:
        """Slow Tensor-output reference used only by tiny correctness tests."""
        layer, base, layer_function = self._functions(layer_id)
        if tuple(vector.shape) != layer.matrix_shape:
            raise ValueError(
                f"Vector for {layer_id} must have shape {layer.matrix_shape}"
            )
        with _preserve_execution_state(self.model):
            weight = base.detach().clone().requires_grad_(True)
            output = layer_function(weight)
            if not isinstance(output, Tensor):
                raise GGNOperatorError(
                    "Double-autograd reference requires a Tensor model output"
                )
            output_gradient = torch.autograd.grad(
                self.batch.loss_fn(output), output, create_graph=True
            )[0]
            tangent = torch.autograd.functional.jvp(
                layer_function,
                weight,
                vector.to(weight),
                create_graph=False,
                strict=True,
            )[1]
            output_hvp = torch.autograd.grad(
                output_gradient, output, grad_outputs=tangent, retain_graph=True
            )[0]
            product = torch.autograd.grad(output, weight, grad_outputs=output_hvp)[0]
        return product.detach().to(vector)

    def explicit_matrix_for_testing(
        self, layer_id: str, maximum_elements: int = 256
    ) -> Tensor:
        layer = self.registry.by_id(layer_id)
        count = layer.weight.numel()
        if count > maximum_elements:
            raise ValueError(
                "Explicit GGN construction is test-only and limited to tiny layers"
            )
        dtype = torch.float64 if layer.weight.dtype == torch.float64 else torch.float32
        identity = torch.eye(count, device=layer.weight.device, dtype=dtype)
        columns = [
            self.matvec(layer_id, identity[:, index].reshape(layer.matrix_shape))
            for index in range(count)
        ]
        return torch.stack([column.reshape(-1) for column in columns], dim=1)


class GGNFullOperator:
    """Matrix-free generalized Gauss--Newton product over all trainable parameters."""

    def __init__(self, model: nn.Module, batch: FunctionalCurvatureBatch) -> None:
        self.model = model
        self.batch = batch
        named = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if not named:
            raise ValueError("GGNFullOperator requires trainable parameters")
        self.parameter_names = tuple(name for name, _ in named)
        self.parameters = tuple(parameter for _, parameter in named)
        self._sizes = tuple(parameter.numel() for parameter in self.parameters)
        self.matvec_count = 0

    @property
    def numel(self) -> int:
        return sum(self._sizes)

    def _functions(self):
        state = dict(self.model.named_parameters())
        state.update(
            {name: value.detach().clone() for name, value in self.model.named_buffers()}
        )
        base = tuple(state[name] for name in self.parameter_names)

        def model_function(*values: Tensor) -> Any:
            candidate = dict(state)
            candidate.update(zip(self.parameter_names, values, strict=True))
            return functional_call(
                self.model,
                candidate,
                self.batch.args,
                dict(self.batch.kwargs),
                strict=False,
            )

        return base, model_function

    def _unflatten(self, vector: Tensor, base: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if vector.numel() != self.numel:
            raise ValueError("Full GGN vector has the wrong size")
        values = []
        offset = 0
        for size, parameter in zip(self._sizes, base, strict=True):
            values.append(
                vector[offset : offset + size]
                .reshape_as(parameter)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            offset += size
        return tuple(values)

    @staticmethod
    def _flatten(values: tuple[Tensor, ...], *, dtype: torch.dtype) -> Tensor:
        return torch.cat(
            tuple(value.detach().reshape(-1).to(dtype=dtype) for value in values)
        )

    def gradient(self) -> Tensor:
        with _preserve_execution_state(self.model):
            base, model_function = self._functions()
            output, pullback = vjp(model_function, *base)
            loss = self.batch.loss_fn(output)
            output_gradient = grad(self.batch.loss_fn)(output)
            if not torch.isfinite(loss).all() or not _tree_is_finite(output_gradient):
                raise GGNOperatorError("Curvature loss or output gradient is non-finite")
            values = pullback(output_gradient)
        dtype = torch.float64 if base[0].dtype == torch.float64 else torch.float32
        return self._flatten(values, dtype=dtype)

    def matvec(self, vector: Tensor) -> Tensor:
        with _preserve_execution_state(self.model):
            base, model_function = self._functions()
            tangent = self._unflatten(vector, base)
            output, pullback = vjp(model_function, *base)
            _, output_tangent = jvp(model_function, base, tangent)
            if self.batch.output_hvp_fn is None:
                loss = self.batch.loss_fn(output)
                output_gradient_fn = grad(self.batch.loss_fn)
                output_gradient = output_gradient_fn(output)
                if not torch.isfinite(loss).all() or not _tree_is_finite(
                    output_gradient
                ):
                    raise GGNOperatorError(
                        "Curvature loss or output gradient is non-finite"
                    )
                _, output_hvp = jvp(
                    output_gradient_fn, (output,), (output_tangent,)
                )
            else:
                output_hvp = self.batch.output_hvp_fn(output, output_tangent)
                if not _tree_is_finite(output_hvp):
                    raise GGNOperatorError("Analytic output Hessian product is non-finite")
            values = pullback(output_hvp)
        dtype = torch.float64 if vector.dtype == torch.float64 else torch.float32
        result = self._flatten(values, dtype=dtype).to(device=vector.device)
        self.matvec_count += 1
        if not torch.isfinite(result).all():
            raise GGNOperatorError("GGN product is non-finite")
        return result


class AveragedGGNOperator:
    """Average several exact full-GGN mini-batch operators without extra peak memory."""

    def __init__(self, operators: tuple[GGNFullOperator, ...]) -> None:
        if not operators:
            raise ValueError("AveragedGGNOperator requires at least one operator")
        first = operators[0]
        parameter_ids = tuple(id(parameter) for parameter in first.parameters)
        if any(
            tuple(id(parameter) for parameter in operator.parameters) != parameter_ids
            for operator in operators[1:]
        ):
            raise ValueError("All averaged GGN operators must share parameters")
        self.operators = operators
        self.parameters = first.parameters
        self._sizes = first._sizes

    @property
    def numel(self) -> int:
        return self.operators[0].numel

    def gradient(self) -> Tensor:
        total = None
        for operator in self.operators:
            gradient = operator.gradient()
            total = gradient if total is None else total.add(gradient.to(total))
        assert total is not None
        return total / len(self.operators)

    def gradient_statistics(self) -> tuple[Tensor, Tensor]:
        total = None
        squared_total = None
        for operator in self.operators:
            gradient = operator.gradient()
            if total is None:
                total = gradient
                squared_total = gradient.square()
            else:
                total.add_(gradient.to(total))
                assert squared_total is not None
                squared_total.add_(gradient.to(squared_total).square())
        assert total is not None and squared_total is not None
        count = len(self.operators)
        return total / count, squared_total / count

    def matvec(self, vector: Tensor) -> Tensor:
        total = torch.zeros_like(vector)
        for operator in self.operators:
            total.add_(operator.matvec(vector).to(total))
        return total / len(self.operators)


class SplitBatchGGNOperator:
    """Use a large gradient batch and a smaller fixed curvature batch."""

    def __init__(
        self,
        *,
        gradient_operator: AveragedGGNOperator,
        curvature_operator: AveragedGGNOperator,
    ) -> None:
        gradient_parameter_ids = tuple(
            id(parameter) for parameter in gradient_operator.parameters
        )
        curvature_parameter_ids = tuple(
            id(parameter) for parameter in curvature_operator.parameters
        )
        if gradient_parameter_ids != curvature_parameter_ids:
            raise ValueError("Gradient and curvature operators must share parameters")
        self.gradient_operator = gradient_operator
        self.curvature_operator = curvature_operator
        self.parameters = gradient_operator.parameters
        self._sizes = gradient_operator._sizes

    @property
    def numel(self) -> int:
        return self.gradient_operator.numel

    def gradient(self) -> Tensor:
        return self.gradient_operator.gradient()

    def gradient_statistics(self) -> tuple[Tensor, Tensor]:
        return self.gradient_operator.gradient_statistics()

    def matvec(self, vector: Tensor) -> Tensor:
        return self.curvature_operator.matvec(vector)
