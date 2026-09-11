from __future__ import annotations

import time

import torch

from .adapter import ProposalAdapter
from .attention import edge_factors, filter_routing_increment, qk_correction
from .layernorm import layernorm_correction
from .linalg import ProbeFailure, finite, low_rank_prox
from .probes import collect_factors, collect_head_probe, diagnostic_mode, functional_logits, lm_loss, paired_embedding_sketch
from .temporal import cohort_step, fixed_sketch, guarded_filter_step, predictive_maps, remap_gradient


METHODS = ("adamw", "muon", "qk_defect_v1", "routing_resistance_v1",
           "tied_path_curvature_v1", "proposal_notch_v1", "feature_remap_cohort_v1",
           "ln_response_v1")


class OptimizerV2:
    def __init__(self, model, method, options=None, *, seed=0, baseline=None):
        if method not in (*METHODS, "feature_remap_v1"):
            raise ValueError(f"unknown optimizer 2.0 method: {method}")
        self.model, self.method = model, method
        self.options = dict(options or {})
        allowed = {
            "qk_defect_v1": {"interval", "query_rows", "rho", "lambda_rel", "nu_rel", "full_loss_interval"},
            "routing_resistance_v1": {"interval", "query_rows", "edges_per_row", "mixture", "rho"},
            "tied_path_curvature_v1": {"interval", "max_age", "probes", "rho"},
            "proposal_notch_v1": {"radius"},
            "feature_remap_cohort_v1": {"interval", "max_age", "block_size", "audit_interval"},
            "feature_remap_v1": {"interval", "max_age", "block_size", "audit_interval"},
            "ln_response_v1": {"interval"}, "adamw": set(), "muon": set(),
        }[method] | {"enabled"}
        if set(self.options) - allowed:
            raise ValueError(f"unsupported {method} options: {sorted(set(self.options) - allowed)}")
        for key in ("interval", "query_rows", "edges_per_row", "probes", "max_age", "block_size", "audit_interval", "full_loss_interval"):
            if key in self.options and (not isinstance(self.options[key], int) or self.options[key] < 1):
                raise ValueError(f"{key} must be a positive integer")
        if self.options.get("rho", 1) < 0 or self.options.get("lambda_rel", 1) <= 0 or self.options.get("nu_rel", 1) < 0:
            raise ValueError("invalid correction strength or damping")
        if not 0 <= self.options.get("mixture", 0.05) <= 1 or not 0 < self.options.get("radius", 0.8) < 1:
            raise ValueError("invalid sampling mixture or filter radius")
        self.seed = seed
        self.adapter = ProposalAdapter(model, adam_only=method == "adamw", **(baseline or {}))
        self.steps = 0
        self.state = {}
        self.signs = {}
        self.last_diagnostics = {}
        self.cost = {key: 0 for key in ("auxiliary_input_tokens", "auxiliary_forward_calls",
                                       "auxiliary_backward_calls", "auxiliary_jvp_calls", "auxiliary_vjp_calls")}
        self.cost["optimizer_seconds"] = 0.0
        self.cost["auxiliary_seconds"] = 0.0
        self.cost["fallback_events"] = 0
        self.cost["local_attention_replays"] = 0
        self.targets = [name for name in self.adapter.parameters if name.endswith("mlp.c_proj.weight")]

    @property
    def param_groups(self):
        return self.adapter.param_groups

    def zero_grad(self, set_to_none=True):
        self.adapter.zero_grad(set_to_none)

    def state_dict(self):
        return {"version": 1, "method": self.method, "options": self.options, "seed": self.seed,
                "steps": self.steps, "adapter": self.adapter.state_dict(), "state": self.state,
                "cost": self.cost, "parameter_layout": [(n, tuple(p.shape)) for n, p in self.adapter.parameters.items()]}

    def load_state_dict(self, saved):
        expected = [(n, tuple(p.shape)) for n, p in self.adapter.parameters.items()]
        if (saved["version"] != 1 or saved["method"] != self.method or saved["options"] != self.options
                or saved["seed"] != self.seed or saved["parameter_layout"] != expected):
            raise ValueError("optimizer version, options, seed, or parameter layout changed")
        self.adapter.load_state_dict(saved["adapter"])
        self.steps, self.state, self.cost = saved["steps"], saved["state"], saved["cost"]

    def generator(self, step, stream=0):
        return torch.Generator().manual_seed(self.seed * 1_000_003 + step * 1009 + stream + 17)

    def _count(self, ids, *, forward=0, backward=0, jvp=0, vjp=0):
        self.cost["auxiliary_forward_calls"] += forward
        self.cost["auxiliary_backward_calls"] += backward
        self.cost["auxiliary_jvp_calls"] += jvp
        self.cost["auxiliary_vjp_calls"] += vjp
        self.cost["auxiliary_input_tokens"] += ids.numel() * (forward + backward + jvp + vjp)

    def _synchronize(self):
        parameter = next(iter(self.adapter.parameters.values()))
        if parameter.is_cuda:
            torch.cuda.synchronize(parameter.device)

    def _factors(self, ids):
        return collect_factors(self.model, ids, [name.removesuffix(".weight") for name in self.targets],
                               counter=lambda **calls: self._count(ids, **calls))

    def _local_replay(self):
        self.cost["local_attention_replays"] += 1

    def _prediction_diagnostic(self, state, step, fit_ids, check_ids):
        if step > 128 or (step != 1 and step % 8 != 0):
            return state, {}
        fit, check = self._factors(fit_ids), self._factors(check_ids)
        old = state.get("prediction_anchors")
        diagnostics = {}
        totals = dict(state.get("prediction_totals", {"evaluated": 0, "accepted": 0, "raw_error": 0.0, "map_error": 0.0}))
        if old:
            for name in fit:
                _, diagnostic = predictive_maps(old["fit"][name], fit[name], old["check"][name], check[name])
                diagnostics[name] = diagnostic
                totals["evaluated"] += 1
                totals["accepted"] += diagnostic["accepted"]
                totals["raw_error"] += diagnostic["raw_error"]
                totals["map_error"] += diagnostic["map_error"]
        state = dict(state, prediction_anchors={"fit": fit, "check": check}, prediction_totals=totals)
        if step == 128:
            totals["completed"] = True
            totals["prediction_improved"] = totals["map_error"] < totals["raw_error"]
            state.pop("prediction_anchors")
        return state, {"prediction_check": diagnostics, "prediction_totals": totals}

    def _feature(self, state, step, fit_ids, check_ids, audit_ids=None):
        interval = 1 if self.method == "feature_remap_v1" else int(self.options.get("interval", 8))
        due = (step - 1) % interval == 0
        fit = check = None
        diagnostics = {}
        audit = old_audit = None
        if due:
            try:
                fit, check = self._factors(fit_ids), self._factors(check_ids)
                for values in (*fit.values(), *check.values()):
                    finite(*values)
            except (ProbeFailure, torch.linalg.LinAlgError) as error:
                diagnostics["probe_failure"] = str(error)
                self.cost["fallback_events"] += 1
                fit = check = None
        if due and fit is not None:
            try:
                if audit_ids is not None:
                    audit = self._factors(audit_ids)
                if (step - 1) % self.options.get("audit_interval", 128) == 0:
                    old_audit, audit_cache = {}, {}
                    for name, previous in state.get("cohorts", {}).items():
                        if "audit_ids" in previous:
                            key = tuple(previous["audit_ids"].flatten().tolist())
                            if key not in audit_cache:
                                audit_cache[key] = self._factors(previous["audit_ids"])
                            module = name.removesuffix(".weight")
                            old_audit[module] = audit_cache[key][module]
            except (ProbeFailure, torch.linalg.LinAlgError) as error:
                diagnostics["audit_failure"] = str(error)
                audit = old_audit = None
        overrides, cohorts = {}, {}
        for name in self.targets:
            parameter = self.adapter.parameters[name]
            _, group = self.adapter.owners[id(parameter)]
            beta = group["momentum"]
            previous = state.get("cohorts", {}).get(name, {})
            historical = previous.get("historical", self.adapter.momentum(name))
            fresh = previous.get("fresh", torch.zeros_like(historical))
            record = dict(previous)
            maps = None
            valid = fit is not None
            module = name.removesuffix(".weight")
            if valid and "fit" in previous and step - previous["snapshot_step"] <= self.options.get("max_age", 32):
                try:
                    maps, diagnostic = predictive_maps(previous["fit"], fit[module], previous["check"], check[module],
                                                       self.options.get("block_size", 32))
                    diagnostics[name] = diagnostic
                except (ProbeFailure, torch.linalg.LinAlgError) as error:
                    valid = False
                    diagnostics[name] = {"probe_failure": str(error)}
                    self.cost["fallback_events"] += 1
                if valid and old_audit is not None and module in old_audit and "audit" in previous:
                    try:
                        old_gradient = previous["audit"][0].T @ previous["audit"][1]
                        current_gradient = old_audit[module][0].T @ old_audit[module][1]
                        mapped = remap_gradient(old_gradient, *maps)
                        finite(current_gradient, old_gradient, mapped)
                        diagnostic["fresh_audit_raw_error"] = float((current_gradient.double() - old_gradient.double()).square().sum())
                        diagnostic["fresh_audit_mapped_error"] = float((current_gradient.double() - mapped.double()).square().sum())
                    except (ProbeFailure, torch.linalg.LinAlgError) as error:
                        diagnostic["fresh_audit_failure"] = str(error)
            momentum, older, younger = cohort_step(historical, fresh, parameter.grad, beta, maps)
            if due and not valid:
                # Failed probes use the stock next momentum while keeping the old age boundary.
                momentum = self.adapter.momentum(name).clone().lerp_(parameter.grad, 1 - beta)
                younger = momentum - older
            if valid:
                record.update(historical=momentum, fresh=torch.zeros_like(momentum),
                              fit=fit[module], check=check[module], snapshot_step=step)
                if audit is not None:
                    record["audit"] = audit[module]
                    record["audit_ids"] = audit_ids.detach().clone()
                else:
                    record.pop("audit", None)
                    record.pop("audit_ids", None)
            else:
                record.update(historical=older, fresh=younger)
                if step - record.get("snapshot_step", step) > self.options.get("max_age", 32):
                    record.pop("fit", None)
                    record.pop("check", None)
                    record.update(historical=momentum, fresh=torch.zeros_like(momentum))
            overrides[name], cohorts[name] = momentum, record
        state = dict(state, cohorts=cohorts)
        return overrides, state, diagnostics

    def _head_correction(self, proposals, step, fit_ids, check_ids):
        interval = int(self.options.get("interval", 8))
        event = step // interval - 1
        head = event % (self.model.config.n_layer * self.model.config.n_head)
        block, head = divmod(head, self.model.config.n_head)
        count = int(self.options.get("query_rows", 8 if self.method == "qk_defect_v1" else 4))
        fit = collect_head_probe(self.model, fit_ids, block, head, count, self.generator(step),
                                 need_error=self.method == "qk_defect_v1",
                                 counter=lambda **calls: self._count(fit_ids, **calls))
        width = self.model.config.n_embd
        head_width = width // self.model.config.n_head
        slices = [slice(offset * width + head * head_width, offset * width + (head + 1) * head_width) for offset in range(3)]
        root = f"transformer.h.{block}.attn"
        weight_name, bias_name = root + ".c_attn.weight", root + ".c_attn.bias"
        output_name = root + ".c_proj.weight"
        if self.method == "qk_defect_v1":
            check = collect_head_probe(self.model, check_ids, block, head, count, self.generator(step, 1), need_error=True,
                                       counter=lambda **calls: self._count(check_ids, **calls))
            delta_w = proposals[weight_name].value - self.adapter.parameters[weight_name]
            delta_b = proposals[bias_name].value - self.adapter.parameters[bias_name]
            delta_o = proposals[output_name].value - self.adapter.parameters[output_name]
            delta = {"wq": delta_w[:, slices[0]], "wk": delta_w[:, slices[1]], "wv": delta_w[:, slices[2]],
                     "bq": delta_b[slices[0]], "bk": delta_b[slices[1]], "bv": delta_b[slices[2]],
                     "wo": delta_o[slices[0]]}
            cq, ck, diagnostic = qk_correction(fit, check, delta, rho=self.options.get("rho", 0.25),
                                              lambda_rel=self.options.get("lambda_rel", 0.01),
                                              nu_rel=self.options.get("nu_rel", 1.0), replay_counter=self._local_replay)
        else:
            factors, probabilities = edge_factors(fit["x"], fit["q"], fit["k"], fit["rows"],
                                                  edges_per_row=self.options.get("edges_per_row", 4),
                                                  mixture=self.options.get("mixture", 0.05), generator=self.generator(step, 2))
            learning = proposals[weight_name].learning
            dq, dk = learning[:, slices[0]], learning[:, slices[1]]
            fq, fk, diagnostic = filter_routing_increment(factors, dq, dk, self.options.get("rho", 1.0))
            cq, ck = fq - dq, fk - dk
            diagnostic.update(sample_probability_min=min(probabilities), sample_probability_max=max(probabilities),
                              edges=len(probabilities))
        correction = torch.zeros_like(proposals[weight_name].learning)
        correction[:, slices[0]], correction[:, slices[1]] = cq, ck
        if self.method == "qk_defect_v1" and step % self.options.get("full_loss_interval", 512) == 0:
            parameters = {name: proposals[name].value if name in proposals else value.detach()
                          for name, value in self.model.named_parameters()}
            with diagnostic_mode(self.model), torch.no_grad():
                self._count(check_ids, forward=1)
                baseline_loss = lm_loss(functional_logits(self.model, parameters, check_ids), check_ids)
                finite(baseline_loss)
                diagnostic["full_model_baseline_check_loss"] = float(baseline_loss)
                parameters[weight_name] = parameters[weight_name] + correction
                self._count(check_ids, forward=1)
                corrected_loss = lm_loss(functional_logits(self.model, parameters, check_ids), check_ids)
                finite(corrected_loss)
                diagnostic["full_model_corrected_check_loss"] = float(corrected_loss)
        return {weight_name: correction}, {"block": block, "head": head, **diagnostic}

    @torch.no_grad()
    def step(self, fit_ids=None, check_ids=None, audit_ids=None):
        flags = [torch.isfinite(p.grad).all() for p in self.adapter.parameters.values() if p.grad is not None]
        if not flags or not bool(torch.stack(flags).all()):
            raise FloatingPointError("nonfinite or missing training gradients; no optimizer state committed")
        started = time.perf_counter()
        step = self.steps + 1
        self.last_diagnostics = {"step": step, "method": self.method}
        disabled = not self.options.get("enabled", True) or self.options.get("rho", 1.0) == 0
        if self.method == "adamw" or disabled:
            self.adapter.step()
        elif self.method == "muon":
            if fit_ids is not None and check_ids is not None:
                auxiliary_started = time.perf_counter()
                try:
                    self.state, diagnostic = self._prediction_diagnostic(self.state, step, fit_ids, check_ids)
                    self.last_diagnostics.update(diagnostic)
                except (ProbeFailure, torch.linalg.LinAlgError) as error:
                    self.last_diagnostics["fallback"] = str(error)
                    self.cost["fallback_events"] += 1
                self._synchronize()
                self.cost["auxiliary_seconds"] += time.perf_counter() - auxiliary_started
            self.adapter.step()
        elif self.method in ("qk_defect_v1", "routing_resistance_v1", "ln_response_v1") and step % self.options.get("interval", 32 if self.method == "ln_response_v1" else 8):
            self.adapter.step()
        else:
            state = dict(self.state)
            corrections = {}
            proposals = None
            auxiliary_started = time.perf_counter()
            baseline_seconds = 0.0
            def baseline_call(function, *args):
                nonlocal baseline_seconds
                self._synchronize()
                baseline_started = time.perf_counter()
                value = function(*args)
                self._synchronize()
                baseline_seconds += time.perf_counter() - baseline_started
                return value
            try:
                if self.method.startswith("feature_remap"):
                    overrides, state, diagnostic = self._feature(state, step, fit_ids, check_ids, audit_ids)
                    proposals = baseline_call(self.adapter.propose, overrides)
                else:
                    proposals = baseline_call(self.adapter.propose)
                    if self.method in ("qk_defect_v1", "routing_resistance_v1"):
                        corrections, diagnostic = self._head_correction(proposals, step, fit_ids, check_ids)
                    elif self.method == "tied_path_curvature_v1":
                        interval = int(self.options.get("interval", 16))
                        diagnostic = {}
                        if (step - 1) % interval == 0:
                            count = int(self.options.get("probes", 2))
                            try:
                                columns, paths = paired_embedding_sketch(self.model, fit_ids, count, self.generator(step),
                                                                         counter=lambda **calls: self._count(fit_ids, **calls))
                                finite(*columns)
                                state.update(columns=columns, cache_step=step)
                                diagnostic["paths"] = paths
                            except (ProbeFailure, torch.linalg.LinAlgError) as error:
                                diagnostic["refresh_failure"] = str(error)
                                self.cost["fallback_events"] += 1
                        age = step - state.get("cache_step", -1_000_000)
                        diagnostic["metric_age"] = age
                        if "columns" in state and age < self.options.get("max_age", 16):
                            name = "transformer.wte.weight"
                            filtered, metric = low_rank_prox(proposals[name].learning, state["columns"], self.options.get("rho", 1.0))
                            corrections[name] = filtered - proposals[name].learning
                            diagnostic.update(metric)
                        else:
                            diagnostic["stale_cache_bypass"] = True
                    elif self.method == "proposal_notch_v1":
                        diagnostic, filters = {}, {}
                        for index, name in enumerate(self.targets):
                            direction = proposals[name].direction
                            sketch, signs = fixed_sketch(direction, self.seed + 10_007 * (index + 1), self.signs.get(name))
                            self.signs[name] = signs
                            output, record, diagnostic[name] = guarded_filter_step(
                                direction, self.adapter.parameters[name], state.get("filters", {}).get(name, {}),
                                step, sketch, radius=self.options.get("radius", 0.8))
                            filters[name] = record
                            if output is not direction:
                                _, group = self.adapter.owners[id(self.adapter.parameters[name])]
                                corrections[name] = -float(group["lr"]) * (output - direction)
                        state["filters"] = filters
                    else:
                        block = (step // self.options.get("interval", 32) - 1) % self.model.config.n_layer
                        corrections, diagnostic, _, _ = layernorm_correction(self.model, proposals, fit_ids, check_ids, block, self._count)
                        diagnostic["block"] = block
                for correction in corrections.values():
                    finite(correction)
                baseline_call(self.adapter.commit, proposals, corrections)
                self.state = state
                self.last_diagnostics.update(diagnostic)
            except (ProbeFailure, torch.linalg.LinAlgError) as error:
                # The live model and baseline state have not been touched by trial evaluation.
                baseline_call(self.adapter.step)
                self.cost["fallback_events"] += 1
                self.last_diagnostics["fallback"] = str(error)
            self.cost["auxiliary_seconds"] += time.perf_counter() - auxiliary_started - baseline_seconds
        self._synchronize()
        self.steps = step
        self.cost["optimizer_seconds"] += time.perf_counter() - started

    def persistent_state_bytes(self):
        seen = set()
        def count(value):
            if isinstance(value, torch.Tensor):
                key = (value.device, value.untyped_storage().data_ptr())
                if key in seen:
                    return 0
                seen.add(key)
                return value.untyped_storage().nbytes()
            if isinstance(value, dict):
                return sum(count(item) for item in value.values())
            if isinstance(value, (tuple, list)):
                return sum(count(item) for item in value)
            return 0
        return {"baseline_state_bytes": count(self.adapter.state_dict()),
                "proposal_state_bytes": count(self.state), "sketch_cache_bytes": count(self.signs)}
