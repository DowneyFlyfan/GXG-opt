"""Bounded RTX 5090 shape/timing probe; never an optimizer-quality benchmark.

No data or weights are downloaded. Random tokens exercise the complete dense
forward/backward/update at the proposed shapes. Use real frozen data and the
actual candidate's auxiliary work before locking an experiment budget.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel


def build_model(layers=12, width=512, heads=8, context=512, attention="sdpa"):
    config = GPT2Config(
        vocab_size=50257, n_positions=context, n_ctx=context,
        n_layer=layers, n_embd=width, n_head=heads, n_inner=4 * width,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        activation_function="gelu_new", tie_word_embeddings=True,
        use_cache=False, bos_token_id=50256, eos_token_id=50256,
    )
    config._attn_implementation = attention
    return GPT2LMHeadModel(config).float()


def build_optimizers(model, method):
    # GPT-2 Conv1D stores [in, out]; match_rms_adamw's symmetric shape factor
    # avoids the orientation dependence of the stock 'original' adjustment.
    matrices = [p for n, p in model.named_parameters()
                if n.startswith("transformer.h.") and p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    auxiliary = [p for p in model.parameters() if id(p) not in matrix_ids]
    if method == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=6e-4,
                                  betas=(0.9, 0.95), weight_decay=0.0,
                                  fused=True)]
    return [
        torch.optim.Muon(matrices, lr=0.01, momentum=0.95, nesterov=True,
                         ns_steps=5, adjust_lr_fn="match_rms_adamw",
                         weight_decay=0.0),
        torch.optim.AdamW(auxiliary, lr=6e-4, betas=(0.9, 0.95),
                          weight_decay=0.0, fused=True),
    ]


def calibrate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for timing")
    gpu = torch.cuda.get_device_name(0)
    if "RTX 5090" not in gpu:
        raise RuntimeError(f"Expected RTX 5090, found {gpu}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    free, total = torch.cuda.mem_get_info()
    if free < 12 * 2**30:
        raise RuntimeError("Less than 12 GiB free; defer the probe until GPU is available")
    model = build_model(args.layers, args.width, args.heads, args.context, args.attention).cuda()
    optimizers = build_optimizers(model, args.method)
    accumulation = args.effective_batch // args.micro_batch
    generator = torch.Generator().manual_seed(args.seed + 1)
    # CPU pinned source and transfer are included in each update's timer.
    batches = [torch.randint(0, 50257, (args.micro_batch, args.context + 1),
                             generator=generator).pin_memory()
               for _ in range(accumulation)]

    def step():
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        detached_losses = []
        for cpu_ids in batches:
            ids = cpu_ids.to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids[:, :-1], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
                ) / accumulation
            loss.backward()
            detached_losses.append(loss.detach())
            del logits, loss, ids
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()
        return torch.stack(detached_losses).sum()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    timings = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite probe loss")
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    # Measure a representative forward-only evaluation batch separately.
    model.eval()
    evaluation_times = []
    with torch.inference_mode():
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            ids = batches[0].to("cuda", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids[:, :-1], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
                )
            torch.cuda.synchronize()
            evaluation_times.append(time.perf_counter() - started)
            del logits, ids, loss
    mean = statistics.mean(timings)
    median = statistics.median(timings)
    p90 = sorted(timings)[math.ceil(0.9 * len(timings)) - 1]
    try:
        telemetry = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,power.limit",
             "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        telemetry = None
    result = {
        "measurement_kind": "synthetic_shape_calibration_only",
        "quality_evidence": False,
        "gpu": gpu, "gpu_total_gib": total / 2**30,
        "gpu_free_gib_before": free / 2**30, "telemetry_after": telemetry,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "muon_source_sha256": hashlib.sha256(
            Path(importlib.import_module("torch.optim._muon").__file__).read_bytes()
        ).hexdigest(),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "layers": args.layers, "width": args.width, "heads": args.heads,
        "context": args.context, "attention": args.attention,
        "method": args.method, "micro_batch": args.micro_batch,
        "effective_batch": args.effective_batch, "accumulation": accumulation,
        "prediction_tokens_per_step": args.effective_batch * args.context,
        "warmup_steps": args.warmup, "warmup_seconds": warmup_seconds,
        "measured_steps": args.steps, "step_seconds": timings,
        "mean_step_seconds": mean, "median_step_seconds": median,
        "p90_step_seconds": p90,
        "prediction_tokens_per_second": args.effective_batch * args.context / mean,
        "evaluation_batch_seconds": statistics.median(evaluation_times[1:]),
        "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
        "model_config": model.config.to_dict(),
        "limitations": ["Synthetic repeating tokens; no dataset-learning conclusion",
                        "Short eager execution without torch.compile; no checkpoint I/O",
                        "Candidate probes and full-dataset evaluation not measured"],
    }
    del model, optimizers, batches
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--effective-batch", type=int, default=32)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for field in ("layers", "width", "heads", "context", "micro_batch", "effective_batch", "warmup", "steps"):
        if getattr(args, field) < 1:
            parser.error(f"{field} must be positive")
    if args.width % args.heads or args.effective_batch % args.micro_batch:
        parser.error("width must divide by heads; effective batch must divide by micro batch")
    if args.output.exists():
        parser.error("output already exists; use a fresh path to preserve measurements")
    result = calibrate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "model_parameters", "method", "micro_batch", "accumulation",
        "mean_step_seconds", "p90_step_seconds", "prediction_tokens_per_second",
        "evaluation_batch_seconds", "peak_allocated_gib", "peak_reserved_gib",
    )}, indent=2))


if __name__ == "__main__":
    main()
