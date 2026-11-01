from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from .model import SmallTransformerClassifier, SmallTransformerConfig
from .selector import AnchorSelector, AnchorSelectorConfig


METHODS = (
    "full",
    "local",
    "d-rgsa",
    "d-rgsa-mlp",
    "d-rgsa-student",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark optimized Full Attention and truly structured "
            "local/global D-RGSA attention."
        )
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=16)
    parser.add_argument("--anchor-budget", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-size", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=30522)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--selector-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / (1024.0**2)
    return value / 1024.0


def accelerator_memory_mb(device: torch.device) -> dict[str, float]:
    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / (1024.0**2),
            "reserved_mb": torch.cuda.memory_reserved(device) / (1024.0**2),
        }
    if device.type == "mps":
        return {
            "allocated_mb": torch.mps.current_allocated_memory() / (1024.0**2),
            "driver_mb": torch.mps.driver_allocated_memory() / (1024.0**2),
        }
    return {}


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_state(module: torch.nn.Module, path: Path | None) -> None:
    if path is None:
        return
    state = torch.load(path, map_location="cpu", weights_only=True)
    module.load_state_dict(state)


def build_models(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    strategy = {
        "full": "full",
        "local": "local",
        "d-rgsa": "distilled",
        "d-rgsa-mlp": "distilled",
        "d-rgsa-student": "distilled",
    }[args.method]
    student_config = SmallTransformerConfig(
        vocab_size=args.vocab_size,
        num_labels=args.num_labels,
        pad_token_id=0,
        max_length=args.length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_size=args.ffn_size,
        dropout=0.0,
    )
    student = SmallTransformerClassifier(
        student_config,
        strategy=strategy,
        window_radius=args.window_radius,
        attention_backend="structured_sparse",
    )
    load_state(student, args.student_checkpoint)
    student.to(device=device, dtype=dtype).eval()
    selector = None
    if args.method in {"d-rgsa", "d-rgsa-mlp"}:
        architecture = "bigru" if args.method == "d-rgsa" else "mlp"
        selector_config = AnchorSelectorConfig(
            vocab_size=args.vocab_size,
            pad_token_id=0,
            max_length=args.length,
            embedding_size=128,
            hidden_size=64,
            architecture=architecture,
            dropout=0.0,
        )
        selector = AnchorSelector(selector_config)
        load_state(selector, args.selector_checkpoint)
        selector.to(device=device, dtype=dtype).eval()
    return student, selector


def logical_edge_count(
    length: int,
    radius: int,
    anchors: list[int],
    *,
    method: str,
) -> int:
    if method == "full":
        return length * length
    anchor_set = set(anchors) if method.startswith("d-rgsa") else set()
    active = 0
    for query in range(length):
        if query in anchor_set:
            active += length
            continue
        local_start = max(0, query - radius)
        local_end = min(length, query + radius + 1)
        local = local_end - local_start
        global_outside_local = sum(
            not (local_start <= anchor < local_end) for anchor in anchor_set
        )
        active += local + global_outside_local
    return active


def kernel_score_count(
    length: int,
    radius: int,
    anchor_budget: int,
    *,
    method: str,
) -> int:
    if method == "full":
        return length * length
    local_scores = length * (2 * radius + 1)
    if method == "local":
        return local_scores
    return local_scores + 2 * length * anchor_budget


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    args = arguments()
    if min(args.length, args.batch_size, args.repeats) <= 0:
        raise ValueError("length, batch-size, and repeats must be positive")
    if min(args.window_radius, args.anchor_budget, args.warmup) < 0:
        raise ValueError("window-radius, anchor-budget, and warmup cannot be negative")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = dtype_from_name(args.dtype)
    student, selector = build_models(args, device, dtype)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        1,
        args.vocab_size,
        (args.batch_size, args.length),
        generator=generator,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    empty_anchor_mask = torch.zeros_like(input_ids)
    latest_anchor_mask = empty_anchor_mask
    if args.method == "d-rgsa-student":
        fixed_positions = torch.linspace(
            0,
            args.length - 1,
            steps=min(args.anchor_budget, args.length),
            device=device,
        ).round().long()
        latest_anchor_mask = torch.zeros_like(attention_mask).scatter(
            1,
            fixed_positions.unsqueeze(0).expand(args.batch_size, -1),
            1,
        )

    @torch.inference_mode()
    def run_once() -> torch.Tensor:
        nonlocal latest_anchor_mask
        if args.method == "d-rgsa-student":
            pass
        elif selector is None:
            latest_anchor_mask = empty_anchor_mask
        else:
            selector_logits = selector(input_ids, attention_mask)
            count = min(args.anchor_budget, args.length)
            selected = torch.topk(
                selector_logits.masked_fill(~attention_mask.bool(), -1.0e4),
                k=count,
                dim=1,
            ).indices
            latest_anchor_mask = torch.zeros_like(attention_mask).scatter(
                1,
                selected,
                1,
            )
        return student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            anchor_mask=latest_anchor_mask,
        )["logits"]

    rss_after_models = peak_rss_mb()
    accelerator_after_models = accelerator_memory_mb(device)
    for _ in range(args.warmup):
        run_once()
    synchronize(device)
    latencies_ms: list[float] = []
    accelerator_observed: list[dict[str, float]] = []
    checksum = 0.0
    for _ in range(args.repeats):
        synchronize(device)
        start = time.perf_counter()
        logits = run_once()
        synchronize(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        accelerator_observed.append(accelerator_memory_mb(device))
        checksum += float(logits.float().sum().cpu())
    anchors = (
        latest_anchor_mask[0].nonzero(as_tuple=False).flatten().cpu().tolist()
        if args.method.startswith("d-rgsa")
        else []
    )
    logical_edges = logical_edge_count(
        args.length,
        args.window_radius,
        [int(value) for value in anchors],
        method=args.method,
    )
    kernel_scores = kernel_score_count(
        args.length,
        args.window_radius,
        len(anchors),
        method=args.method,
    )
    mean_latency = statistics.mean(latencies_ms)
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": args.method,
        "attention_backend": (
            "optimized_full_sdpa" if args.method == "full" else "structured_sparse"
        ),
        "selector_included": selector is not None,
        "length": args.length,
        "batch_size": args.batch_size,
        "window_radius": args.window_radius,
        "anchor_budget": args.anchor_budget,
        "actual_anchor_count": len(anchors),
        "logical_edges_per_sample": logical_edges,
        "full_edges_per_sample": args.length**2,
        "logical_sparsity": 1.0 - logical_edges / (args.length**2),
        "kernel_score_elements_per_sample": kernel_scores,
        "kernel_score_reduction": 1.0 - kernel_scores / (args.length**2),
        "latency_mean_ms": mean_latency,
        "latency_median_ms": statistics.median(latencies_ms),
        "latency_std_ms": statistics.stdev(latencies_ms)
        if len(latencies_ms) > 1
        else 0.0,
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "throughput_samples_per_second": args.batch_size / (mean_latency / 1000.0),
        "throughput_tokens_per_second": (
            args.batch_size * args.length / (mean_latency / 1000.0)
        ),
        "process_rss_after_models_mb": rss_after_models,
        "process_peak_rss_mb": peak_rss_mb(),
        "accelerator_memory_after_models_mb": accelerator_after_models,
        "accelerator_memory_observed_peak_mb": {
            key: max(sample.get(key, 0.0) for sample in accelerator_observed)
            for key in {
                item
                for sample in accelerator_observed
                for item in sample
            }
        },
        "student_trainable_parameters": sum(
            parameter.numel() for parameter in student.parameters()
        ),
        "selector_trainable_parameters": sum(
            parameter.numel() for parameter in selector.parameters()
        )
        if selector is not None
        else 0,
        "device": str(device),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_num_threads": torch.get_num_threads(),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
        "latencies_ms": latencies_ms,
        "checksum": checksum,
        "environment": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.getenv(
                "PYTORCH_ENABLE_MPS_FALLBACK"
            ),
        },
    }
    if device.type == "cuda":
        result["accelerator_memory_observed_peak_mb"][
            "allocated_mb"
        ] = torch.cuda.max_memory_allocated(device) / (1024.0**2)
        result["accelerator_memory_observed_peak_mb"][
            "reserved_mb"
        ] = torch.cuda.max_memory_reserved(device) / (1024.0**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
