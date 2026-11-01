from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from .benchmark_cost import (
    accelerator_memory_mb,
    choose_device,
    dtype_from_name,
    peak_rss_mb,
    percentile,
    synchronize,
)
from .model import SmallTransformerClassifier, SmallTransformerConfig
from .selector import AnchorSelector, AnchorSelectorConfig

METHODS = ("full", "local", "d-rgsa", "d-rgsa-mlp")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Full, Local, and end-to-end D-RGSA on real tokenized "
            "examples with deterministic length bucketing and dynamic padding."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--student-dir", type=Path, required=True)
    parser.add_argument("--selector-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=16)
    parser.add_argument("--anchor-budget", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Disable length sorting. The default is deterministic length bucketing.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str):
                raise TypeError(f"{path}:{line_number} has no text field")
            if row.get("annotation_valid", True):
                texts.append(text)
    if not texts:
        raise ValueError(f"No valid text records found in {path}")
    return texts


def load_student(
    student_dir: Path,
    *,
    method: str,
    window_radius: int,
    device: torch.device,
    dtype: torch.dtype,
) -> SmallTransformerClassifier:
    from safetensors.torch import load_file

    config = SmallTransformerConfig(**read_json(student_dir / "student_config.json"))
    strategy = {"full": "full", "local": "local"}.get(method, "distilled")
    model = SmallTransformerClassifier(
        config,
        strategy=strategy,
        window_radius=window_radius,
        attention_backend="structured_sparse",
    )
    model.load_state_dict(load_file(student_dir / "model.safetensors"), strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_selector(
    selector_dir: Path | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> AnchorSelector | None:
    if selector_dir is None:
        return None
    config = AnchorSelectorConfig(**read_json(selector_dir / "selector_config.json"))
    selector = AnchorSelector(config)
    state = torch.load(
        selector_dir / "selector.pt",
        map_location="cpu",
        weights_only=True,
    )
    selector.load_state_dict(state, strict=True)
    return selector.to(device=device, dtype=dtype).eval()


def prepare_batches(
    texts: list[str],
    tokenizer,
    *,
    max_length: int,
    batch_size: int,
    bucket_by_length: bool,
) -> tuple[list[dict[str, torch.Tensor]], list[int]]:
    encoded: list[dict[str, list[int]]] = []
    natural_lengths: list[int] = []
    for start in range(0, len(texts), 128):
        chunk = texts[start : start + 128]
        natural = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        natural_lengths.extend(int(value) for value in natural["length"])
        tokenized = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )
        for index in range(len(chunk)):
            encoded.append(
                {
                    "input_ids": [int(value) for value in tokenized["input_ids"][index]],
                    "attention_mask": [
                        int(value) for value in tokenized["attention_mask"][index]
                    ],
                    "special_tokens_mask": [
                        int(value)
                        for value in tokenized["special_tokens_mask"][index]
                    ],
                }
            )
    order = list(range(len(encoded)))
    if bucket_by_length:
        order.sort(key=lambda index: (len(encoded[index]["input_ids"]), index))
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(order), batch_size):
        rows = [encoded[index] for index in order[start : start + batch_size]]
        padded = tokenizer.pad(
            [
                {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                }
                for row in rows
            ],
            padding=True,
            return_tensors="pt",
        )
        padded_length = int(padded["input_ids"].shape[1])
        special = torch.tensor(
            [
                row["special_tokens_mask"]
                + [1] * (padded_length - len(row["special_tokens_mask"]))
                for row in rows
            ],
            dtype=torch.bool,
        )
        candidate_mask = padded["attention_mask"].bool() & ~special
        batches.append(
            {
                "input_ids": padded["input_ids"],
                "attention_mask": padded["attention_mask"],
                "candidate_mask": candidate_mask,
            }
        )
    return batches, natural_lengths


def score_count(
    *,
    method: str,
    batch_size: int,
    padded_length: int,
    radius: int,
    anchors: int,
) -> int:
    if method == "full":
        return batch_size * padded_length**2
    local = batch_size * padded_length * (2 * radius + 1)
    if method == "local":
        return local
    return local + 2 * batch_size * padded_length * anchors


def main() -> None:
    from transformers import AutoTokenizer

    args = arguments()
    if min(args.batch_size, args.passes) <= 0:
        raise ValueError("batch-size and passes must be positive")
    if min(args.warmup, args.window_radius, args.anchor_budget) < 0:
        raise ValueError("warmup, window-radius, and anchor-budget cannot be negative")
    if args.method.startswith("d-rgsa") and args.selector_dir is None:
        raise ValueError(f"{args.method} requires --selector-dir")
    if not args.method.startswith("d-rgsa") and args.selector_dir is not None:
        raise ValueError(f"{args.method} must not receive --selector-dir")

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.student_dir, use_fast=True)
    student = load_student(
        args.student_dir,
        method=args.method,
        window_radius=args.window_radius,
        device=device,
        dtype=dtype,
    )
    selector = load_selector(args.selector_dir, device=device, dtype=dtype)
    if selector is not None and selector.config.max_length < student.config.max_length:
        raise ValueError("Selector maximum length is smaller than the student's")
    texts = read_texts(args.data_file)
    batches, natural_lengths = prepare_batches(
        texts,
        tokenizer,
        max_length=student.config.max_length,
        batch_size=args.batch_size,
        bucket_by_length=not args.preserve_order,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        torch.mps.empty_cache()
    memory_after_models = accelerator_memory_mb(device)
    rss_after_models = peak_rss_mb()

    empty_anchor_cache: dict[tuple[int, int], torch.Tensor] = {}

    @torch.inference_mode()
    def run_once(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        candidate_mask = batch["candidate_mask"].to(device)
        shape = (int(input_ids.shape[0]), int(input_ids.shape[1]))
        if selector is None:
            anchor_mask = empty_anchor_cache.get(shape)
            if anchor_mask is None:
                anchor_mask = torch.zeros_like(attention_mask)
                empty_anchor_cache[shape] = anchor_mask
            anchor_count = 0
        else:
            selector_logits = selector(input_ids, attention_mask)
            count = min(args.anchor_budget, int(input_ids.shape[1]))
            selected = torch.topk(
                selector_logits.masked_fill(~candidate_mask, -1.0e4),
                k=count,
                dim=1,
            ).indices
            anchor_mask = torch.zeros_like(attention_mask).scatter(1, selected, 1)
            anchor_mask = anchor_mask * candidate_mask.to(anchor_mask.dtype)
            anchor_count = int(anchor_mask.sum(dim=1).max().item())
        logits = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            anchor_mask=anchor_mask,
        )["logits"]
        return logits, anchor_count

    largest_batch = max(batches, key=lambda batch: int(batch["input_ids"].shape[1]))
    if args.warmup:
        # Dynamic padding creates many tensor shapes. Warm every real batch shape
        # once so first-use MPS/CUDA overhead is not counted as model latency.
        for batch in batches:
            run_once(batch)
        for _ in range(max(0, args.warmup - 1)):
            run_once(largest_batch)
    synchronize(device)

    batch_measurements: list[dict[str, Any]] = []
    observed_memory: list[dict[str, float]] = []
    total_samples = 0
    total_true_tokens = 0
    total_padded_tokens = 0
    total_seconds = 0.0
    checksum = 0.0
    scores_per_pass = 0
    full_scores_per_pass = 0
    anchors_seen: list[int] = []
    for pass_index in range(args.passes):
        for batch_index, batch in enumerate(batches):
            batch_samples = int(batch["input_ids"].shape[0])
            padded_length = int(batch["input_ids"].shape[1])
            true_tokens = int(batch["attention_mask"].sum().item())
            synchronize(device)
            start = time.perf_counter()
            logits, anchor_count = run_once(batch)
            synchronize(device)
            elapsed = time.perf_counter() - start
            checksum += float(logits.float().sum().cpu())
            observed_memory.append(accelerator_memory_mb(device))
            total_samples += batch_samples
            total_true_tokens += true_tokens
            total_padded_tokens += batch_samples * padded_length
            total_seconds += elapsed
            anchors_seen.append(anchor_count)
            batch_measurements.append(
                {
                    "pass": pass_index,
                    "batch": batch_index,
                    "samples": batch_samples,
                    "true_tokens": true_tokens,
                    "padded_tokens": batch_samples * padded_length,
                    "padded_length": padded_length,
                    "latency_ms": elapsed * 1000.0,
                    "anchor_count": anchor_count,
                }
            )
            if pass_index == 0:
                scores_per_pass += score_count(
                    method=args.method,
                    batch_size=batch_samples,
                    padded_length=padded_length,
                    radius=args.window_radius,
                    anchors=anchor_count,
                )
                full_scores_per_pass += batch_samples * padded_length**2

    latencies = [float(row["latency_ms"]) for row in batch_measurements]
    truncated_lengths = [min(value, student.config.max_length) for value in natural_lengths]
    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_type": "real_input_dynamic_padding",
        "dataset": args.dataset,
        "data_file": str(args.data_file.resolve()),
        "method": args.method,
        "attention_backend": (
            "optimized_full_sdpa" if args.method == "full" else "structured_sparse"
        ),
        "selector_included": selector is not None,
        "selector_architecture": (
            selector.config.architecture if selector is not None else None
        ),
        "student_dir": str(args.student_dir.resolve()),
        "selector_dir": str(args.selector_dir.resolve()) if args.selector_dir else None,
        "samples": len(texts),
        "batch_size": args.batch_size,
        "batches_per_pass": len(batches),
        "passes": args.passes,
        "warmup": args.warmup,
        "warmup_shape_passes": int(args.warmup > 0),
        "length_bucketed": not args.preserve_order,
        "dynamic_padding": True,
        "tokenization_included_in_latency": False,
        "host_to_device_transfer_included": True,
        "max_length": student.config.max_length,
        "window_radius": args.window_radius,
        "anchor_budget": args.anchor_budget,
        "mean_actual_anchor_count": statistics.mean(anchors_seen),
        "natural_length_mean": statistics.mean(natural_lengths),
        "natural_length_median": statistics.median(natural_lengths),
        "natural_length_p95": percentile(
            [float(value) for value in natural_lengths], 0.95
        ),
        "natural_length_max": max(natural_lengths),
        "truncated_length_mean": statistics.mean(truncated_lengths),
        "truncated_fraction": sum(
            value > student.config.max_length for value in natural_lengths
        )
        / len(natural_lengths),
        "mean_padded_batch_length": statistics.mean(
            int(batch["input_ids"].shape[1]) for batch in batches
        ),
        "p95_padded_batch_length": percentile(
            [float(batch["input_ids"].shape[1]) for batch in batches], 0.95
        ),
        "max_padded_batch_length": max(
            int(batch["input_ids"].shape[1]) for batch in batches
        ),
        "padding_fraction": 1.0 - total_true_tokens / total_padded_tokens,
        "latency_mean_batch_ms": statistics.mean(latencies),
        "latency_median_batch_ms": statistics.median(latencies),
        "latency_p95_batch_ms": percentile(latencies, 0.95),
        "latency_std_batch_ms": statistics.stdev(latencies)
        if len(latencies) > 1
        else 0.0,
        "latency_mean_sample_ms": total_seconds * 1000.0 / total_samples,
        "throughput_samples_per_second": total_samples / total_seconds,
        "throughput_true_tokens_per_second": total_true_tokens / total_seconds,
        "throughput_padded_tokens_per_second": total_padded_tokens / total_seconds,
        "kernel_score_elements_per_pass": scores_per_pass,
        "full_score_elements_per_pass": full_scores_per_pass,
        "kernel_score_reduction": 1.0 - scores_per_pass / full_scores_per_pass,
        "process_rss_after_models_mb": rss_after_models,
        "process_peak_rss_mb": peak_rss_mb(),
        "accelerator_memory_after_models_mb": memory_after_models,
        "accelerator_memory_observed_peak_mb": {
            key: max(sample.get(key, 0.0) for sample in observed_memory)
            for key in {key for sample in observed_memory for key in sample}
        },
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "selector_parameters": (
            sum(parameter.numel() for parameter in selector.parameters())
            if selector is not None
            else 0
        ),
        "device": str(device),
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "checksum": checksum,
        "batch_measurements": batch_measurements,
    }
    if device.type == "cuda":
        result["accelerator_memory_observed_peak_mb"]["allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
        )
        result["accelerator_memory_observed_peak_mb"]["reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / (1024.0**2)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    printable = {key: value for key, value in result.items() if key != "batch_measurements"}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
