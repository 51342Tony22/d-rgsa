from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

SEEDS = (13, 42, 87, 101, 123)
METHOD_LABELS = {
    "full": "Full",
    "local": "Local",
    "random": "Random",
    "tfidf": "TF-IDF",
    "rationale": "Direct-RGSA",
    "distilled": "D-RGSA",
}
FROZEN_DATASETS = {
    "SST-2": ("sst2-6k-preliminary", "sst2-6k"),
    "TREC": ("trec", "trec"),
    "AG News": ("agnews-6k", "agnews-6k"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect quality, cost, and token-length results for RQ1-RQ5."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tokenizer")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_row(
    path: Path,
    *,
    dataset: str,
    method: str,
    source: str,
    seed: int,
) -> dict[str, Any]:
    metrics = read_json(path)
    return {
        "dataset": dataset,
        "method": method,
        "strategy": metrics.get("strategy"),
        "seed": seed,
        "source": source,
        "test_macro_f1": float(metrics["test_macro_f1"]),
        "test_accuracy": float(metrics["test_accuracy"]),
        "test_sparsity": float(metrics["test_sparsity"]),
        "test_connectivity_ratio": float(metrics["test_connectivity_ratio"]),
        "test_samples": int(metrics.get("test_samples", 0)),
        "train_samples": int(metrics.get("train_samples", 0)),
        "validation_samples": int(metrics.get("validation_samples", 0)),
        "window_radius": int(metrics.get("window_radius", 0)),
        "anchor_budget": int(metrics.get("anchor_budget", 0)),
        "attention_backend": metrics.get("attention_backend", "dense_mask"),
        "metrics_path": str(path),
    }


def collect_quality(project_root: Path, baseline_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, (baseline_dir, distilled_dir) in FROZEN_DATASETS.items():
        for strategy in ("full", "local", "random", "tfidf", "rationale"):
            for seed in SEEDS:
                path = (
                    baseline_root
                    / "outputs"
                    / baseline_dir
                    / strategy
                    / f"seed-{seed}"
                    / "metrics.json"
                )
                if path.exists():
                    rows.append(
                        metric_row(
                            path,
                            dataset=dataset,
                            method=METHOD_LABELS[strategy],
                            source="frozen_main",
                            seed=seed,
                        )
                    )
        for seed in SEEDS:
            path = (
                project_root
                / "outputs"
                / "d-rgsa-pilot"
                / distilled_dir
                / "deepseek"
                / "bigru"
                / f"seed-{seed}"
                / "metrics.json"
            )
            if path.exists():
                rows.append(
                    metric_row(
                        path,
                        dataset=dataset,
                        method="D-RGSA",
                        source="frozen_main",
                        seed=seed,
                    )
                )
    imdb_root = (
        project_root / "outputs" / "long-text-extension" / "quality" / "imdb-6k"
    )
    for strategy, method in METHOD_LABELS.items():
        for seed in SEEDS:
            path = imdb_root / strategy / f"seed-{seed}" / "metrics.json"
            if path.exists():
                rows.append(
                    metric_row(
                        path,
                        dataset="IMDb",
                        method=method,
                        source="long_text_extension",
                        seed=seed,
                    )
                )
    return rows


def summarize_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    groups = sorted({(row["dataset"], row["method"]) for row in rows})
    for dataset, method in groups:
        group = [
            row for row in rows if row["dataset"] == dataset and row["method"] == method
        ]
        record: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "runs": len(group),
            "seeds": " ".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
            "test_samples": statistics.mean(row["test_samples"] for row in group),
            "train_samples": statistics.mean(row["train_samples"] for row in group),
        }
        for field in (
            "test_macro_f1",
            "test_accuracy",
            "test_sparsity",
            "test_connectivity_ratio",
        ):
            values = [float(row[field]) for row in group]
            record[f"{field}_mean"] = statistics.mean(values)
            record[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary.append(record)
    return summary


def collect_selector_ablation(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configurations = [
        (
            "IMDb",
            "bigru",
            "D-RGSA (BiGRU)",
            project_root
            / "outputs"
            / "long-text-extension"
            / "quality"
            / "imdb-6k"
            / "distilled",
        ),
        (
            "IMDb",
            "mlp",
            "D-RGSA (MLP)",
            project_root
            / "outputs"
            / "long-text-extension"
            / "quality"
            / "imdb-6k-mlp"
            / "distilled",
        ),
        (
            "AG News",
            "bigru",
            "D-RGSA (BiGRU)",
            project_root
            / "outputs"
            / "d-rgsa-pilot"
            / "agnews-6k"
            / "deepseek"
            / "bigru",
        ),
        (
            "AG News",
            "mlp",
            "D-RGSA (MLP)",
            project_root
            / "outputs"
            / "long-text-extension"
            / "quality"
            / "agnews-6k-mlp"
            / "distilled",
        ),
    ]
    for dataset, architecture, method, root in configurations:
        for seed in SEEDS:
            path = root / f"seed-{seed}" / "metrics.json"
            if not path.exists():
                continue
            row = metric_row(
                path,
                dataset=dataset,
                method=method,
                source="selector_architecture_ablation",
                seed=seed,
            )
            row["selector_architecture"] = architecture
            rows.append(row)
    return rows


def selector_ablation_summary(
    project_root: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = summarize_quality(rows)
    architectures = {
        "D-RGSA (BiGRU)": "bigru",
        "D-RGSA (MLP)": "mlp",
    }
    for row in summary:
        architecture = architectures[row["method"]]
        if row["dataset"] == "IMDb":
            selector_root = (
                project_root
                / "outputs"
                / "long-text-extension"
                / "selectors"
                / "imdb-6k"
                / "deepseek"
                / architecture
                / "seed-42"
            )
        elif architecture == "bigru":
            selector_root = (
                project_root
                / "outputs"
                / "distilled-selectors"
                / "agnews-6k"
                / "deepseek"
                / "bigru"
                / "seed-42"
            )
        else:
            selector_root = (
                project_root
                / "outputs"
                / "long-text-extension"
                / "selectors"
                / "agnews-6k"
                / "deepseek"
                / "mlp"
                / "seed-42"
            )
        selector_path = selector_root / "selector_metrics.json"
        row["selector_architecture"] = architecture
        row["selector_metrics_path"] = str(selector_path)
        if selector_path.exists():
            metrics = read_json(selector_path)
            row["selector_test_anchor_f1"] = float(metrics["test"]["anchor_f1"])
            row["selector_validation_anchor_f1"] = float(
                metrics["validation"]["anchor_f1"]
            )
            row["selector_trainable_parameters"] = int(
                metrics["trainable_parameters"]
            )
    return summary


def paired_comparisons(
    quality_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from scipy import stats

    comparisons = [
        ("Direct-RGSA", "Full"),
        ("D-RGSA", "Full"),
        ("D-RGSA", "Direct-RGSA"),
        ("Direct-RGSA", "Random"),
        ("D-RGSA", "Random"),
        ("D-RGSA", "TF-IDF"),
        ("D-RGSA", "Local"),
    ]
    output: list[dict[str, Any]] = []

    def add_comparison(
        rows: list[dict[str, Any]],
        *,
        dataset: str,
        first: str,
        second: str,
        family: str,
    ) -> None:
        first_values = {
            int(row["seed"]): 100 * float(row["test_macro_f1"])
            for row in rows
            if row["dataset"] == dataset and row["method"] == first
        }
        second_values = {
            int(row["seed"]): 100 * float(row["test_macro_f1"])
            for row in rows
            if row["dataset"] == dataset and row["method"] == second
        }
        seeds = sorted(set(first_values) & set(second_values))
        if len(seeds) < 2:
            return
        differences = [first_values[seed] - second_values[seed] for seed in seeds]
        mean = statistics.mean(differences)
        standard_deviation = statistics.stdev(differences)
        standard_error = standard_deviation / len(differences) ** 0.5
        lower, upper = stats.t.interval(
            0.95,
            len(differences) - 1,
            loc=mean,
            scale=standard_error,
        )
        output.append(
            {
                "family": family,
                "dataset": dataset,
                "first_method": first,
                "second_method": second,
                "difference_definition": "first_method - second_method",
                "runs": len(seeds),
                "seeds": " ".join(str(seed) for seed in seeds),
                "mean_difference_macro_f1_points": mean,
                "difference_std": standard_deviation,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "paired_t_p_value": float(
                    stats.ttest_rel(
                        [first_values[seed] for seed in seeds],
                        [second_values[seed] for seed in seeds],
                    ).pvalue
                ),
            }
        )

    for dataset in sorted({row["dataset"] for row in quality_rows}):
        for first, second in comparisons:
            add_comparison(
                quality_rows,
                dataset=dataset,
                first=first,
                second=second,
                family="main_quality",
            )
    for dataset in ("AG News", "IMDb"):
        add_comparison(
            selector_rows,
            dataset=dataset,
            first="D-RGSA (MLP)",
            second="D-RGSA (BiGRU)",
            family="selector_architecture_ablation",
        )
    return output


def collect_cost(project_root: Path) -> list[dict[str, Any]]:
    root = project_root / "outputs" / "long-text-extension" / "cost" / "raw"
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")) if root.exists() else []:
        row = read_json(path)
        flattened = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        for prefix in (
            "accelerator_memory_after_models_mb",
            "accelerator_memory_observed_peak_mb",
        ):
            for key, value in row.get(prefix, {}).items():
                flattened[f"{prefix}_{key}"] = value
        flattened["raw_path"] = str(path)
        rows.append(flattened)
    return rows


def cost_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_length = {
        int(row["length"]): row for row in rows if row.get("method") == "full"
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        length = int(row["length"])
        full = full_by_length.get(length)
        if full is None:
            continue
        latency = float(row["latency_median_ms"])
        full_latency = float(full["latency_median_ms"])
        memory_key = (
            "accelerator_memory_observed_peak_mb_driver_mb"
            if "accelerator_memory_observed_peak_mb_driver_mb" in row
            else "process_peak_rss_mb"
        )
        memory = float(row[memory_key])
        full_memory = float(full[memory_key])
        allocated_key = "accelerator_memory_observed_peak_mb_allocated_mb"
        driver_key = "accelerator_memory_observed_peak_mb_driver_mb"
        allocated = float(row.get(allocated_key, 0.0))
        full_allocated = float(full.get(allocated_key, 0.0))
        driver = float(row.get(driver_key, 0.0))
        full_driver = float(full.get(driver_key, 0.0))
        process_rss = float(row["process_peak_rss_mb"])
        full_process_rss = float(full["process_peak_rss_mb"])
        comparisons.append(
            {
                "method": row["method"],
                "length": length,
                "batch_size": int(row["batch_size"]),
                "latency_median_ms": latency,
                "full_latency_median_ms": full_latency,
                "speedup_vs_full": full_latency / latency,
                "latency_change_vs_full_pct": 100 * (latency / full_latency - 1),
                "throughput_samples_per_second": float(
                    row["throughput_samples_per_second"]
                ),
                "memory_metric": memory_key,
                "observed_memory_mb": memory,
                "full_observed_memory_mb": full_memory,
                "memory_reduction_vs_full_pct": 100 * (1 - memory / full_memory),
                "observed_allocated_mb": allocated,
                "full_observed_allocated_mb": full_allocated,
                "allocated_reduction_vs_full_pct": (
                    100 * (1 - allocated / full_allocated)
                    if full_allocated
                    else 0.0
                ),
                "observed_driver_mb": driver,
                "full_observed_driver_mb": full_driver,
                "driver_reduction_vs_full_pct": (
                    100 * (1 - driver / full_driver) if full_driver else 0.0
                ),
                "process_peak_rss_mb": process_rss,
                "full_process_peak_rss_mb": full_process_rss,
                "process_rss_reduction_vs_full_pct": 100
                * (1 - process_rss / full_process_rss),
                "logical_sparsity_pct": 100 * float(row["logical_sparsity"]),
                "kernel_score_reduction_pct": 100
                * float(row["kernel_score_reduction"]),
                "selector_included": row["selector_included"],
                "device": row["device"],
            }
        )
    return comparisons


def collect_real_cost(project_root: Path) -> list[dict[str, Any]]:
    root = (
        project_root
        / "outputs"
        / "long-text-extension"
        / "real-cost"
        / "raw"
    )
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")) if root.exists() else []:
        row = read_json(path)
        flattened = {
            key: value
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
        for prefix in (
            "accelerator_memory_after_models_mb",
            "accelerator_memory_observed_peak_mb",
        ):
            for key, value in row.get(prefix, {}).items():
                flattened[f"{prefix}_{key}"] = value
        flattened["raw_path"] = str(path)
        rows.append(flattened)
    return rows


def real_cost_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_by_dataset = {
        str(row["dataset"]): row for row in rows if row.get("method") == "full"
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row["dataset"])
        full = full_by_dataset.get(dataset)
        if full is None:
            continue
        latency = float(row["latency_median_batch_ms"])
        full_latency = float(full["latency_median_batch_ms"])
        memory_key = (
            "accelerator_memory_observed_peak_mb_driver_mb"
            if "accelerator_memory_observed_peak_mb_driver_mb" in row
            else "process_peak_rss_mb"
        )
        memory = float(row[memory_key])
        full_memory = float(full[memory_key])
        allocated_key = "accelerator_memory_observed_peak_mb_allocated_mb"
        driver_key = "accelerator_memory_observed_peak_mb_driver_mb"
        allocated = float(row.get(allocated_key, 0.0))
        full_allocated = float(full.get(allocated_key, 0.0))
        driver = float(row.get(driver_key, 0.0))
        full_driver = float(full.get(driver_key, 0.0))
        process_rss = float(row["process_peak_rss_mb"])
        full_process_rss = float(full["process_peak_rss_mb"])
        comparisons.append(
            {
                "dataset": dataset,
                "method": row["method"],
                "selector_architecture": row.get("selector_architecture"),
                "samples": int(row["samples"]),
                "batch_size": int(row["batch_size"]),
                "max_length": int(row["max_length"]),
                "mean_padded_batch_length": float(
                    row["mean_padded_batch_length"]
                ),
                "latency_median_batch_ms": latency,
                "latency_p95_batch_ms": float(row["latency_p95_batch_ms"]),
                "latency_mean_sample_ms": float(row["latency_mean_sample_ms"]),
                "speedup_vs_full": full_latency / latency,
                "latency_change_vs_full_pct": 100 * (latency / full_latency - 1),
                "throughput_samples_per_second": float(
                    row["throughput_samples_per_second"]
                ),
                "throughput_true_tokens_per_second": float(
                    row["throughput_true_tokens_per_second"]
                ),
                "observed_memory_mb": memory,
                "full_observed_memory_mb": full_memory,
                "memory_metric": memory_key,
                "memory_reduction_vs_full_pct": 100 * (1 - memory / full_memory),
                "observed_allocated_mb": allocated,
                "full_observed_allocated_mb": full_allocated,
                "allocated_reduction_vs_full_pct": (
                    100 * (1 - allocated / full_allocated)
                    if full_allocated
                    else 0.0
                ),
                "observed_driver_mb": driver,
                "full_observed_driver_mb": full_driver,
                "driver_reduction_vs_full_pct": (
                    100 * (1 - driver / full_driver) if full_driver else 0.0
                ),
                "process_peak_rss_mb": process_rss,
                "full_process_peak_rss_mb": full_process_rss,
                "process_rss_reduction_vs_full_pct": 100
                * (1 - process_rss / full_process_rss),
                "kernel_score_reduction_pct": 100
                * float(row["kernel_score_reduction"]),
                "padding_fraction_pct": 100 * float(row["padding_fraction"]),
                "truncated_fraction_pct": 100 * float(row["truncated_fraction"]),
                "device": row["device"],
                "raw_path": row["raw_path"],
            }
        )
    return comparisons


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def collect_dataset_stats(project_root: Path, tokenizer_name: str) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # We are measuring the natural length distribution, not feeding these
    # sequences into the 512-position model in this routine.
    tokenizer.model_max_length = 1_000_000
    dataset_dirs = {
        "SST-2": "sst2-6k",
        "TREC": "trec",
        "AG News": "agnews-6k",
        "IMDb": "imdb-6k",
    }
    rows: list[dict[str, Any]] = []
    for dataset, directory in dataset_dirs.items():
        for split in ("train", "validation", "test"):
            path = project_root / "data" / directory / f"{split}.jsonl"
            if not path.exists():
                continue
            records = read_jsonl(path)
            lengths: list[int] = []
            for start in range(0, len(records), 128):
                encoded = tokenizer(
                    [row["text"] for row in records[start : start + 128]],
                    add_special_tokens=True,
                    truncation=False,
                    return_length=True,
                )
                lengths.extend(int(value) for value in encoded["length"])
            labels = Counter(str(row["gold_label"]) for row in records)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "samples": len(records),
                    "label_distribution": json.dumps(
                        dict(sorted(labels.items())),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "tokens_mean": statistics.mean(lengths),
                    "tokens_median": statistics.median(lengths),
                    "tokens_p90": percentile(lengths, 0.90),
                    "tokens_p95": percentile(lengths, 0.95),
                    "tokens_max": max(lengths),
                    "fraction_over_128": sum(value > 128 for value in lengths)
                    / len(lengths),
                    "fraction_over_256": sum(value > 256 for value in lengths)
                    / len(lengths),
                    "fraction_over_512": sum(value > 512 for value in lengths)
                    / len(lengths),
                    "fraction_over_1024": sum(value > 1024 for value in lengths)
                    / len(lengths),
                }
            )
    return rows


def main() -> None:
    args = arguments()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir or (
        project_root / "outputs" / "long-text-extension" / "tables"
    )
    tokenizer = args.tokenizer or str(
        project_root
        / "outputs"
        / "distilled-selectors"
        / "sst2-6k"
        / "deepseek"
        / "bigru"
        / "seed-42"
    )
    quality = collect_quality(project_root, args.baseline_root.resolve())
    quality_summary = summarize_quality(quality)
    selector_ablation = collect_selector_ablation(project_root)
    selector_summary = selector_ablation_summary(project_root, selector_ablation)
    paired = paired_comparisons(quality, selector_ablation)
    cost = collect_cost(project_root)
    cost_comparison = cost_comparisons(cost)
    real_cost = collect_real_cost(project_root)
    real_cost_comparison = real_cost_comparisons(real_cost)
    dataset_stats = collect_dataset_stats(project_root, tokenizer)
    write_csv(output_dir / "quality_raw.csv", quality)
    write_csv(output_dir / "quality_summary.csv", quality_summary)
    write_csv(output_dir / "selector_ablation_raw.csv", selector_ablation)
    write_csv(output_dir / "selector_ablation_summary.csv", selector_summary)
    write_csv(output_dir / "paired_comparisons.csv", paired)
    write_csv(output_dir / "cost_raw.csv", cost)
    write_csv(output_dir / "cost_comparison_vs_full.csv", cost_comparison)
    write_csv(output_dir / "real_input_cost_raw.csv", real_cost)
    write_csv(
        output_dir / "real_input_cost_comparison_vs_full.csv",
        real_cost_comparison,
    )
    write_csv(output_dir / "dataset_stats.csv", dataset_stats)
    status = {
        "quality_runs": len(quality),
        "quality_summary_rows": len(quality_summary),
        "selector_ablation_runs": len(selector_ablation),
        "selector_ablation_summary_rows": len(selector_summary),
        "paired_comparison_rows": len(paired),
        "cost_records": len(cost),
        "real_input_cost_records": len(real_cost),
        "dataset_stat_rows": len(dataset_stats),
        "expected_seeds": list(SEEDS),
        "imdb_quality_complete": sum(
            row["dataset"] == "IMDb" for row in quality
        )
        == len(METHOD_LABELS) * len(SEEDS),
        "imdb_selector_ablation_complete": sum(
            row["dataset"] == "IMDb" for row in selector_ablation
        )
        == 2 * len(SEEDS),
        "agnews_selector_ablation_complete": sum(
            row["dataset"] == "AG News" for row in selector_ablation
        )
        == 2 * len(SEEDS),
        "real_input_cost_complete": len(real_cost) == 8,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "collection_status.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(status, handle, indent=2, sort_keys=True)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
