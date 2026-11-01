from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics and compute recovery rate.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows: list[dict] = []
    for path in sorted(args.input_root.rglob("metrics.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["run"] = str(path.parent.relative_to(args.input_root))
        rows.append(row)
    if not rows:
        raise ValueError(f"No metrics.json files found under {args.input_root}")
    metric_name = (
        "test_macro_f1" if any("test_macro_f1" in row for row in rows) else "eval_macro_f1"
    )
    f1_by_strategy: dict[str, list[float]] = {}
    for row in rows:
        f1_by_strategy.setdefault(row["strategy"], []).append(float(row[metric_name]))
    means = {name: sum(values) / len(values) for name, values in f1_by_strategy.items()}
    denominator = means.get("full", 0.0) - means.get("local", 0.0)
    for row in rows:
        if row["strategy"] == "rationale" and denominator != 0:
            row["recovery_rate"] = (
                float(row[metric_name]) - means["local"]
            ) / denominator
        else:
            row["recovery_rate"] = ""
    fields = [
        "run",
        "strategy",
        "eval_accuracy",
        "eval_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "sparsity",
        "connectivity_ratio",
        "active_connections",
        "full_connections",
        "test_sparsity",
        "test_connectivity_ratio",
        "eval_runtime",
        "window_radius",
        "anchor_budget",
        "seed",
        "trainable_parameters",
        "train_samples",
        "validation_samples",
        "test_samples",
        "recovery_rate",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, float | int]] = {}
    reported_metrics = (
        "test_accuracy",
        "test_macro_f1",
        "test_sparsity",
        "test_connectivity_ratio",
    )
    for strategy, strategy_rows in sorted(
        ((name, [row for row in rows if row["strategy"] == name]) for name in f1_by_strategy)
    ):
        aggregate: dict[str, float | int] = {"runs": len(strategy_rows)}
        for metric in reported_metrics:
            values = [float(row[metric]) for row in strategy_rows if metric in row]
            if values:
                aggregate[f"{metric}_mean"] = statistics.mean(values)
                aggregate[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
        summary[strategy] = aggregate
    if {"full", "local", "rationale"} <= set(means) and denominator != 0:
        summary["rationale"]["recovery_rate_from_means"] = (
            means["rationale"] - means["local"]
        ) / denominator
        summary["rationale"]["f1_gap_vs_full"] = means["rationale"] - means["full"]
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Wrote {len(rows)} runs to {args.output}")
    print(f"Wrote aggregate summary to {summary_path}")


if __name__ == "__main__":
    main()
