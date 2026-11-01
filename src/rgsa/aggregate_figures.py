from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STRATEGY_ORDER = ("full", "local", "random", "tfidf", "rationale")
STRATEGY_LABELS = {
    "full": "Full",
    "local": "Local",
    "random": "Random",
    "tfidf": "TF-IDF",
    "rationale": "RGSA",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cross-dataset summary tables and paper-style figures."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset mapping in the form NAME=/path/to/results.summary.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name.strip():
        raise ValueError(f"Dataset name is empty in {value!r}")
    return name.strip(), Path(path).expanduser()


def load_rows(dataset_args: list[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for value in dataset_args:
        dataset, path = parse_dataset_argument(value)
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        for strategy in STRATEGY_ORDER:
            if strategy not in summary:
                continue
            metrics = summary[strategy]
            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "runs": int(metrics.get("runs", 0)),
                    "accuracy_mean": float(metrics.get("test_accuracy_mean", 0.0)),
                    "accuracy_std": float(metrics.get("test_accuracy_std", 0.0)),
                    "macro_f1_mean": float(metrics.get("test_macro_f1_mean", 0.0)),
                    "macro_f1_std": float(metrics.get("test_macro_f1_std", 0.0)),
                    "sparsity_mean": float(metrics.get("test_sparsity_mean", 0.0)),
                    "sparsity_std": float(metrics.get("test_sparsity_std", 0.0)),
                    "connectivity_mean": float(
                        metrics.get("test_connectivity_ratio_mean", 0.0)
                    ),
                    "connectivity_std": float(
                        metrics.get("test_connectivity_ratio_std", 0.0)
                    ),
                }
            )
    return rows


def write_tables(rows: list[dict[str, float | int | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "strategy",
        "strategy_label",
        "runs",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "sparsity_mean",
        "sparsity_std",
        "connectivity_mean",
        "connectivity_std",
    ]
    with (output_dir / "cross_dataset_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "cross_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def plot_grouped_metric(
    rows: list[dict[str, float | int | str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    values = {
        (str(row["dataset"]), str(row["strategy"])): (
            float(row[f"{metric}_mean"]),
            float(row[f"{metric}_std"]),
        )
        for row in rows
    }
    x = np.arange(len(datasets))
    width = 0.15
    offsets = np.linspace(
        -width * (len(STRATEGY_ORDER) - 1) / 2,
        width * (len(STRATEGY_ORDER) - 1) / 2,
        len(STRATEGY_ORDER),
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for offset, strategy in zip(offsets, STRATEGY_ORDER, strict=True):
        means = [values.get((dataset, strategy), (0.0, 0.0))[0] * 100 for dataset in datasets]
        stds = [values.get((dataset, strategy), (0.0, 0.0))[1] * 100 for dataset in datasets]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=STRATEGY_LABELS[strategy],
        )
    ax.set_xticks(x, datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=len(STRATEGY_ORDER), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_rgsa_gain(rows: list[dict[str, float | int | str]], output: Path) -> None:
    import matplotlib.pyplot as plt

    datasets = list(dict.fromkeys(str(row["dataset"]) for row in rows))
    lookup = {
        (str(row["dataset"]), str(row["strategy"])): float(row["macro_f1_mean"])
        for row in rows
    }
    baselines = ("full", "local", "random", "tfidf")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for baseline in baselines:
        gains = [
            (lookup.get((dataset, "rationale"), 0.0) - lookup.get((dataset, baseline), 0.0))
            * 100
            for dataset in datasets
        ]
        ax.plot(datasets, gains, marker="o", linewidth=2, label=f"RGSA - {STRATEGY_LABELS[baseline]}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Macro-F1 gain (percentage points)")
    ax.set_title("RGSA gain over baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main() -> None:
    args = arguments()
    rows = load_rows(args.dataset)
    if not rows:
        raise ValueError("No summary rows found")
    write_tables(rows, args.output_dir)
    plot_grouped_metric(
        rows,
        metric="accuracy",
        ylabel="Accuracy (%)",
        title="Accuracy across datasets",
        output=args.output_dir / "accuracy_by_dataset.png",
    )
    plot_grouped_metric(
        rows,
        metric="macro_f1",
        ylabel="Macro-F1 (%)",
        title="Macro-F1 across datasets",
        output=args.output_dir / "macro_f1_by_dataset.png",
    )
    plot_grouped_metric(
        rows,
        metric="sparsity",
        ylabel="Attention sparsity (%)",
        title="Theoretical attention sparsity",
        output=args.output_dir / "sparsity_by_dataset.png",
    )
    plot_rgsa_gain(rows, args.output_dir / "rgsa_f1_gain.png")
    print(f"Wrote cross-dataset tables and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
