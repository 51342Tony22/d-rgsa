from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = ("test_accuracy", "test_macro_f1", "test_sparsity")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate RGSA ablation runs.")
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-seeds", nargs="+", type=int, default=[13, 42, 87])
    return parser.parse_args()


def load_metrics(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("metrics.json")):
        with path.open(encoding="utf-8") as handle:
            row = json.load(handle)
        row["run"] = str(path.parent.relative_to(root))
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"runs": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def main() -> None:
    args = arguments()
    ablation_rows = load_metrics(args.ablation_root)
    main_rows = [
        row
        for row in load_metrics(args.main_root)
        if int(row.get("seed", -1)) in args.main_seeds
    ]
    if not ablation_rows:
        raise ValueError(f"No ablation metrics found under {args.ablation_root}")

    budget_rows = [row for row in ablation_rows if row["run"].startswith("anchor-budget-")]
    budget_rows.extend(
        row
        for row in main_rows
        if row.get("strategy") == "rationale"
        and int(row.get("anchor_budget", -1)) == 4
        and int(row.get("window_radius", -1)) == 4
    )

    radius_rows = [row for row in ablation_rows if row["run"].startswith("window-radius-")]
    radius_rows.extend(
        row
        for row in main_rows
        if row.get("strategy") in {"local", "rationale"}
        and int(row.get("window_radius", -1)) == 4
        and int(row.get("anchor_budget", -1)) == 4
    )

    budgets = {
        str(budget): aggregate(
            [row for row in budget_rows if int(row["anchor_budget"]) == budget]
        )
        for budget in sorted({int(row["anchor_budget"]) for row in budget_rows})
    }
    radii: dict[str, dict[str, dict[str, float | int]]] = {}
    for radius in sorted({int(row["window_radius"]) for row in radius_rows}):
        radii[str(radius)] = {
            strategy: aggregate(
                [
                    row
                    for row in radius_rows
                    if int(row["window_radius"]) == radius and row["strategy"] == strategy
                ]
            )
            for strategy in ("local", "rationale")
        }

    output = {
        "anchor_budget": budgets,
        "window_radius": radii,
        "main_seeds_used_for_shared_settings": args.main_seeds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(f"Wrote ablation summary to {args.output}")


if __name__ == "__main__":
    main()
