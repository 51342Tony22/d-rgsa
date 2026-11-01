from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

METHOD_ORDER = ["Full", "Local", "Random", "TF-IDF", "Direct-RGSA", "D-RGSA"]
METHOD_COLORS = {
    "Full": "#4C78A8",
    "Local": "#F58518",
    "Random": "#54A24B",
    "TF-IDF": "#B279A2",
    "Direct-RGSA": "#E45756",
    "D-RGSA": "#2E8B8B",
}
COST_LABELS = {
    "full": "Full",
    "local": "Local",
    "d-rgsa": "D-RGSA (BiGRU, end-to-end)",
    "d-rgsa-mlp": "D-RGSA (MLP, end-to-end)",
    "d-rgsa-student": "D-RGSA student only",
}
COST_ORDER = list(COST_LABELS.values())
COST_COLORS = [
    METHOD_COLORS["Full"],
    METHOD_COLORS["Local"],
    METHOD_COLORS["D-RGSA"],
    "#72B7B2",
    "#1B6F6A",
]
REAL_COST_LABELS = {
    "full": "Full",
    "local": "Local",
    "d-rgsa": "D-RGSA (BiGRU)",
    "d-rgsa-mlp": "D-RGSA (MLP)",
}
REAL_COST_ORDER = list(REAL_COST_LABELS.values())
REAL_COST_COLORS = COST_COLORS[:4]
SELECTOR_ORDER = ["D-RGSA (BiGRU)", "D-RGSA (MLP)"]
SELECTOR_COLORS = [METHOD_COLORS["D-RGSA"], "#72B7B2"]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Altair paper figures for RQ4 and RQ5."
    )
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    return float(value) if value not in {None, ""} else default


def save_chart(chart: Any, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_base.with_suffix(".html")))
    for suffix in (".svg", ".png"):
        try:
            chart.save(str(output_base.with_suffix(suffix)), scale_factor=2)
        except TypeError:
            chart.save(str(output_base.with_suffix(suffix)))


def long_text_quality_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    values = []
    for row in rows:
        if row["dataset"] not in {"AG News", "IMDb"}:
            continue
        mean = 100 * number(row, "test_macro_f1_mean")
        std = 100 * number(row, "test_macro_f1_std")
        values.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "macro_f1": mean,
                "lower": mean - std,
                "upper": mean + std,
                "runs": int(number(row, "runs")),
                "accuracy": 100 * number(row, "test_accuracy_mean"),
                "sparsity": 100 * number(row, "test_sparsity_mean"),
            }
        )
    domain = [name for name in METHOD_ORDER if any(row["method"] == name for row in values)]
    base = alt.Chart(alt.Data(values=values)).encode(
        y=alt.Y("method:N", sort=domain, title=None),
        color=alt.Color(
            "method:N",
            scale=alt.Scale(
                domain=domain,
                range=[METHOD_COLORS[name] for name in domain],
            ),
            legend=None,
        ),
        tooltip=[
            "dataset:N",
            "method:N",
            alt.Tooltip("macro_f1:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("accuracy:Q", format=".2f", title="Accuracy"),
            alt.Tooltip("sparsity:Q", format=".2f", title="Sparsity"),
            "runs:Q",
        ],
    )
    intervals = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("lower:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=105).encode(
        x=alt.X("macro_f1:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False))
    )
    return (
        (intervals + points)
        .properties(width=250, height=190)
        .facet(column=alt.Column("dataset:N", title=None, sort=["AG News", "IMDb"]))
        .resolve_scale(x="independent")
        .properties(title="Medium- and long-text classification quality")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
        .configure_title(fontSize=16, anchor="start")
    )


def cost_scaling_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    values = []
    for row in rows:
        method = COST_LABELS.get(row.get("method", ""))
        if method is None:
            continue
        memory = number(
            row,
            "accelerator_memory_observed_peak_mb_driver_mb",
            number(
                row,
                "accelerator_memory_observed_peak_mb_allocated_mb",
                number(row, "process_peak_rss_mb"),
            ),
        )
        values.append(
            {
                "method": method,
                "length": int(number(row, "length")),
                "latency": number(row, "latency_median_ms"),
                "throughput": number(row, "throughput_samples_per_second"),
                "memory": memory,
                "logical_sparsity": 100 * number(row, "logical_sparsity"),
                "score_reduction": 100 * number(row, "kernel_score_reduction"),
                "device": row.get("device", ""),
                "batch_size": int(number(row, "batch_size")),
            }
        )
    color = alt.Color(
        "method:N",
        scale=alt.Scale(
            domain=COST_ORDER,
            range=COST_COLORS,
        ),
        title=None,
    )
    common = {
        "x": alt.X(
            "length:Q",
            title="Sequence length",
            scale=alt.Scale(type="log", base=2),
            axis=alt.Axis(values=[128, 256, 512, 1024, 2048], format="d"),
        ),
        "color": color,
        "tooltip": [
            "method:N",
            "length:Q",
            alt.Tooltip("latency:Q", format=".2f", title="Median latency (ms)"),
            alt.Tooltip("throughput:Q", format=".2f", title="Samples/s"),
            alt.Tooltip("memory:Q", format=".2f", title="Observed memory (MB)"),
            alt.Tooltip("logical_sparsity:Q", format=".2f", title="Sparsity (%)"),
            "device:N",
            "batch_size:Q",
        ],
    }
    latency = (
        alt.Chart(alt.Data(values=values))
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.5)
        .encode(
            **common,
            y=alt.Y(
                "latency:Q",
                title="Median end-to-end latency (ms)",
                scale=alt.Scale(type="log"),
            ),
        )
        .properties(width=300, height=230, title="(a) Latency")
    )
    memory = (
        alt.Chart(alt.Data(values=values))
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.5)
        .encode(
            **common,
            y=alt.Y(
                "memory:Q",
                title="Observed accelerator-driver memory (MB)",
                scale=alt.Scale(zero=False),
            ),
        )
        .properties(width=300, height=230, title="(b) Memory")
    )
    return (
        alt.hconcat(latency, memory)
        .resolve_scale(color="shared")
        .properties(title="Inference-cost scaling with sequence length")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_title(fontSize=15, anchor="start")
        .configure_legend(labelFontSize=12)
    )


def score_reduction_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    values = []
    for row in rows:
        method = COST_LABELS.get(row.get("method", ""))
        if method is None:
            continue
        values.append(
            {
                "method": method,
                "length": int(number(row, "length")),
                "logical_sparsity": 100 * number(row, "logical_sparsity"),
                "score_reduction": 100 * number(row, "kernel_score_reduction"),
            }
        )
    return (
        alt.Chart(alt.Data(values=values))
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.5)
        .encode(
            x=alt.X(
                "length:Q",
                title="Sequence length",
                scale=alt.Scale(type="log", base=2),
                axis=alt.Axis(values=[128, 256, 512, 1024, 2048], format="d"),
            ),
            y=alt.Y(
                "score_reduction:Q",
                title="Attention-score reduction vs Full (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=COST_ORDER,
                    range=COST_COLORS,
                ),
                title=None,
            ),
            tooltip=[
                "method:N",
                "length:Q",
                alt.Tooltip("score_reduction:Q", format=".2f"),
                alt.Tooltip("logical_sparsity:Q", format=".2f"),
            ],
        )
        .properties(
            width=620,
            height=260,
            title="Structured sparse computation removes an increasing share of scores",
        )
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_title(fontSize=16, anchor="start")
    )


def selector_architecture_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    values = []
    for row in rows:
        mean = 100 * number(row, "test_macro_f1_mean")
        std = 100 * number(row, "test_macro_f1_std")
        values.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "macro_f1": mean,
                "lower": mean - std,
                "upper": mean + std,
                "accuracy": 100 * number(row, "test_accuracy_mean"),
                "sparsity": 100 * number(row, "test_sparsity_mean"),
                "anchor_f1": 100 * number(row, "selector_test_anchor_f1"),
                "runs": int(number(row, "runs")),
            }
        )
    base = alt.Chart(alt.Data(values=values)).encode(
        y=alt.Y("method:N", sort=SELECTOR_ORDER, title=None),
        color=alt.Color(
            "method:N",
            scale=alt.Scale(domain=SELECTOR_ORDER, range=SELECTOR_COLORS),
            legend=None,
        ),
        tooltip=[
            "method:N",
            alt.Tooltip("macro_f1:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("accuracy:Q", format=".2f", title="Accuracy"),
            alt.Tooltip("sparsity:Q", format=".2f", title="Sparsity"),
            alt.Tooltip("anchor_f1:Q", format=".2f", title="Selector anchor-F1"),
            "runs:Q",
        ],
    )
    intervals = base.mark_rule(strokeWidth=3).encode(
        x=alt.X(
            "lower:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        ),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=150).encode(
        x=alt.X(
            "macro_f1:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        )
    )
    labels = points.mark_text(dx=10, align="left", fontSize=12).encode(
        text=alt.Text("macro_f1:Q", format=".2f")
    )
    return (
        (intervals + points + labels)
        .properties(width=260, height=125)
        .facet(column=alt.Column("dataset:N", title=None, sort=["AG News", "IMDb"]))
        .resolve_scale(x="independent")
        .properties(title="Selector-architecture ablation (mean ± s.d., five seeds)")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
        .configure_title(fontSize=16, anchor="start")
    )


def performance_cost_chart(
    quality_rows: list[dict[str, Any]],
    selector_rows: list[dict[str, Any]],
    real_cost_rows: list[dict[str, Any]],
) -> Any | None:
    import altair as alt

    real_latency = {
        (row["dataset"], REAL_COST_LABELS[row["method"]]): number(
            row, "latency_median_batch_ms"
        )
        for row in real_cost_rows
        if row.get("method") in REAL_COST_LABELS
    }
    values = []
    for row in quality_rows:
        if row["dataset"] not in {"AG News", "IMDb"}:
            continue
        if row["method"] not in {"Full", "Local", "D-RGSA"}:
            continue
        cost_method = row["method"]
        if cost_method == "D-RGSA":
            cost_method = "D-RGSA (BiGRU)"
        key = (row["dataset"], cost_method)
        if key not in real_latency:
            continue
        values.append(
            {
                "dataset": row["dataset"],
                "method": cost_method,
                "macro_f1": 100 * number(row, "test_macro_f1_mean"),
                "latency": real_latency[key],
                "sparsity": 100 * number(row, "test_sparsity_mean"),
            }
        )
    for row in selector_rows:
        if row["method"] != "D-RGSA (MLP)":
            continue
        cost_method = "D-RGSA (MLP)"
        key = (row["dataset"], cost_method)
        if key not in real_latency:
            continue
        values.append(
            {
                "dataset": row["dataset"],
                "method": cost_method,
                "macro_f1": 100 * number(row, "test_macro_f1_mean"),
                "latency": real_latency[key],
                "sparsity": 100 * number(row, "test_sparsity_mean"),
            }
        )
    if not values:
        return None
    points = (
        alt.Chart(alt.Data(values=values))
        .mark_point(filled=True, size=120)
        .encode(
            x=alt.X(
                "latency:Q",
                title="Real-input median batch latency (ms)",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("macro_f1:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "method:N",
                scale=alt.Scale(
                    domain=REAL_COST_ORDER,
                    range=REAL_COST_COLORS,
                ),
                title=None,
            ),
            tooltip=[
                "dataset:N",
                "method:N",
                alt.Tooltip("macro_f1:Q", format=".2f"),
                alt.Tooltip("latency:Q", format=".2f"),
                alt.Tooltip("sparsity:Q", format=".2f"),
            ],
        )
    )
    labels = points.mark_text(dx=9, align="left", fontSize=11).encode(text="method:N")
    return (
        (points + labels)
        .properties(width=300, height=240)
        .facet(column=alt.Column("dataset:N", title=None, sort=["AG News", "IMDb"]))
        .resolve_scale(y="independent")
        .properties(title="Classification quality versus measured inference cost")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
        .configure_title(fontSize=16, anchor="start")
    )


def real_input_cost_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    values = []
    for row in rows:
        method = REAL_COST_LABELS.get(row.get("method", ""))
        if method is None:
            continue
        memory = number(
            row,
            "accelerator_memory_observed_peak_mb_driver_mb",
            number(
                row,
                "accelerator_memory_observed_peak_mb_allocated_mb",
                number(row, "process_peak_rss_mb"),
            ),
        )
        values.append(
            {
                "dataset": row["dataset"],
                "method": method,
                "latency": number(row, "latency_median_batch_ms"),
                "latency_p95": number(row, "latency_p95_batch_ms"),
                "throughput": number(row, "throughput_samples_per_second"),
                "memory": memory,
                "score_reduction": 100 * number(row, "kernel_score_reduction"),
                "mean_padded_length": number(row, "mean_padded_batch_length"),
            }
        )
    color = alt.Color(
        "method:N",
        scale=alt.Scale(domain=REAL_COST_ORDER, range=REAL_COST_COLORS),
        title=None,
    )
    tooltip = [
        "dataset:N",
        "method:N",
        alt.Tooltip("latency:Q", format=".2f", title="Median latency (ms)"),
        alt.Tooltip("latency_p95:Q", format=".2f", title="P95 latency (ms)"),
        alt.Tooltip("throughput:Q", format=".2f", title="Samples/s"),
        alt.Tooltip("memory:Q", format=".2f", title="Observed memory (MB)"),
        alt.Tooltip("score_reduction:Q", format=".2f", title="Score reduction (%)"),
        alt.Tooltip("mean_padded_length:Q", format=".2f", title="Mean padded length"),
    ]
    latency = (
        alt.Chart(alt.Data(values=values))
        .mark_bar()
        .encode(
            x=alt.X(
                "method:N",
                sort=REAL_COST_ORDER,
                title=None,
                axis=alt.Axis(labelAngle=-20),
            ),
            y=alt.Y("latency:Q", title="Median batch latency (ms)"),
            color=color,
            tooltip=tooltip,
        )
        .properties(width=270, height=230)
        .facet(column=alt.Column("dataset:N", title=None, sort=["AG News", "IMDb"]))
        .resolve_scale(y="independent")
    )
    return (
        latency.properties(title="Real-input end-to-end inference cost")
        .configure_axis(labelFontSize=11, titleFontSize=13)
        .configure_header(labelFontSize=14)
        .configure_title(fontSize=16, anchor="start")
    )


def paper_extension_summary_chart(
    selector_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
) -> Any:
    """Compact two-panel figure for the paper's RQ4--RQ5 extension."""
    import altair as alt

    selector_values = []
    selector_order = []
    for dataset in ("AG News", "IMDb"):
        for method, short_name in (
            ("D-RGSA (BiGRU)", "BiGRU"),
            ("D-RGSA (MLP)", "MLP"),
        ):
            matching = [
                row
                for row in selector_rows
                if row.get("dataset") == dataset and row.get("method") == method
            ]
            if not matching:
                continue
            row = matching[0]
            mean = 100 * number(row, "test_macro_f1_mean")
            std = 100 * number(row, "test_macro_f1_std")
            label = f"{dataset} · {short_name}"
            selector_order.append(label)
            selector_values.append(
                {
                    "condition": label,
                    "selector": short_name,
                    "macro_f1": mean,
                    "lower": mean - std,
                    "upper": mean + std,
                }
            )

    selector_base = alt.Chart(alt.Data(values=selector_values)).encode(
        y=alt.Y("condition:N", sort=selector_order, title=None),
        color=alt.Color(
            "selector:N",
            scale=alt.Scale(domain=["BiGRU", "MLP"], range=SELECTOR_COLORS),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            "condition:N",
            alt.Tooltip("macro_f1:Q", format=".2f", title="Macro-F1"),
        ],
    )
    selector_intervals = selector_base.mark_rule(strokeWidth=2).encode(
        x=alt.X(
            "lower:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False, padding=18),
        ),
        x2="upper:Q",
    )
    selector_points = selector_base.mark_point(filled=True, size=105).encode(
        x=alt.X(
            "macro_f1:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False, padding=18),
        )
    )
    selector_panel = (selector_intervals + selector_points).properties(
        width=285,
        height=175,
    )

    cost_values = []
    cost_names = {
        "full": "Full",
        "d-rgsa-mlp": "D-RGSA (MLP)",
    }
    for row in cost_rows:
        method = cost_names.get(row.get("method", ""))
        if method is None:
            continue
        cost_values.append(
            {
                "method": method,
                "length": int(number(row, "length")),
                "latency": number(row, "latency_median_ms"),
            }
        )
    cost_color = alt.Color(
        "method:N",
        scale=alt.Scale(
            domain=["Full", "D-RGSA (MLP)"],
            range=[METHOD_COLORS["Full"], "#72B7B2"],
        ),
        legend=alt.Legend(title=None, orient="bottom"),
    )
    cost_base = alt.Chart(alt.Data(values=cost_values)).encode(
        x=alt.X(
            "length:Q",
            title="Sequence length",
            scale=alt.Scale(type="log", base=2),
            axis=alt.Axis(values=[128, 256, 512, 1024, 2048], format="d"),
        ),
        y=alt.Y(
            "latency:Q",
            title="Median batch latency (ms)",
            scale=alt.Scale(type="log"),
        ),
        color=cost_color,
        tooltip=[
            "method:N",
            "length:Q",
            alt.Tooltip("latency:Q", format=".2f", title="Latency (ms)"),
        ],
    )
    cost_panel = (cost_base.mark_line(strokeWidth=2.5) + cost_base.mark_point(
        filled=True, size=85
    )).properties(
        width=285,
        height=175,
    )

    return (
        alt.hconcat(selector_panel, cost_panel, spacing=34)
        .resolve_scale(color="independent")
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, symbolSize=100)
        .configure_view(stroke="#D9D9D9")
    )


def main() -> None:
    args = arguments()
    quality = read_csv(args.tables_dir / "quality_summary.csv")
    selector_ablation = read_csv(args.tables_dir / "selector_ablation_summary.csv")
    cost = read_csv(args.tables_dir / "cost_raw.csv")
    real_cost = read_csv(args.tables_dir / "real_input_cost_raw.csv")
    if quality:
        save_chart(
            long_text_quality_chart(quality),
            args.output_dir / "long_text_macro_f1",
        )
    if cost:
        save_chart(cost_scaling_chart(cost), args.output_dir / "cost_scaling")
        save_chart(
            score_reduction_chart(cost),
            args.output_dir / "score_reduction",
        )
    if real_cost:
        save_chart(
            real_input_cost_chart(real_cost),
            args.output_dir / "real_input_cost",
        )
        pareto = performance_cost_chart(quality, selector_ablation, real_cost)
        if pareto is not None:
            save_chart(pareto, args.output_dir / "performance_cost_pareto")
    if selector_ablation:
        save_chart(
            selector_architecture_chart(selector_ablation),
            args.output_dir / "selector_architecture_medium_long",
        )
    if selector_ablation and cost:
        save_chart(
            paper_extension_summary_chart(selector_ablation, cost),
            args.output_dir / "paper_extension_summary",
        )
    print(
        f"Generated extension figures in {args.output_dir} "
        f"from {len(quality)} quality rows and {len(cost)} cost rows"
    )


if __name__ == "__main__":
    main()
