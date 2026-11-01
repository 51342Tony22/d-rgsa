from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .alignment import rationale_token_positions
from .attention import STRATEGIES, logical_pattern


STRATEGY_ORDER = ["full", "local", "random", "tfidf", "rationale"]
STRATEGY_LABELS = {
    "full": "Full",
    "local": "Local",
    "random": "Random",
    "tfidf": "TF-IDF",
    "rationale": "RGSA",
}
STRATEGY_COLORS = {
    "Full": "#4C78A8",
    "Local": "#F58518",
    "Random": "#54A24B",
    "TF-IDF": "#B279A2",
    "RGSA": "#E45756",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cleaner Altair figures for paper analysis."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Project root containing data/ and outputs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to PROJECT_ROOT/outputs/altair-figures.",
    )
    parser.add_argument("--tokenizer-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--max-heatmap-length", type=int, default=64)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def save_chart(chart: Any, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    html = output_base.with_suffix(".html")
    chart.save(str(html))
    try:
        chart.save(str(output_base.with_suffix(".svg")))
    except Exception as exc:
        print(f"Skipped SVG for {output_base.name}: {exc}")
    print(f"Saved {html}")


def write_paper_main_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "strategy",
        "macro_f1",
        "accuracy",
        "sparsity",
        "runs",
    ]
    table_rows: list[dict[str, str | int]] = []
    for row in rows:
        table_rows.append(
            {
                "dataset": str(row["dataset"]),
                "strategy": str(row["strategy_label"]),
                "macro_f1": f"{float(row['macro_f1_mean']):.2f} ± {float(row['macro_f1_std']):.2f}",
                "accuracy": f"{float(row['accuracy_mean']):.2f} ± {float(row['accuracy_std']):.2f}",
                "sparsity": f"{float(row['sparsity_mean']):.2f} ± {float(row['sparsity_std']):.2f}",
                "runs": int(row["runs"]),
            }
        )
    with (output_dir / "paper_main_results_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    lines = [
        "| Dataset | Strategy | Macro-F1 (%) | Accuracy (%) | Sparsity (%) | Runs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | {row['macro_f1']} | "
            f"{row['accuracy']} | {row['sparsity']} | {row['runs']} |"
        )
    (output_dir / "paper_main_results_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def paper_macro_f1_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(rows)
    df["lower"] = df["macro_f1_mean"] - df["macro_f1_std"]
    df["upper"] = df["macro_f1_mean"] + df["macro_f1_std"]
    order = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    base = alt.Chart(df).encode(
        x=alt.X(
            "strategy_label:N",
            sort=order,
            title="Attention pattern",
            axis=alt.Axis(labelAngle=0),
        ),
        color=alt.Color(
            "strategy_label:N",
            scale=alt.Scale(domain=list(STRATEGY_COLORS), range=list(STRATEGY_COLORS.values())),
            legend=None,
        ),
        tooltip=[
            "dataset:N",
            "strategy_label:N",
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            alt.Tooltip("accuracy_mean:Q", format=".2f", title="Accuracy"),
            alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
        ],
    )
    points = base.mark_point(size=95, filled=True).encode(
        y=alt.Y("macro_f1_mean:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False))
    )
    lines = base.mark_line(strokeWidth=2, opacity=0.65).encode(
        y=alt.Y("macro_f1_mean:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
        detail="dataset:N",
        color=alt.value("#666666"),
    )
    error = base.mark_errorbar(ticks=True).encode(
        y=alt.Y("lower:Q", title="Macro-F1 (%)"),
        y2="upper:Q",
        color=alt.value("#333333"),
    )
    return (
        (lines + points + error)
        .facet(column=alt.Column("dataset:N", title=None))
        .resolve_scale(y="shared")
        .properties(title="Main result: Macro-F1 by attention pattern")
    )


def read_summary(project_root: Path) -> list[dict[str, Any]]:
    dataset_paths = {
        "SST-2": project_root / "outputs/sst2-6k-preliminary/results.summary.json",
        "TREC": project_root / "outputs/trec/results.summary.json",
        "AG News": project_root / "outputs/agnews-6k/results.summary.json",
    }
    rows: list[dict[str, Any]] = []
    for dataset, path in dataset_paths.items():
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        for strategy in STRATEGY_ORDER:
            metrics = summary[strategy]
            rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "runs": int(metrics["runs"]),
                    "accuracy_mean": 100 * float(metrics["test_accuracy_mean"]),
                    "accuracy_std": 100 * float(metrics["test_accuracy_std"]),
                    "macro_f1_mean": 100 * float(metrics["test_macro_f1_mean"]),
                    "macro_f1_std": 100 * float(metrics["test_macro_f1_std"]),
                    "sparsity_mean": 100 * float(metrics["test_sparsity_mean"]),
                    "sparsity_std": 100 * float(metrics["test_sparsity_std"]),
                    "connectivity_mean": 100
                    * float(metrics["test_connectivity_ratio_mean"]),
                }
            )
    return rows


def result_bar_chart(rows: list[dict[str, Any]], metric: str, title: str) -> Any:
    import altair as alt
    import pandas as pd

    mean_field = f"{metric}_mean"
    std_field = f"{metric}_std"
    df = pd.DataFrame(rows)
    df["lower"] = df[mean_field] - df[std_field]
    df["upper"] = df[mean_field] + df[std_field]

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(
                "strategy_label:N",
                sort=[STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                title=None,
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(f"{mean_field}:Q", title=f"{title} (%)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "strategy_label:N",
                scale=alt.Scale(
                    domain=list(STRATEGY_COLORS),
                    range=list(STRATEGY_COLORS.values()),
                ),
                title="Strategy",
            ),
            tooltip=[
                "dataset:N",
                "strategy_label:N",
                alt.Tooltip(f"{mean_field}:Q", format=".2f", title=f"{title} mean"),
                alt.Tooltip(f"{std_field}:Q", format=".2f", title="Std"),
                alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
            ],
        )
        .properties(width=155, height=300)
    )
    error = (
        alt.Chart(df)
        .mark_errorbar(ticks=True)
        .encode(
            x=alt.X(
                "strategy_label:N",
                sort=[STRATEGY_LABELS[s] for s in STRATEGY_ORDER],
                title=None,
            ),
            y=alt.Y("lower:Q", title=f"{title} (%)"),
            y2="upper:Q",
        )
    )
    return (
        (bars + error)
        .facet(column=alt.Column("dataset:N", title=None))
        .resolve_scale(y="shared")
        .properties(title=title)
    )


def sparsity_f1_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(rows)
    chart = (
        alt.Chart(df)
        .mark_circle(size=120, opacity=0.9, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("sparsity_mean:Q", title="Logical sparsity (%)"),
            y=alt.Y("macro_f1_mean:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "strategy_label:N",
                scale=alt.Scale(
                    domain=list(STRATEGY_COLORS),
                    range=list(STRATEGY_COLORS.values()),
                ),
                title="Strategy",
            ),
            tooltip=[
                "dataset:N",
                "strategy_label:N",
                alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
                alt.Tooltip("macro_f1_std:Q", format=".2f", title="F1 std"),
                alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
                alt.Tooltip("connectivity_mean:Q", format=".2f", title="Connectivity"),
            ],
        )
        .properties(width=220, height=260)
    )
    labels = (
        alt.Chart(df)
        .mark_text(dx=8, dy=-7, fontSize=10)
        .encode(
            x="sparsity_mean:Q",
            y="macro_f1_mean:Q",
            text="strategy_label:N",
            color=alt.value("#333333"),
        )
        .properties(width=220, height=260)
    )
    return (
        (chart + labels)
        .facet(column=alt.Column("dataset:N", title=None))
        .properties(title="Quality–sparsity trade-off")
    )


def rgsa_gain_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt
    import pandas as pd

    lookup = {
        (row["dataset"], row["strategy"]): float(row["macro_f1_mean"]) for row in rows
    }
    gain_rows: list[dict[str, Any]] = []
    for dataset in ["SST-2", "TREC", "AG News"]:
        for baseline in ["full", "local", "random", "tfidf"]:
            gain_rows.append(
                {
                    "dataset": dataset,
                    "baseline": STRATEGY_LABELS[baseline],
                    "gain": lookup[(dataset, "rationale")] - lookup[(dataset, baseline)],
                }
            )
    df = pd.DataFrame(gain_rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("baseline:N", title="Baseline"),
            y=alt.Y("gain:Q", title="RGSA Macro-F1 gain (points)"),
            color=alt.condition(
                alt.datum.gain >= 0,
                alt.value("#4C78A8"),
                alt.value("#E45756"),
            ),
            column=alt.Column("dataset:N", title=None),
            tooltip=[
                "dataset:N",
                "baseline:N",
                alt.Tooltip("gain:Q", format="+.2f", title="Gain"),
            ],
        )
        .properties(width=155, height=270, title="RGSA gain over baselines")
    )


def teacher_quality_chart(project_root: Path) -> Any:
    import altair as alt
    import pandas as pd

    audit_paths = {
        "SST-2": project_root / "data/sst2-6k/teacher-audit.json",
        "TREC": project_root / "data/trec/teacher-audit.json",
        "AG News": project_root / "data/agnews-6k/teacher-audit.json",
    }
    rows: list[dict[str, Any]] = []
    for dataset, path in audit_paths.items():
        audit = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "metric": "Teacher/gold agreement",
                    "value": 100 * float(audit["teacher_gold_agreement"]),
                },
                {
                    "dataset": dataset,
                    "metric": "Anchor validity",
                    "value": 100 * float(audit["anchor_valid_rate"]),
                },
            ]
        )
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("dataset:N", title=None),
            y=alt.Y("value:Q", title="Rate (%)", scale=alt.Scale(domain=[75, 100])),
            color=alt.Color("metric:N", title="Metric"),
            xOffset="metric:N",
            tooltip=[
                "dataset:N",
                "metric:N",
                alt.Tooltip("value:Q", format=".2f", title="Rate"),
            ],
        )
        .properties(width=520, height=300, title="Teacher annotation quality")
    )


def ablation_rows(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = project_root / "outputs/sst2-6k-ablations/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    anchor_rows: list[dict[str, Any]] = []
    for budget, metrics in sorted(summary["anchor_budget"].items(), key=lambda item: int(item[0])):
        anchor_rows.append(
            {
                "budget": int(budget),
                "macro_f1_mean": 100 * float(metrics["test_macro_f1_mean"]),
                "macro_f1_std": 100 * float(metrics["test_macro_f1_std"]),
                "sparsity_mean": 100 * float(metrics["test_sparsity_mean"]),
                "runs": int(metrics["runs"]),
            }
        )
    radius_rows: list[dict[str, Any]] = []
    for radius, variants in sorted(summary["window_radius"].items(), key=lambda item: int(item[0])):
        for strategy, metrics in variants.items():
            radius_rows.append(
                {
                    "radius": int(radius),
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "macro_f1_mean": 100 * float(metrics["test_macro_f1_mean"]),
                    "macro_f1_std": 100 * float(metrics["test_macro_f1_std"]),
                    "sparsity_mean": 100 * float(metrics["test_sparsity_mean"]),
                    "runs": int(metrics["runs"]),
                }
            )
    return anchor_rows, radius_rows


def anchor_budget_ablation_chart(anchor_rows: list[dict[str, Any]]) -> Any:
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(anchor_rows)
    df["lower"] = df["macro_f1_mean"] - df["macro_f1_std"]
    df["upper"] = df["macro_f1_mean"] + df["macro_f1_std"]
    f1 = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2, color="#E45756")
        .encode(
            x=alt.X("budget:O", title="Rationale anchor budget k"),
            y=alt.Y("macro_f1_mean:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
            tooltip=[
                "budget:O",
                alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
                alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
                alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
            ],
        )
        .properties(width=390, height=260, title="Anchor budget ablation: Macro-F1")
    )
    err = (
        alt.Chart(df)
        .mark_errorbar(ticks=True, color="#333333")
        .encode(x="budget:O", y="lower:Q", y2="upper:Q")
    )
    sparsity = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2, color="#4C78A8")
        .encode(
            x=alt.X("budget:O", title="Rationale anchor budget k"),
            y=alt.Y("sparsity_mean:Q", title="Sparsity (%)", scale=alt.Scale(zero=False)),
            tooltip=[
                "budget:O",
                alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
                alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            ],
        )
        .properties(width=390, height=220, title="Anchor budget ablation: sparsity")
    )
    return (f1 + err) & sparsity


def window_radius_ablation_chart(radius_rows: list[dict[str, Any]]) -> Any:
    import altair as alt
    import pandas as pd

    df = pd.DataFrame(radius_rows)
    df["lower"] = df["macro_f1_mean"] - df["macro_f1_std"]
    df["upper"] = df["macro_f1_mean"] + df["macro_f1_std"]
    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("radius:O", title="Local window radius r"),
            y=alt.Y("macro_f1_mean:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "strategy_label:N",
                scale=alt.Scale(domain=["Local", "RGSA"], range=["#F58518", "#E45756"]),
                title="Strategy",
            ),
            tooltip=[
                "radius:O",
                "strategy_label:N",
                alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
                alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
                alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
            ],
        )
        .properties(width=450, height=300, title="Window-radius ablation on SST-2")
    )
    err = (
        alt.Chart(df)
        .mark_errorbar(ticks=True)
        .encode(
            x="radius:O",
            y="lower:Q",
            y2="upper:Q",
            color=alt.Color(
                "strategy_label:N",
                scale=alt.Scale(domain=["Local", "RGSA"], range=["#F58518", "#E45756"]),
                legend=None,
            ),
        )
    )
    gains: list[dict[str, Any]] = []
    for radius in sorted(set(df["radius"])):
        local = float(df[(df["radius"] == radius) & (df["strategy"] == "local")]["macro_f1_mean"].iloc[0])
        rgsa = float(df[(df["radius"] == radius) & (df["strategy"] == "rationale")]["macro_f1_mean"].iloc[0])
        gains.append({"radius": int(radius), "gain": rgsa - local})
    gain_df = pd.DataFrame(gains)
    gain_chart = (
        alt.Chart(gain_df)
        .mark_bar(color="#4C78A8", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("radius:O", title="Local window radius r"),
            y=alt.Y("gain:Q", title="RGSA gain over Local (F1 points)"),
            tooltip=["radius:O", alt.Tooltip("gain:Q", format="+.2f", title="Gain")],
        )
        .properties(width=450, height=220, title="RGSA gain shrinks as local window grows")
    )
    return (chart + err) & gain_chart


def schematic_pattern_overview() -> Any:
    import altair as alt
    import pandas as pd

    n = 32
    local_radius = 2
    anchor_sets = {
        "Full": [],
        "Local": [],
        "Random": [5, 17, 27],
        "TF-IDF": [8, 15, 24],
        "RGSA": [3, 13, 22],
    }
    rows: list[dict[str, Any]] = []
    for strategy_label, anchors in anchor_sets.items():
        for query in range(n):
            for key in range(n):
                if strategy_label == "Full":
                    attends = True
                else:
                    attends = abs(query - key) <= local_radius
                    if strategy_label in {"Random", "TF-IDF", "RGSA"}:
                        attends = attends or key in anchors or query in anchors
                rows.append(
                    {
                        "strategy": strategy_label,
                        "query": query,
                        "key": key,
                        "attends": int(attends),
                        "query_anchor": query in anchors,
                        "key_anchor": key in anchors,
                    }
                )
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("key:O", title="Key token index", axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y(
                "query:O",
                title="Query token index",
                sort="ascending",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            color=alt.Color(
                "attends:O",
                scale=alt.Scale(domain=[0, 1], range=["#F1F1F1", "#2F6DB3"]),
                legend=alt.Legend(title="Attend"),
            ),
            column=alt.Column(
                "strategy:N",
                sort=["Full", "Local", "Random", "TF-IDF", "RGSA"],
                title=None,
            ),
            tooltip=[
                "strategy:N",
                "query:O",
                "key:O",
                "attends:O",
                "query_anchor:N",
                "key_anchor:N",
            ],
        )
        .properties(
            width=145,
            height=145,
        )
    )


def heatmap_rows(
    input_path: Path,
    *,
    strategy: str,
    sample_index: int,
    tokenizer_name: str,
    max_length: int,
    window_radius: int,
    anchor_budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    from transformers import AutoTokenizer

    with input_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    row = rows[sample_index]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoded = tokenizer(
        row["text"],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    source = row.get("rationale_tokens", []) if strategy == "rationale" else []
    if strategy == "tfidf":
        source = row.get("tfidf_tokens", [])
    positions = rationale_token_positions(row["text"], source, offsets)[:anchor_budget]
    anchors = [index in positions for index in range(len(encoded["input_ids"]))]
    valid = [bool(x) for x in encoded["attention_mask"]]
    graph = logical_pattern(
        valid, anchors, strategy=strategy, window_radius=window_radius
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    valid_indices = [index for index, keep in enumerate(valid) if keep]
    cells: list[dict[str, Any]] = []
    for query in valid_indices:
        for key in valid_indices:
            cells.append(
                {
                    "query_index": query,
                    "key_index": key,
                    "attends": int(graph[query][key]),
                    "query_token": tokens[query],
                    "key_token": tokens[key],
                    "query_anchor": anchors[query],
                    "key_anchor": anchors[key],
                }
            )
    token_rows = [
        {
            "index": index,
            "token": tokens[index],
            "anchor": anchors[index],
        }
        for index in valid_indices
    ]
    return cells, token_rows, row["text"]


def attention_heatmap(cells: list[dict[str, Any]], token_rows: list[dict[str, Any]], title: str) -> Any:
    import altair as alt
    import pandas as pd

    cell_df = pd.DataFrame(cells)
    token_df = pd.DataFrame(token_rows)
    n_tokens = max(1, len(token_rows))
    size = max(340, min(760, n_tokens * 12))
    base = (
        alt.Chart(cell_df)
        .mark_rect()
        .encode(
            x=alt.X("key_index:O", title="Key token index", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("query_index:O", title="Query token index", sort="ascending"),
            color=alt.Color(
                "attends:O",
                scale=alt.Scale(domain=[0, 1], range=["#F2F2F2", "#2F6DB3"]),
                legend=alt.Legend(title="Attend"),
            ),
            tooltip=[
                alt.Tooltip("query_index:O", title="Query index"),
                alt.Tooltip("query_token:N", title="Query token"),
                alt.Tooltip("key_index:O", title="Key index"),
                alt.Tooltip("key_token:N", title="Key token"),
                alt.Tooltip("attends:O", title="Attend"),
                alt.Tooltip("query_anchor:N", title="Query anchor"),
                alt.Tooltip("key_anchor:N", title="Key anchor"),
            ],
        )
        .properties(width=size, height=size, title=title)
    )
    anchors_x = (
        alt.Chart(token_df)
        .mark_tick(color="#E45756", thickness=2)
        .encode(
            x=alt.X("index:O", title=None, axis=None),
            y=alt.value(5),
            opacity=alt.condition(alt.datum.anchor, alt.value(1.0), alt.value(0.0)),
            tooltip=["index:O", "token:N", "anchor:N"],
        )
        .properties(width=size, height=14)
    )
    anchors_y = (
        alt.Chart(token_df)
        .mark_tick(color="#E45756", thickness=2)
        .encode(
            y=alt.Y("index:O", title=None, axis=None, sort="ascending"),
            x=alt.value(5),
            opacity=alt.condition(alt.datum.anchor, alt.value(1.0), alt.value(0.0)),
            tooltip=["index:O", "token:N", "anchor:N"],
        )
        .properties(width=14, height=size)
    )
    return (anchors_x & (anchors_y | base)).resolve_scale(color="independent")


def generate_attention_heatmaps(
    project_root: Path,
    output_dir: Path,
    *,
    tokenizer_name: str,
    max_length: int,
    sample_index: int,
) -> None:
    dataset_inputs = {
        "sst2-6k": project_root / "data/sst2-6k/test.ready.jsonl",
        "trec": project_root / "data/trec/test.ready.jsonl",
        "agnews-6k": project_root / "data/agnews-6k/test.ready.jsonl",
    }
    for dataset, input_path in dataset_inputs.items():
        if not input_path.exists():
            continue
        for strategy in STRATEGIES:
            cells, token_rows, _text = heatmap_rows(
                input_path,
                strategy=strategy,
                sample_index=sample_index,
                tokenizer_name=tokenizer_name,
                max_length=max_length,
                window_radius=4,
                anchor_budget=4,
            )
            chart = attention_heatmap(
                cells,
                token_rows,
                f"{dataset} · {STRATEGY_LABELS[strategy]} attention pattern",
            )
            save_chart(chart, output_dir / f"attention_{dataset}_{strategy}")


def main() -> None:
    args = arguments()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else project_root / "outputs/altair-figures"
    )
    rows = read_summary(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_paper_main_table(rows, output_dir)
    (output_dir / "altair_summary_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_chart(paper_macro_f1_chart(rows), output_dir / "paper_main_macro_f1")
    save_chart(result_bar_chart(rows, "macro_f1", "Macro-F1"), output_dir / "macro_f1_bars")
    save_chart(result_bar_chart(rows, "accuracy", "Accuracy"), output_dir / "accuracy_bars")
    save_chart(sparsity_f1_chart(rows), output_dir / "quality_sparsity_tradeoff")
    save_chart(rgsa_gain_chart(rows), output_dir / "rgsa_gain")
    save_chart(teacher_quality_chart(project_root), output_dir / "teacher_quality")
    anchor_rows, radius_rows = ablation_rows(project_root)
    (output_dir / "ablation_anchor_budget_rows.json").write_text(
        json.dumps(anchor_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "ablation_window_radius_rows.json").write_text(
        json.dumps(radius_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    save_chart(anchor_budget_ablation_chart(anchor_rows), output_dir / "ablation_anchor_budget")
    save_chart(window_radius_ablation_chart(radius_rows), output_dir / "ablation_window_radius")
    save_chart(schematic_pattern_overview(), output_dir / "attention_pattern_overview")
    generate_attention_heatmaps(
        project_root,
        output_dir,
        tokenizer_name=args.tokenizer_name,
        max_length=args.max_heatmap_length,
        sample_index=args.sample_index,
    )
    print(f"Saved Altair figures to {output_dir}")


if __name__ == "__main__":
    main()
