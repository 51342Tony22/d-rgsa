from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


DATASETS = {
    "SST-2": ("sst2-6k-preliminary", "sst2-6k"),
    "TREC": ("trec", "trec"),
    "AG News": ("agnews-6k", "agnews-6k"),
}
METHODS = {
    "full": "Full",
    "local": "Local",
    "random": "Random",
    "tfidf": "TF-IDF",
    "rationale": "Direct-RGSA",
    "distilled": "D-RGSA",
}
METHOD_ORDER = list(METHODS.values())
METHOD_COLORS = {
    "Full": "#4C78A8",
    "Local": "#F58518",
    "Random": "#54A24B",
    "TF-IDF": "#B279A2",
    "Direct-RGSA": "#E45756",
    "D-RGSA": "#2E8B8B",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Altair figures for Direct-RGSA versus D-RGSA."
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--baseline-root",
        type=Path,
        help="Project containing the frozen Full/Local/Random/TF-IDF/Direct-RGSA runs.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87, 101, 123])
    return parser.parse_args()


def load_metrics(root: Path, seeds: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in seeds:
        path = root / f"seed-{seed}" / "metrics.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise FileNotFoundError(f"No completed metrics under {root}")
    return records


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summary_row(
    dataset: str,
    method: str,
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    f1 = [100 * float(row["test_macro_f1"]) for row in metrics]
    accuracy = [100 * float(row["test_accuracy"]) for row in metrics]
    sparsity = [100 * float(row["test_sparsity"]) for row in metrics]
    f1_mean, f1_std = mean_std(f1)
    accuracy_mean, accuracy_std = mean_std(accuracy)
    sparsity_mean, sparsity_std = mean_std(sparsity)
    return {
        "dataset": dataset,
        "method": method,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "macro_f1_lower": f1_mean - f1_std,
        "macro_f1_upper": f1_mean + f1_std,
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "sparsity_mean": sparsity_mean,
        "sparsity_std": sparsity_std,
        "runs": len(metrics),
    }


def collect_main_rows(
    project_root: Path,
    baseline_root: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, (baseline_dir, distilled_dir) in DATASETS.items():
        for strategy in ("full", "local", "random", "tfidf", "rationale"):
            metrics = load_metrics(
                baseline_root / "outputs" / baseline_dir / strategy,
                seeds,
            )
            rows.append(summary_row(dataset, METHODS[strategy], metrics))
        distilled = load_metrics(
            project_root
            / "outputs"
            / "d-rgsa-pilot"
            / distilled_dir
            / "deepseek"
            / "bigru",
            seeds,
        )
        rows.append(summary_row(dataset, METHODS["distilled"], distilled))
    return rows


def save_chart(chart: Any, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    chart.save(str(output_base.with_suffix(".html")))
    for suffix in (".svg", ".png"):
        try:
            chart.save(str(output_base.with_suffix(suffix)), scale_factor=2)
        except TypeError:
            chart.save(str(output_base.with_suffix(suffix)))
        except Exception as exc:
            print(f"Skipped {suffix} for {output_base.name}: {exc}")


def main_macro_f1_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("method:N", sort=METHOD_ORDER, title=None),
        color=alt.Color(
            "method:N",
            scale=alt.Scale(
                domain=METHOD_ORDER,
                range=[METHOD_COLORS[name] for name in METHOD_ORDER],
            ),
            legend=None,
        ),
        tooltip=[
            "dataset:N",
            "method:N",
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            alt.Tooltip("accuracy_mean:Q", format=".2f", title="Accuracy"),
            alt.Tooltip("sparsity_mean:Q", format=".2f", title="Sparsity"),
            "runs:Q",
        ],
    )
    intervals = base.mark_rule(strokeWidth=2).encode(
        x=alt.X(
            "macro_f1_lower:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        ),
        x2="macro_f1_upper:Q",
    )
    points = base.mark_point(filled=True, size=105).encode(
        x=alt.X(
            "macro_f1_mean:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        )
    )
    return (
        (intervals + points)
        .properties(width=220, height=190)
        .facet(column=alt.Column("dataset:N", title=None, sort=list(DATASETS)))
        .resolve_scale(x="independent")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
    )


def gain_chart(rows: list[dict[str, Any]]) -> Any:
    import altair as alt

    lookup = {(row["dataset"], row["method"]): row["macro_f1_mean"] for row in rows}
    gain_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        distilled = lookup[(dataset, "D-RGSA")]
        for baseline in ["Full", "Local", "Random", "TF-IDF", "Direct-RGSA"]:
            gain_rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "gain": distilled - lookup[(dataset, baseline)],
                }
            )
    data = alt.Data(values=gain_rows)
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y(
                "baseline:N",
                sort=["Full", "Local", "Random", "TF-IDF", "Direct-RGSA"],
                title=None,
            ),
            x=alt.X("gain:Q", title="D-RGSA Macro-F1 difference (points)"),
            color=alt.condition(
                "datum.gain >= 0",
                alt.value("#2E8B8B"),
                alt.value("#D95F59"),
            ),
            tooltip=[
                "dataset:N",
                "baseline:N",
                alt.Tooltip("gain:Q", format="+.2f", title="Difference"),
            ],
        )
        .properties(width=220, height=170)
    )
    zero = alt.Chart(data).mark_rule(color="#666").encode(x=alt.datum(0))
    labels = (
        alt.Chart(data)
        .mark_text(dx=5, align="left", fontSize=11)
        .encode(
            y=alt.Y(
                "baseline:N",
                sort=["Full", "Local", "Random", "TF-IDF", "Direct-RGSA"],
                title=None,
            ),
            x="gain:Q",
            text=alt.Text("gain:Q", format="+.2f"),
        )
    )
    return (
        (bars + zero + labels)
        .facet(column=alt.Column("dataset:N", title=None, sort=list(DATASETS)))
        .resolve_scale(x="shared")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
    )


def selector_chart(project_root: Path) -> Any:
    import altair as alt

    rows: list[dict[str, Any]] = []
    for dataset, (_, directory) in DATASETS.items():
        path = (
            project_root
            / "outputs"
            / "distilled-selectors"
            / directory
            / "deepseek"
            / "bigru"
            / "seed-42"
            / "selector_metrics.json"
        )
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for metric, label in [
            ("anchor_f1", "Anchor F1"),
            ("anchor_precision", "Precision"),
            ("anchor_recall", "Recall"),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "metric": label,
                    "value": 100 * float(metrics["test"][metric]),
                }
            )
    return (
        alt.Chart(alt.Data(values=rows))
        .mark_bar()
        .encode(
            x=alt.X(
                "dataset:N",
                title=None,
                sort=list(DATASETS),
                axis=alt.Axis(labelAngle=0),
            ),
            xOffset=alt.XOffset("metric:N", sort=["Anchor F1", "Precision", "Recall"]),
            y=alt.Y("value:Q", title="Token-selection score (%)"),
            color=alt.Color(
                "metric:N",
                sort=["Anchor F1", "Precision", "Recall"],
                title=None,
            ),
            tooltip=[
                "dataset:N",
                "metric:N",
                alt.Tooltip("value:Q", format=".2f"),
            ],
        )
        .properties(
            width=560,
            height=280,
            title="How well the lightweight selector imitates teacher anchors",
        )
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_title(fontSize=16, anchor="start")
    )


def sst2_ablation_chart(
    project_root: Path,
    baseline_root: Path,
    seeds: list[int],
) -> Any:
    import altair as alt

    common_seeds = [seed for seed in seeds if seed in {13, 42, 87}]
    if not common_seeds:
        raise ValueError("SST-2 ablations require at least one of seeds 13, 42, 87")
    variants = {
        "Direct-RGSA": baseline_root
        / "outputs"
        / "sst2-6k-preliminary"
        / "rationale",
        "D-RGSA (BiGRU)": project_root
        / "outputs"
        / "d-rgsa-pilot"
        / "sst2-6k"
        / "deepseek"
        / "bigru",
        "D-RGSA (MLP)": project_root
        / "outputs"
        / "d-rgsa-pilot"
        / "sst2-6k"
        / "deepseek"
        / "mlp",
        "Shuffled anchors": project_root
        / "outputs"
        / "d-rgsa-ablations"
        / "sst2-6k"
        / "shuffled",
    }
    rows: list[dict[str, Any]] = []
    for label, path in variants.items():
        available = load_metrics(path, common_seeds)
        row = summary_row("SST-2", label, available)
        rows.append(row)
    order = list(variants)
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("method:N", sort=order, title=None),
        tooltip=[
            "method:N",
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            "runs:Q",
        ],
    )
    intervals = base.mark_rule(strokeWidth=2, color="#555").encode(
        x=alt.X(
            "macro_f1_lower:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        ),
        x2="macro_f1_upper:Q",
    )
    points = base.mark_point(filled=True, size=120, color="#2E8B8B").encode(
        x=alt.X(
            "macro_f1_mean:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        )
    )
    return (
        (intervals + points)
        .properties(
            width=430,
            height=180,
        )
        .configure_axis(labelFontSize=12, titleFontSize=13)
    )


def d_rgsa_hyperparameter_rows(
    project_root: Path,
    seeds: list[int],
) -> list[dict[str, Any]]:
    ablation_seeds = [seed for seed in seeds if seed in {13, 42, 87}]
    if not ablation_seeds:
        raise ValueError("Hyperparameter ablations require at least one of seeds 13, 42, 87")
    rows: list[dict[str, Any]] = []
    main_root = (
        project_root
        / "outputs"
        / "d-rgsa-pilot"
        / "sst2-6k"
        / "deepseek"
        / "bigru"
    )
    ablation_root = (
        project_root / "outputs" / "d-rgsa-ablations" / "sst2-6k"
    )
    for budget in [1, 2, 3, 4]:
        root = (
            main_root if budget == 4 else ablation_root / f"anchor-budget-{budget}"
        )
        metrics = load_metrics(root, ablation_seeds)
        values = [100 * float(row["test_macro_f1"]) for row in metrics]
        mean, std = mean_std(values)
        sparsity, _ = mean_std(
            [100 * float(row["test_sparsity"]) for row in metrics]
        )
        rows.append(
            {
                "panel": "Anchor budget",
                "setting": str(budget),
                "order": budget,
                "macro_f1_mean": mean,
                "macro_f1_std": std,
                "lower": mean - std,
                "upper": mean + std,
                "sparsity": sparsity,
            }
        )
    for radius in [2, 4, 8]:
        root = main_root if radius == 4 else ablation_root / f"window-radius-{radius}"
        metrics = load_metrics(root, ablation_seeds)
        values = [100 * float(row["test_macro_f1"]) for row in metrics]
        mean, std = mean_std(values)
        sparsity, _ = mean_std(
            [100 * float(row["test_sparsity"]) for row in metrics]
        )
        rows.append(
            {
                "panel": "Window radius",
                "setting": str(radius),
                "order": radius,
                "macro_f1_mean": mean,
                "macro_f1_std": std,
                "lower": mean - std,
                "upper": mean + std,
                "sparsity": sparsity,
            }
        )
    return rows


def d_rgsa_hyperparameter_chart(
    project_root: Path,
    seeds: list[int],
) -> Any:
    import altair as alt

    rows = d_rgsa_hyperparameter_rows(project_root, seeds)
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X(
            "setting:O",
            sort=alt.SortField("order"),
            title="Setting",
            axis=alt.Axis(labelAngle=0),
        ),
        tooltip=[
            "panel:N",
            "setting:O",
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            alt.Tooltip("sparsity:Q", format=".2f", title="Sparsity"),
        ],
    )
    line = base.mark_line(point=alt.OverlayMarkDef(filled=True, size=85)).encode(
        y=alt.Y(
            "macro_f1_mean:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        ),
        color=alt.value("#2E8B8B"),
    )
    error = base.mark_errorbar(ticks=True, color="#444").encode(
        y=alt.Y("lower:Q", title="Macro-F1 (%)"),
        y2="upper:Q",
    )
    return (
        (line + error)
        .properties(width=310, height=220)
        .facet(column=alt.Column("panel:N", title=None))
        .resolve_scale(x="independent", y="shared")
        .properties(title="D-RGSA sensitivity to anchor budget and local window")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
        .configure_title(fontSize=16, anchor="start")
    )


def d_rgsa_hyperparameter_panel_chart(
    project_root: Path,
    seeds: list[int],
    panel: str,
) -> Any:
    import altair as alt

    rows = [
        row
        for row in d_rgsa_hyperparameter_rows(project_root, seeds)
        if row["panel"] == panel
    ]
    if not rows:
        raise ValueError(f"No D-RGSA hyperparameter rows for {panel}")

    x_title = (
        "Anchor budget k"
        if panel == "Anchor budget"
        else "Local window radius r"
    )
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X(
            "setting:O",
            sort=alt.SortField("order"),
            title=x_title,
            axis=alt.Axis(labelAngle=0),
        ),
        tooltip=[
            alt.Tooltip("setting:O", title="Setting"),
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            alt.Tooltip("sparsity:Q", format=".2f", title="Logical sparsity"),
        ],
    )
    f1_line = base.mark_line(
        point=alt.OverlayMarkDef(
            filled=True,
            size=90,
            color="#2E8B8B",
        ),
        strokeWidth=2.4,
        color="#2E8B8B",
    ).encode(
        y=alt.Y(
            "macro_f1_mean:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False, padding=18),
            axis=alt.Axis(
                titleColor="#2E8B8B",
                labelColor="#2E8B8B",
                tickColor="#2E8B8B",
            ),
        )
    )
    error = base.mark_errorbar(
        ticks=True,
        thickness=1.5,
        color="#333333",
    ).encode(
        y=alt.Y(
            "lower:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False, padding=18),
            axis=alt.Axis(
                titleColor="#2E8B8B",
                labelColor="#2E8B8B",
                tickColor="#2E8B8B",
            ),
        ),
        y2="upper:Q",
    )
    sparsity_line = base.mark_line(
        point=alt.OverlayMarkDef(
            filled=True,
            size=70,
            color="#E17C05",
        ),
        strokeWidth=2,
        strokeDash=[5, 3],
        color="#E17C05",
    ).encode(
        y=alt.Y(
            "sparsity:Q",
            title="Logical sparsity (%)",
            scale=alt.Scale(zero=False, padding=18),
            axis=alt.Axis(
                orient="right",
                titleColor="#E17C05",
                labelColor="#E17C05",
                tickColor="#E17C05",
                grid=False,
            ),
        )
    )
    f1_layer = alt.layer(f1_line, error)
    return (
        alt.layer(f1_layer, sparsity_line)
        .resolve_scale(y="independent")
        .properties(width=310, height=225)
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_view(stroke="#D9D9D9")
    )


def teacher_sensitivity_chart(
    project_root: Path,
    baseline_root: Path,
    seeds: list[int],
) -> Any | None:
    import altair as alt

    rows: list[dict[str, Any]] = []
    for dataset, (baseline_dir, distilled_dir) in DATASETS.items():
        qwen_direct = (
            project_root
            / "outputs"
            / "qwen-teacher"
            / distilled_dir
            / "direct"
        )
        qwen_distilled = (
            project_root
            / "outputs"
            / "qwen-teacher"
            / distilled_dir
            / "distilled"
        )
        qwen_seeds = [
            seed
            for seed in seeds
            if (qwen_direct / f"seed-{seed}" / "metrics.json").exists()
            and (qwen_distilled / f"seed-{seed}" / "metrics.json").exists()
        ]
        if not qwen_seeds:
            continue
        comparison_seeds = qwen_seeds
        roots = {
            ("DeepSeek", "Direct-RGSA"): baseline_root
            / "outputs"
            / baseline_dir
            / "rationale",
            ("DeepSeek", "D-RGSA"): project_root
            / "outputs"
            / "d-rgsa-pilot"
            / distilled_dir
            / "deepseek"
            / "bigru",
            ("Qwen", "Direct-RGSA"): qwen_direct,
            ("Qwen", "D-RGSA"): qwen_distilled,
        }
        for (teacher, method), root in roots.items():
            try:
                metrics = load_metrics(root, comparison_seeds)
            except FileNotFoundError:
                continue
            values = [100 * float(row["test_macro_f1"]) for row in metrics]
            mean, std = mean_std(values)
            rows.append(
                {
                    "dataset": dataset,
                    "teacher": teacher,
                    "method": method,
                    "series": f"{teacher} → {method}",
                    "macro_f1_mean": mean,
                    "macro_f1_std": std,
                    "lower": mean - std,
                    "upper": mean + std,
                    "runs": len(values),
                }
            )
    if not any(row["teacher"] == "Qwen" for row in rows):
        return None
    order = [
        "DeepSeek → Direct-RGSA",
        "DeepSeek → D-RGSA",
        "Qwen → Direct-RGSA",
        "Qwen → D-RGSA",
    ]
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("series:N", sort=order, title=None),
        color=alt.Color(
            "teacher:N",
            scale=alt.Scale(
                domain=["DeepSeek", "Qwen"],
                range=["#4C78A8", "#E45756"],
            ),
            title="Teacher",
        ),
        tooltip=[
            "dataset:N",
            "teacher:N",
            "method:N",
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            "runs:Q",
        ],
    )
    interval = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("lower:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
        x2="upper:Q",
    )
    points = base.mark_point(filled=True, size=100).encode(
        x=alt.X(
            "macro_f1_mean:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        )
    )
    return (
        (interval + points)
        .properties(width=245, height=150)
        .facet(column=alt.Column("dataset:N", title=None, sort=list(DATASETS)))
        .resolve_scale(x="independent")
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_header(labelFontSize=14)
    )


def teacher_correctness_ablation_chart(
    project_root: Path,
    seeds: list[int],
) -> Any | None:
    import altair as alt
    from sklearn.metrics import f1_score

    rows: list[dict[str, Any]] = []
    for method_dir, method_label in [
        ("direct", "Direct-RGSA"),
        ("distilled", "D-RGSA"),
    ]:
        root = (
            project_root
            / "outputs"
            / "qwen-teacher"
            / "sst2-6k"
            / method_dir
        )
        grouped: dict[bool, list[float]] = {True: [], False: []}
        counts: dict[bool, int] = {True: 0, False: 0}
        for seed in seeds:
            path = root / f"seed-{seed}" / "test_predictions.jsonl"
            if not path.exists():
                continue
            predictions = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for teacher_correct in (True, False):
                subset = [
                    row
                    for row in predictions
                    if row.get("teacher_correct") is teacher_correct
                ]
                if not subset:
                    continue
                grouped[teacher_correct].append(
                    100
                    * float(
                        f1_score(
                            [row["gold_label"] for row in subset],
                            [row["predicted_label"] for row in subset],
                            average="macro",
                        )
                    )
                )
                counts[teacher_correct] = len(subset)
        for teacher_correct, label in [
            (True, "Teacher correct"),
            (False, "Teacher wrong"),
        ]:
            values = grouped[teacher_correct]
            if not values:
                continue
            mean, std = mean_std(values)
            rows.append(
                {
                    "group": label,
                    "method": method_label,
                    "macro_f1_mean": mean,
                    "macro_f1_std": std,
                    "lower": mean - std,
                    "upper": mean + std,
                    "samples": counts[teacher_correct],
                    "runs": len(values),
                }
            )
    if not rows:
        return None
    order = ["Teacher correct", "Teacher wrong"]
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("group:N", sort=order, title=None),
        yOffset=alt.YOffset(
            "method:N", sort=["Direct-RGSA", "D-RGSA"]
        ),
        color=alt.Color(
            "method:N",
            scale=alt.Scale(
                domain=["Direct-RGSA", "D-RGSA"],
                range=["#E45756", "#2E8B8B"],
            ),
            title=None,
        ),
        tooltip=[
            "group:N",
            "method:N",
            alt.Tooltip("macro_f1_mean:Q", format=".2f", title="Macro-F1"),
            alt.Tooltip("macro_f1_std:Q", format=".2f", title="Std"),
            "samples:Q",
            "runs:Q",
        ],
    )
    interval = base.mark_rule(strokeWidth=2).encode(
        x=alt.X("lower:Q", title="Macro-F1 (%)", scale=alt.Scale(zero=False)),
        x2="upper:Q",
    )
    point = base.mark_point(filled=True, size=110).encode(
        x=alt.X(
            "macro_f1_mean:Q",
            title="Macro-F1 (%)",
            scale=alt.Scale(zero=False),
        )
    )
    return (
        (interval + point)
        .properties(
            width=500,
            height=150,
            title="SST-2: behavior when the Qwen teacher is correct or wrong",
        )
        .configure_axis(labelFontSize=12, titleFontSize=13)
        .configure_title(fontSize=16, anchor="start")
    )


def write_tables(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "dataset",
        "method",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "accuracy_std",
        "sparsity_mean",
        "sparsity_std",
        "runs",
    ]
    with (output_dir / "main_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Dataset | Method | Macro-F1 (%) | Accuracy (%) | Sparsity (%) | Runs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | "
            f"{row['macro_f1_mean']:.2f} ± {row['macro_f1_std']:.2f} | "
            f"{row['accuracy_mean']:.2f} ± {row['accuracy_std']:.2f} | "
            f"{row['sparsity_mean']:.2f} ± {row['sparsity_std']:.2f} | "
            f"{row['runs']} |"
        )
    (output_dir / "main_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = arguments()
    project_root = args.project_root.resolve()
    baseline_root = (args.baseline_root or project_root).resolve()
    output_dir = (
        args.output_dir or project_root / "outputs" / "distilled-altair-figures"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_main_rows(project_root, baseline_root, args.seeds)
    write_tables(rows, output_dir)
    (output_dir / "main_results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_chart(main_macro_f1_chart(rows), output_dir / "main_macro_f1")
    save_chart(gain_chart(rows), output_dir / "d_rgsa_gain")
    save_chart(selector_chart(project_root), output_dir / "selector_quality")
    save_chart(
        sst2_ablation_chart(project_root, baseline_root, args.seeds),
        output_dir / "sst2_ablation",
    )
    save_chart(
        d_rgsa_hyperparameter_chart(project_root, args.seeds),
        output_dir / "d_rgsa_hyperparameters",
    )
    save_chart(
        d_rgsa_hyperparameter_panel_chart(
            project_root, args.seeds, "Anchor budget"
        ),
        output_dir / "d_rgsa_anchor_budget",
    )
    save_chart(
        d_rgsa_hyperparameter_panel_chart(
            project_root, args.seeds, "Window radius"
        ),
        output_dir / "d_rgsa_window_radius",
    )
    teacher_chart = teacher_sensitivity_chart(
        project_root, baseline_root, args.seeds
    )
    if teacher_chart is not None:
        save_chart(teacher_chart, output_dir / "teacher_sensitivity")
    correctness_chart = teacher_correctness_ablation_chart(
        project_root, args.seeds
    )
    if correctness_chart is not None:
        save_chart(
            correctness_chart,
            output_dir / "teacher_correctness_ablation",
        )
    print(f"Saved distilled RGSA figures to {output_dir}")


if __name__ == "__main__":
    main()
