from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    hub_name: str
    text_field: str
    label_field: str
    label_names: tuple[str, ...]
    config_name: str | None = None


SPECS = {
    "sst2": DatasetSpec(
        "stanfordnlp/sst2", "sentence", "label", ("negative", "positive")
    ),
    "ag_news": DatasetSpec(
        "fancyzhx/ag_news",
        "text",
        "label",
        ("world", "sports", "business", "science/technology"),
    ),
    "trec": DatasetSpec(
        "CogComp/trec",
        "text",
        "coarse_label",
        ("abbreviation", "entity", "description", "human", "location", "numeric"),
    ),
}


def load_benchmark(name: str, split: str):
    from datasets import load_dataset

    if name not in SPECS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(SPECS)}")
    spec = SPECS[name]
    return load_dataset(spec.hub_name, spec.config_name, split=split), spec


def normalize(
    raw: dict[str, Any], spec: DatasetSpec, dataset: str, split: str, index: int
) -> dict[str, str]:
    label_id = int(raw[spec.label_field])
    if not 0 <= label_id < len(spec.label_names):
        raise ValueError(f"Unexpected label id {label_id} in {dataset}")
    return {
        "sample_id": f"{dataset}:{split}:{index}",
        "text": str(raw[spec.text_field]).strip(),
        "gold_label": spec.label_names[label_id],
    }

