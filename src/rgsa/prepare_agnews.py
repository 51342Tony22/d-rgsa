from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .prepare import write_jsonl


LABEL_NAMES = ("world", "sports", "business", "sci/tech")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a stratified labeled AG News 6K subset."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=4800)
    parser.add_argument("--validation-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_sample(
    labels: list[int],
    *,
    sample_size: int,
    seed: int,
    exclude: set[int] | None = None,
) -> list[int]:
    from sklearn.model_selection import train_test_split

    candidates = [index for index in range(len(labels)) if index not in (exclude or set())]
    if sample_size > len(candidates):
        raise ValueError(
            f"Requested {sample_size} examples, but only {len(candidates)} are available"
        )
    sampled, _ = train_test_split(
        candidates,
        train_size=sample_size,
        random_state=seed,
        stratify=[labels[index] for index in candidates],
    )
    return sorted(sampled)


def main() -> None:
    from datasets import load_dataset

    args = arguments()
    source = load_dataset("ag_news")
    train_source = source["train"]
    test_source = source["test"]
    train_labels = [int(value) for value in train_source["label"]]
    test_labels = [int(value) for value in test_source["label"]]

    train_indices = stratified_sample(
        train_labels,
        sample_size=args.train_size,
        seed=args.seed,
    )
    validation_indices = stratified_sample(
        train_labels,
        sample_size=args.validation_size,
        seed=args.seed + 1,
        exclude=set(train_indices),
    )
    test_indices = stratified_sample(
        test_labels,
        sample_size=args.test_size,
        seed=args.seed + 2,
    )
    split_sources = {
        "train": (train_source, train_indices, "official_train"),
        "validation": (train_source, validation_indices, "official_train"),
        "test": (test_source, test_indices, "official_test"),
    }
    manifest = {
        "dataset": "ag_news",
        "seed": args.seed,
        "label_names": list(LABEL_NAMES),
        "total_size": args.train_size + args.validation_size + args.test_size,
        "splits": {},
    }
    for split, (dataset, indices, source_split) in split_sources.items():
        records = [
            {
                "sample_id": f"agnews-6k:{split}:{position}",
                "source_idx": int(index),
                "source_split": source_split,
                "text": str(dataset[index]["text"]).strip(),
                "gold_label": LABEL_NAMES[int(dataset[index]["label"])],
            }
            for position, index in enumerate(indices)
        ]
        path = args.output_dir / f"{split}.jsonl"
        digest = write_jsonl(path, records)
        manifest["splits"][split] = {
            "file": path.name,
            "source_split": source_split,
            "size": len(records),
            "label_distribution": dict(
                sorted(Counter(row["gold_label"] for row in records).items())
            ),
            "sha256": digest,
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
